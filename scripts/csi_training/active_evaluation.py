"""Paired duration metrics on aligned, independently labelled room intervals.

No hardware access, training, model promotion or truth authentication. The caller
must establish label provenance and frozen room splits outside this evaluator.
"""
import json
import math
import os
import random
import stat

MAX_BYTES = 4 * 1024 * 1024
STATES = {"empty", "occupied", "unknown"}


def number(value, name, minimum=0, maximum=1e12):
    if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"invalid {name}")
    return float(value)


def exact(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("unexpected schema fields")


def load(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as file:
        info = os.fstat(file.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES:
            raise ValueError("input must be bounded regular file")
        raw = file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("input exceeds limit")
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _id(value):
    if not isinstance(value, str) or not value or len(value) > 128 or not value.isascii() or any(ord(c) < 32 for c in value):
        raise ValueError("invalid identifier")
    return value


def evaluate(document, resamples=1000, seed=42):
    exact(document, ("schemaVersion", "evidence", "sessions"))
    if type(document["schemaVersion"]) is not int or document["schemaVersion"] != 1 or document["evidence"] not in ("SYNTHETIC", "MEASURED"):
        raise ValueError("unsupported schema or evidence label")
    sessions = document["sessions"]
    if not isinstance(sessions, list) or not 1 <= len(sessions) <= 1000:
        raise ValueError("invalid session count")
    if type(resamples) is not int or not 100 <= resamples <= 10000 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("invalid bootstrap settings")
    rooms, identities, interval_count = {}, set(), 0
    for session in sessions:
        exact(session, ("room", "session", "budget", "intervals"))
        room, sid = _id(session["room"]), _id(session["session"])
        if sid in identities:
            raise ValueError("duplicate session identifier")
        identities.add(sid)
        budget = session["budget"]
        exact(budget, ("baseline", "candidate"))
        for side in budget.values():
            exact(side, ("packets", "airtime_ms", "energy_mj", "available_sensors"))
            if type(side["packets"]) is not int:
                raise ValueError("packet count must be integer")
            for dimension in ("packets", "airtime_ms", "energy_mj"):
                number(side[dimension], dimension)
            sensors = side["available_sensors"]
            if not isinstance(sensors, list) or not 1 <= len(sensors) <= 64:
                raise ValueError("invalid availability")
            if len({_id(sensor) for sensor in sensors}) != len(sensors):
                raise ValueError("duplicate sensor")
        if set(budget["baseline"]["available_sensors"]) != set(budget["candidate"]["available_sensors"]):
            raise ValueError("unequal sensor availability")
        for dimension in ("packets", "airtime_ms", "energy_mj"):
            if budget["candidate"][dimension] != budget["baseline"][dimension]:
                raise ValueError("unequal sensing budget")
        intervals = session["intervals"]
        if not isinstance(intervals, list) or not intervals:
            raise ValueError("missing intervals")
        interval_count += len(intervals)
        if interval_count > 100000:
            raise ValueError("too many intervals")
        # empty, occupied, total; per policy false, hit, unknown, unknown-empty, unknown-occupied
        totals = rooms.setdefault(room, [0.] * 13)
        previous = None
        for interval in intervals:
            exact(interval, ("start_ms", "end_ms", "truth", "baseline", "candidate"))
            start = number(interval["start_ms"], "start")
            end = number(interval["end_ms"], "end")
            if end <= start or (previous is not None and start != previous):
                raise ValueError("intervals must be increasing and contiguous")
            previous = end
            if interval["truth"] not in ("empty", "occupied"):
                raise ValueError("independent binary truth required")
            duration = (end - start) / 1000
            occupied = interval["truth"] == "occupied"
            totals[int(occupied)] += duration
            totals[2] += duration
            for j, side in enumerate(("baseline", "candidate")):
                prediction = interval[side]
                if not isinstance(prediction, str) or prediction not in STATES:
                    raise ValueError("invalid prediction")
                offset = 3 + j * 5
                totals[offset] += duration * (not occupied and prediction == "occupied")
                totals[offset+1] += duration * (occupied and prediction == "occupied")
                totals[offset+2] += duration * (prediction == "unknown")
                totals[offset+3] += duration * (prediction == "unknown" and not occupied)
                totals[offset+4] += duration * (prediction == "unknown" and occupied)

    def summarize(values):
        total = [sum(row[i] for row in values) for i in range(13)]
        empty, occupied, duration = total[:3]
        if empty == 0 or occupied == 0:
            return None
        bfalse, bhit, bunk, bue, buo, cfalse, chit, cunk, cue, cuo = total[3:]
        return {"empty_seconds": empty, "occupied_seconds": occupied,
                "baseline_false_seconds_per_empty_hour": bfalse / empty * 3600,
                "candidate_false_seconds_per_empty_hour": cfalse / empty * 3600,
                "baseline_recall": bhit / occupied, "candidate_recall": chit / occupied,
                "baseline_unknown_fraction": bunk / duration,
                "candidate_unknown_fraction": cunk / duration,
                "false_duration_relative_reduction": None if bfalse == 0 else 1-cfalse/bfalse,
                "recall_delta": (chit-bhit)/occupied,
                "unknown_empty_delta": (cue-bue)/empty,
                "unknown_occupied_delta": (cuo-buo)/occupied}

    point = summarize(list(rooms.values()))
    if point is None:
        raise ValueError("both empty and occupied exposure required")
    rng = random.Random(seed)
    clusters = list(rooms.values())
    reduction, recall, unknown_empty, unknown_occupied = [], [], [], []
    for _ in range(resamples):
        sample = summarize([clusters[rng.randrange(len(clusters))] for _ in clusters])
        if sample is not None:
            recall.append(sample["recall_delta"])
            unknown_empty.append(sample["unknown_empty_delta"])
            unknown_occupied.append(sample["unknown_occupied_delta"])
            if sample["false_duration_relative_reduction"] is not None:
                reduction.append(sample["false_duration_relative_reduction"])
    def interval(values):
        ordered = sorted(values)
        return [ordered[int((len(ordered)-1)*q)] for q in (.025, .975)] if ordered else None
    rci, dci = interval(reduction), interval(recall)
    eci, oci = interval(unknown_empty), interval(unknown_occupied)
    evaluable = len(rooms) >= 2 and len(reduction) >= .95 * resamples and point["false_duration_relative_reduction"] is not None
    passes = bool(evaluable and point["false_duration_relative_reduction"] >= .20 and rci[0] > 0 and dci[0] >= -.01 and eci[1] <= .01 and oci[1] <= .01)
    return {"schemaVersion": 1, "inputDeclaredEvidence": document["evidence"],
            "labelProvenanceVerified": False, "promotionAuthorized": False,
            "rooms": len(rooms), "sessions": len(sessions), "intervals": interval_count,
            "metrics": point, "bootstrap": {"cluster": "room", "seed": seed,
            "resamples": resamples, "valid": len(reduction), "confidence": .95,
            "reduction_interval": rci, "recall_delta_interval": dci,
            "recall_valid": len(recall), "unknown_empty_delta_interval": eci,
            "unknown_occupied_delta_interval": oci},
            "numericalGate": "pass" if passes else "fail" if evaluable else "inconclusive",
            "fieldAcceptance": "not_established",
            "limitations": ["Declared labels and budget costs are not independently authenticated",
                            "Fresh room splits and capture completeness require external review",
                            "Bootstrap intervals on few rooms may be unstable; size a confirmation study",
                            "Event counts and detection delay require separate transition evaluation"]}
