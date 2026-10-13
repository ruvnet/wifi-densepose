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


TRANSITIONS = ("empty_to_occupied", "occupied_to_empty")
SIDES = ("baseline", "candidate")


def _new_transition_totals():
    return {"empty_seconds": 0.0,
            "baseline_false_episodes": 0, "candidate_false_episodes": 0,
            "empty_to_occupied": [0, 0.0, 0.0, 0, 0],
            "occupied_to_empty": [0, 0.0, 0.0, 0, 0]}


def _collect_transitions(total, intervals):
    previous_truth = None
    previous_predictions = {}
    for start, end, truth, baseline, candidate in intervals:
        if truth == "empty":
            total["empty_seconds"] += (end-start)/1000
        for side, prediction in zip(SIDES, (baseline, candidate)):
            if (truth == "empty" and prediction == "occupied"
                    and (previous_truth != "empty"
                         or previous_predictions.get(side) != "occupied")):
                total[f"{side}_false_episodes"] += 1
            previous_predictions[side] = prediction
        previous_truth = truth

    run_start = 0
    while run_start < len(intervals):
        truth = intervals[run_start][2]
        run_stop = run_start + 1
        while run_stop < len(intervals) and intervals[run_stop][2] == truth:
            run_stop += 1
        if run_start:
            direction = f"{intervals[run_start-1][2]}_to_{truth}"
            values = total[direction]
            values[0] += 1
            start_ms = intervals[run_start][0]
            end_ms = intervals[run_stop-1][1]
            for side_index, tuple_index in enumerate((3, 4)):
                matched = next((row[0] for row in intervals[run_start:run_stop]
                                if row[tuple_index] == truth), None)
                if matched is None:
                    matched = end_ms
                    values[3+side_index] += 1
                values[1+side_index] += matched-start_ms
        run_start = run_stop


def _summarize_transitions(values):
    empty = sum(row["empty_seconds"] for row in values)
    if empty == 0:
        return None
    baseline_episodes = sum(row["baseline_false_episodes"] for row in values)
    candidate_episodes = sum(row["candidate_false_episodes"] for row in values)
    result = {
        "empty_seconds": empty,
        "baseline_false_episodes_per_empty_hour": baseline_episodes/empty*3600,
        "candidate_false_episodes_per_empty_hour": candidate_episodes/empty*3600,
        "false_episode_delta_per_empty_hour": (candidate_episodes-baseline_episodes)/empty*3600,
    }
    for direction in TRANSITIONS:
        totals = [sum(row[direction][i] for row in values) for i in range(5)]
        events, baseline_delay, candidate_delay, baseline_unresolved, candidate_unresolved = totals
        result[direction] = None if events == 0 else {
            "events": int(events),
            "baseline_mean_delay_ms": baseline_delay/events,
            "candidate_mean_delay_ms": candidate_delay/events,
            "delay_delta_ms": (candidate_delay-baseline_delay)/events,
            "baseline_unresolved_fraction": baseline_unresolved/events,
            "candidate_unresolved_fraction": candidate_unresolved/events,
            "unresolved_fraction_delta": (candidate_unresolved-baseline_unresolved)/events,
        }
    return result


def evaluate(document, resamples=1000, seed=42):
    exact(document, ("schemaVersion", "evidence", "sessions"))
    if type(document["schemaVersion"]) is not int or document["schemaVersion"] != 1 or document["evidence"] not in ("SYNTHETIC", "MEASURED"):
        raise ValueError("unsupported schema or evidence label")
    sessions = document["sessions"]
    if not isinstance(sessions, list) or not 1 <= len(sessions) <= 1000:
        raise ValueError("invalid session count")
    if type(resamples) is not int or not 100 <= resamples <= 10000 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("invalid bootstrap settings")
    rooms, transition_rooms, identities, interval_count = {}, {}, set(), 0
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
        transition_totals = transition_rooms.setdefault(room, _new_transition_totals())
        validated_intervals = []
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
            validated_intervals.append((start, end, interval["truth"],
                                        interval["baseline"], interval["candidate"]))
        _collect_transitions(transition_totals, validated_intervals)

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
    numerical_gate = "pass" if passes else "fail" if evaluable else "inconclusive"

    transition_point = _summarize_transitions(list(transition_rooms.values()))
    transition_clusters = list(transition_rooms.values())
    transition_values = {direction: {"delay": [], "unresolved": []}
                         for direction in TRANSITIONS}
    false_episodes = []
    transition_valid = 0
    for _ in range(resamples):
        sample = _summarize_transitions([
            transition_clusters[rng.randrange(len(transition_clusters))]
            for _ in transition_clusters])
        if sample is not None and all(sample[direction] is not None
                                      for direction in TRANSITIONS):
            transition_valid += 1
            false_episodes.append(sample["false_episode_delta_per_empty_hour"])
            for direction in TRANSITIONS:
                transition_values[direction]["delay"].append(
                    sample[direction]["delay_delta_ms"])
                transition_values[direction]["unresolved"].append(
                    sample[direction]["unresolved_fraction_delta"])
    transition_intervals = {
        direction: {
            "delay_delta_ms_interval": interval(transition_values[direction]["delay"]),
            "unresolved_fraction_delta_interval": interval(
                transition_values[direction]["unresolved"]),
        } for direction in TRANSITIONS}
    false_episode_interval = interval(false_episodes)
    transition_room_coverage = {
        direction: sum(row[direction][0] > 0 for row in transition_clusters)
        for direction in TRANSITIONS}
    transition_evaluable = bool(
        transition_point is not None
        and all(transition_room_coverage[direction] >= 2 for direction in TRANSITIONS)
        and transition_valid >= .95*resamples)
    transition_passes = bool(
        transition_evaluable and false_episode_interval[1] <= 0
        and all(transition_intervals[direction]["delay_delta_ms_interval"][1] <= 0
                and transition_intervals[direction]["unresolved_fraction_delta_interval"][1] <= 0
                for direction in TRANSITIONS))
    transition_gate = ("pass" if transition_passes else
                       "fail" if transition_evaluable else "inconclusive")
    overall_gate = ("fail" if "fail" in (numerical_gate, transition_gate) else
                    "pass" if numerical_gate == transition_gate == "pass" else
                    "inconclusive")
    return {"schemaVersion": 1, "inputDeclaredEvidence": document["evidence"],
            "labelProvenanceVerified": False, "promotionAuthorized": False,
            "rooms": len(rooms), "sessions": len(sessions), "intervals": interval_count,
            "metrics": point, "bootstrap": {"cluster": "room", "seed": seed,
            "resamples": resamples, "valid": len(reduction), "confidence": .95,
            "reduction_interval": rci, "recall_delta_interval": dci,
            "recall_valid": len(recall), "unknown_empty_delta_interval": eci,
            "unknown_occupied_delta_interval": oci},
            "numericalGate": numerical_gate,
            "transitionMetrics": transition_point,
            "transitionBootstrap": {"cluster": "room", "seed": seed,
                "resamples": resamples, "valid": transition_valid,
                "confidence": .95, "room_coverage": transition_room_coverage,
                "false_episode_delta_per_empty_hour_interval": false_episode_interval,
                **transition_intervals},
            "transitionGate": transition_gate, "overallGate": overall_gate,
            "fieldAcceptance": "not_established",
            "limitations": ["Declared labels and budget costs are not independently authenticated",
                            "Fresh room splits and capture completeness require external review",
                            "Bootstrap intervals on few rooms may be unstable; size a confirmation study",
                            "Transition timing is interval-aligned and excludes sub-interval, acquisition and actuation latency"]}
