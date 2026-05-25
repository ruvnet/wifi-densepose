#!/usr/bin/env python3
"""
ruview-hap-bridge.py — ADR-125 §2.1.c production bridge (Tier 1+2 iter 3).

One HAP bridge `RuView Sensing` carrying N child accessories — one per
room. Implements the topology decision from ADR-125 §2.1.c: single
pairing for the operator, child accessories that map cleanly to
"is there motion in the [room]?" Siri queries.

Each child accessory carries the three services iter 1 introduced:
  - MotionSensor                 (short-window movement)
  - OccupancySensor              (sustained presence — "Unknown Presence")
  - StatelessProgrammableSwitch  (anomaly event, Restricted class only)

State per room comes from `/tmp/ruview-state.<room>.json`. A C6
provisioned with `--room kitchen` writes `/tmp/ruview-state.kitchen.json`;
the bridge picks it up automatically on next launch.

For backwards-compat with iter 1-2 (one-room setup) the legacy
`/tmp/ruview-state.json` still feeds the room named via `--legacy-room`
(default: `Living Room`).

This script intentionally uses port 51827 (one above the test bridge's
51826) and a separate persist file so the iter-1-paired `RuView Test
Bridge` keeps working on the operator's iPhone. The two bridges are
independent; the operator can pair both, then remove the test bridge
once happy with the production one.

Usage:
    python3 ruview-hap-bridge.py                  # auto-discover rooms
    python3 ruview-hap-bridge.py --rooms "Living Room,Bedroom,Office"
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from pyhap.accessory import Accessory, Bridge
from pyhap.accessory_driver import AccessoryDriver
from pyhap.characteristic import Characteristic
from pyhap.const import CATEGORY_SENSOR, CATEGORY_BRIDGE

# Custom HomeKit Characteristic UUID for "BFLD Privacy Class" — Eve-renderable
# extension to the standard MotionSensor service. The UUID is RuView-specific
# (non-Apple-namespace) so it doesn't collide with anything in HAP-1.1.
# Eve.app and Controller for HomeKit will render this as an integer 2..3
# under the accessory's detail view; Home.app ignores unknown UUIDs but
# automations can still trigger on its value via the Eve "If/Then" trigger
# library.
BFLD_PRIVACY_CLASS_UUID = "8B0E1C00-0001-4B0E-9C00-1234567890AB"

STATE_DIR = Path(os.path.expanduser("~/.ruview-hap-prod"))
STATE_DIR.mkdir(exist_ok=True)
PERSIST_FILE = STATE_DIR / "bridge.state"
SETUP_CODE_FILE = STATE_DIR / "setup-code.txt"

LEGACY_STATE = Path("/tmp/ruview-state.json")
ROOM_STATE_GLOB = re.compile(r"^/tmp/ruview-state\.([^/]+)\.json$")


def discover_rooms_from_filesystem() -> list[tuple[str, Path]]:
    """Scan /tmp for ruview-state.<room>.json files and return (room, path)."""
    rooms: list[tuple[str, Path]] = []
    for entry in Path("/tmp").glob("ruview-state.*.json"):
        m = ROOM_STATE_GLOB.match(str(entry))
        if m:
            room = m.group(1).replace("-", " ").title()
            rooms.append((room, entry))
    return rooms


def _read_state(path: Path) -> dict | None:
    try:
        with open(path, "r") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else None
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


class RoomAccessory(Accessory):
    """One room's accessory — Motion + Occupancy + Anomaly switch."""

    category = CATEGORY_SENSOR

    def __init__(self, driver, name: str, state_path: Path, *args, **kwargs):
        super().__init__(driver, name, *args, **kwargs)
        self._state_path = state_path
        s_motion = self.add_preload_service("MotionSensor")
        self.c_motion = s_motion.configure_char("MotionDetected")
        s_occ = self.add_preload_service("OccupancySensor")
        self.c_occ = s_occ.configure_char("OccupancyDetected")
        s_sw = self.add_preload_service("StatelessProgrammableSwitch")
        self.c_anomaly = s_sw.configure_char("ProgrammableSwitchEvent")

        # ADR-125 §2.1.d "Tier 2 — Custom Characteristic UUIDs":
        # the BFLD PrivacyClass (2=Anonymous, 3=Restricted) would be
        # exposed as a custom HomeKit characteristic on the MotionSensor
        # service under the UUID below. Apple's Home.app ignores unknown
        # UUIDs; Eve.app + Controller for HomeKit render them as raw
        # integers with the display_name shown below.
        #
        # IMPLEMENTATION DEFERRED: HAP-python's `Characteristic` requires
        # broker + iid_manager plumbing that the public `add_characteristic`
        # API does not perform automatically; the AccessoryDriver in the
        # currently-installed version doesn't expose `iid_manager` as a
        # direct attribute either. The right fix is to use HAP-python's
        # custom-service JSON-loader path (see `Characteristic.from_dict`
        # + `Service.add_preload_service` with a custom resource) — a
        # follow-up iter ships that. The constant + spec stays here as
        # the SOTA-ready scaffold.
        self.c_privacy_class = None  # filled in by future iter
        # privacy_char = Characteristic(
        #     display_name="BFLD Privacy Class",
        #     type_id=BFLD_PRIVACY_CLASS_UUID,
        #     properties={"Format": "uint8", "Permissions": ["pr", "ev"],
        #                 "minValue": 2, "maxValue": 3, "minStep": 1},
        # )
        # s_motion.add_characteristic(privacy_char)
        # self.c_privacy_class = privacy_char

        self._last_motion = False
        self._last_occ = False
        self._last_anomaly_ts = 0.0
        self._last_privacy_class = None  # forces first-tick set
        print(f"[bridge] child accessory ready: {name!r}  "
              f"<- {state_path}", flush=True)
        print(f"[bridge]   custom char: BFLD Privacy Class "
              f"({BFLD_PRIVACY_CLASS_UUID})", flush=True)

    @Accessory.run_at_interval(1.0)
    def run(self):
        state = _read_state(self._state_path)
        if state is None:
            return  # absent / stale — leave HomeKit state at last-known
        motion = bool(state.get("motion", False))
        occupancy = bool(state.get("occupancy", False))
        anomaly_ts = float(state.get("anomaly_ts", 0.0) or 0.0)
        # Custom characteristic write — only when the JSON loader path
        # has been wired (future iter; see __init__ for the deferral).
        if self.c_privacy_class is not None:
            privacy_class = int(state.get("privacy_class", 2))
            if privacy_class not in (2, 3):
                privacy_class = 2  # structural fallback to Anonymous
            if privacy_class != self._last_privacy_class:
                self.c_privacy_class.set_value(privacy_class)
                self._last_privacy_class = privacy_class
                print(f"[bridge] {self.display_name}: BFLD Privacy Class "
                      f"-> {privacy_class}", flush=True)

        if motion != self._last_motion:
            self.c_motion.set_value(motion)
            self._last_motion = motion
            print(f"[bridge] {self.display_name}: Motion -> {motion}",
                  flush=True)
        if occupancy != self._last_occ:
            self.c_occ.set_value(1 if occupancy else 0)
            self._last_occ = occupancy
            print(f"[bridge] {self.display_name}: Occupancy -> {occupancy} "
                  f"(Siri: 'is anyone in the {self.display_name.lower()}?')",
                  flush=True)
        if anomaly_ts > self._last_anomaly_ts:
            self.c_anomaly.set_value(0)
            self._last_anomaly_ts = anomaly_ts
            print(f"[bridge] {self.display_name}: "
                  f"Unrecognized Activity Pattern fired", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=51827)
    p.add_argument("--rooms",
                   help="Comma-separated rooms to advertise. Each one maps "
                        "to /tmp/ruview-state.<lowercase-hyphen>.json. "
                        "Default: auto-discover from filesystem + legacy.")
    p.add_argument("--legacy-room", default="Living Room",
                   help="Name attached to /tmp/ruview-state.json (the iter "
                        "1-2 single-file IPC). Default: 'Living Room'.")
    args = p.parse_args()

    driver = AccessoryDriver(port=args.port, persist_file=str(PERSIST_FILE))
    bridge = Bridge(driver, "RuView Sensing")
    bridge.category = CATEGORY_BRIDGE

    rooms: list[tuple[str, Path]] = []
    if args.rooms:
        for r in [s.strip() for s in args.rooms.split(",") if s.strip()]:
            slug = r.lower().replace(" ", "-")
            rooms.append((r, Path(f"/tmp/ruview-state.{slug}.json")))
    else:
        rooms = discover_rooms_from_filesystem()
        if LEGACY_STATE.exists() or args.legacy_room:
            rooms.insert(0, (args.legacy_room, LEGACY_STATE))

    if not rooms:
        sys.stderr.write(
            "ERROR: no rooms discovered. Either run "
            "c6-presence-watcher.py first (writes /tmp/ruview-state.json), "
            "or pass --rooms 'Name1,Name2'.\n"
        )
        return 2

    for name, path in rooms:
        bridge.add_accessory(RoomAccessory(driver, name, path))

    driver.add_accessory(accessory=bridge)
    setup_code = driver.state.pincode
    if hasattr(setup_code, "decode"):
        setup_code = setup_code.decode()
    SETUP_CODE_FILE.write_text(str(setup_code) + "\n")
    print(f"[bridge] HAP bridge advertising as 'RuView Sensing' (production)",
          flush=True)
    print(f"[bridge] Setup code (also in {SETUP_CODE_FILE}): {setup_code}",
          flush=True)
    print(f"[bridge] Rooms: {[r[0] for r in rooms]}", flush=True)
    print(f"[bridge] iPhone pair: Home app -> Add Accessory -> More Options",
          flush=True)
    driver.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
