#!/usr/bin/env python3
"""
hap-test-sensor.py — ADR-125 §2.1.a smoke test.

Stands up a single HomeKit Accessory Protocol (HAP-1.1) bridge with one
child MotionSensor named "RuView Test Motion". Once paired in the Apple
Home app, the HomePod (acting as Home Hub) sees state changes when
TOGGLE_FILE (default /tmp/ruview-motion) is touched / removed.

Usage:
    python3 hap-test-sensor.py

Pair from iPhone: Home app -> Add Accessory -> More Options -> "RuView Test Bridge".
The setup code is printed on stdout AND written to ~/.ruview-hap/setup-code.txt.

Trigger motion:  touch /tmp/ruview-motion
Clear motion:    rm   /tmp/ruview-motion

State persists across restarts in ~/.ruview-hap/accessory.state.
"""

from pathlib import Path
import json
import os
import sys
import time
import signal

from pyhap.accessory import Accessory, Bridge
from pyhap.accessory_driver import AccessoryDriver
from pyhap.const import CATEGORY_SENSOR, CATEGORY_BRIDGE

STATE_DIR = Path(os.path.expanduser("~/.ruview-hap"))
STATE_DIR.mkdir(exist_ok=True)
STATE_FILE = STATE_DIR / "accessory.state"
SETUP_CODE_FILE = STATE_DIR / "setup-code.txt"

# Legacy single-bool toggle (iter 1-3 contract). Still honored for
# backwards-compat with the original c6-presence-watcher.py path.
TOGGLE_FILE = Path(os.environ.get("RUVIEW_MOTION_TOGGLE", "/tmp/ruview-motion"))

# New JSON-state IPC contract (iter 4+). When present, takes precedence
# over the legacy toggle file. Schema:
#   {
#     "motion": bool,        # short-window movement (100 ms feature_state)
#     "occupancy": bool,     # rolling-window sustained presence (1 s+)
#     "anomaly": bool,       # BFLD anomaly drift gate fired (class-3 only)
#     "ts": float,           # unix epoch when the watcher last wrote
#   }
STATE_JSON = Path(os.environ.get("RUVIEW_STATE_JSON", "/tmp/ruview-state.json"))


def _read_state_json():
    """Best-effort read of the JSON IPC file. Returns None on any error."""
    try:
        with open(STATE_JSON, "r") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return None
        return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


class RuViewMotion(Accessory):
    """Three-service HomeKit accessory per ADR-125 §2.1.c.

    Same accessory carries:
      - MotionSensor — short-window movement (motion_score)
      - OccupancySensor — sustained occupancy (presence_score rolling avg)
      - StatelessProgrammableSwitch — "Unrecognized Activity Pattern"
        event (BFLD anomaly gate; Restricted-class only; momentary fire)

    The HomeKit pairing stays intact when adding services to an existing
    accessory — the iPhone re-reads `/accessories` after the bridge's
    config-number bumps and surfaces the new characteristics under the
    same paired entity.
    """
    category = CATEGORY_SENSOR

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        s_motion = self.add_preload_service("MotionSensor")
        self.char_motion = s_motion.configure_char("MotionDetected")
        s_occ = self.add_preload_service("OccupancySensor")
        self.char_occ = s_occ.configure_char("OccupancyDetected")
        s_sw = self.add_preload_service("StatelessProgrammableSwitch")
        self.char_anomaly = s_sw.configure_char("ProgrammableSwitchEvent")
        self._last_motion = False
        self._last_occ = False
        self._last_anomaly_ts = 0.0

    def _legacy_motion(self) -> bool:
        return TOGGLE_FILE.exists()

    @Accessory.run_at_interval(1.0)
    def run(self):
        state = _read_state_json()
        if state is None:
            motion = self._legacy_motion()
            occupancy = motion
            anomaly_fire = False
        else:
            motion = bool(state.get("motion", False))
            occupancy = bool(state.get("occupancy", False))
            anomaly_ts = float(state.get("anomaly_ts", 0.0) or 0.0)
            anomaly_fire = anomaly_ts > self._last_anomaly_ts
            if anomaly_fire:
                self._last_anomaly_ts = anomaly_ts

        if motion != self._last_motion:
            self.char_motion.set_value(motion)
            self._last_motion = motion
            print(f"[hap] MotionDetected -> {motion}", flush=True)
        if occupancy != self._last_occ:
            self.char_occ.set_value(1 if occupancy else 0)
            self._last_occ = occupancy
            print(f"[hap] OccupancyDetected -> {occupancy}", flush=True)
        if anomaly_fire:
            # 0 = single press; semantic-event = "Unrecognized Activity Pattern"
            self.char_anomaly.set_value(0)
            print(
                "[hap] Unrecognized Activity Pattern fired (ProgrammableSwitch=0)",
                flush=True,
            )


def main() -> int:
    driver = AccessoryDriver(port=51826, persist_file=str(STATE_FILE))

    bridge = Bridge(driver, "RuView Test Bridge")
    bridge.category = CATEGORY_BRIDGE
    bridge.add_accessory(RuViewMotion(driver, "RuView Test Motion"))
    driver.add_accessory(accessory=bridge)

    setup_code = driver.state.pincode.decode() if hasattr(driver.state.pincode, "decode") else driver.state.pincode
    SETUP_CODE_FILE.write_text(str(setup_code) + "\n")
    print(f"[hap-test] HAP bridge advertising as 'RuView Test Bridge'")
    print(f"[hap-test] iPhone pair flow: Home app -> Add Accessory -> More Options")
    print(f"[hap-test] Setup code (also in {SETUP_CODE_FILE}):  {setup_code}")
    print(f"[hap-test] State sources:")
    print(f"[hap-test]   primary:  {STATE_JSON}  (multi-characteristic JSON)")
    print(f"[hap-test]   fallback: {TOGGLE_FILE} (motion-only touch file)")
    print(f"[hap-test] Pair state persists in: {STATE_FILE}")

    signal.signal(signal.SIGTERM, lambda *_: driver.stop())
    driver.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
