#!/usr/bin/env python3
"""Provision a node without putting the WiFi password on a command line.

`provision.py` takes --password as an argument, so every invocation leaves the
credential in shell history, process listings, terminal scrollback and any
transcript of the session. That is a poor place for it and an easy thing to
forget you did.

This reads the password from a file, passes it to provision.py as a subprocess
argument (unavoidable -- that is provision.py's only interface), and never
prints or echoes it. The password still appears briefly in the child process's
argv, so this is not protection against a local attacker; it is protection
against the credential being permanently recorded somewhere it does not belong.

It also fills in the two things that are easy to get wrong by hand:

  * node_id comes from board_index.py's MAC index, so a board keeps its
    identity regardless of which USB socket it lands in.
  * --tdm-total is required, not optional. The firmware derives its ESP-NOW
    beacon period from it (c6_sync_espnow_period_ms), and a node provisioned
    without it reports `fleet=1` and falls back to the 80 ms default -- which
    is correct for three nodes and silently overruns the 50 Hz receive gate at
    nine. Measured on 2026-08-28: an over-fast beacon discarded 60-90% of peer
    frames at random and wedged a transmit queue.

    python provision_node.py --port COM3 --tdm-total 9
    python provision_node.py --all --tdm-total 9      # every mounted board
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONF = os.path.join(HERE, "provision_conf.json")


def load_conf():
    if not os.path.exists(CONF):
        sys.exit(
            "no %s.\n\nCreate it with:\n"
            '  {\n'
            '    "ssid": "SilverESP",\n'
            '    "password_file": "D:\\\\path\\\\to\\\\SilverESP.txt",\n'
            '    "target_ip": "192.168.1.66",\n'
            '    "chip": "esp32c6",\n'
            '    "edge_tier": 2\n'
            '  }\n\n'
            "password_file holds the passphrase and nothing else. Keep it "
            "outside this repo." % CONF)
    with open(CONF, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_password(path):
    try:
        with open(path, encoding="utf-8-sig") as fh:
            pw = fh.read().strip()
    except Exception as e:
        sys.exit("cannot read password file: %s" % e)
    if not pw:
        sys.exit("password file is empty")
    if len(pw) < 8:
        sys.exit("password is %d chars; WPA needs at least 8 -- is the file "
                 "holding something else?" % len(pw))
    return pw


def node_id_for(port):
    """Ask board_index.py who this board is, so identity follows the MAC."""
    idx_path = os.path.join(HERE, "board_index.json")
    try:
        with open(idx_path, encoding="utf-8") as fh:
            idx = json.load(fh)
    except Exception:
        return None
    for mac, b in idx.get("boards", {}).items():
        if b.get("last_port") == port:
            return b["node_id"], mac
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port; omit with --all")
    ap.add_argument("--all", action="store_true",
                    help="provision every board currently in the index")
    ap.add_argument("--tdm-total", type=int, required=True,
                    help="TOTAL nodes in the fleet. Required: the firmware "
                         "derives its beacon period from this and gets it "
                         "wrong, silently, if left unset.")
    ap.add_argument("--node-id", type=int,
                    help="override the id from board_index.json")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    conf = load_conf()
    pw = read_password(conf["password_file"])

    targets = []
    if a.all:
        idx_path = os.path.join(HERE, "board_index.json")
        with open(idx_path, encoding="utf-8") as fh:
            idx = json.load(fh)
        for mac, b in sorted(idx["boards"].items(), key=lambda kv: kv[1]["node_id"]):
            if b.get("last_port"):
                targets.append((b["last_port"], b["node_id"], mac))
    else:
        if not a.port:
            sys.exit("--port or --all required")
        nid = a.node_id
        mac = "?"
        found = node_id_for(a.port)
        if found and nid is None:
            nid, mac = found
        if nid is None:
            sys.exit("no node_id for %s -- run board_index.py --assign first, "
                     "or pass --node-id" % a.port)
        targets.append((a.port, nid, mac))

    if a.tdm_total < len(targets):
        sys.exit("--tdm-total %d is smaller than the %d boards being "
                 "provisioned" % (a.tdm_total, len(targets)))

    for port, nid, mac in targets:
        cmd = [sys.executable, os.path.join(HERE, "provision.py"),
               "--port", port,
               "--chip", conf.get("chip", "esp32c6"),
               "--ssid", conf["ssid"],
               "--password", pw,
               "--target-ip", conf["target_ip"],
               "--node-id", str(nid),
               "--tdm-slot", str(nid),
               "--tdm-total", str(a.tdm_total),
               "--edge-tier", str(conf.get("edge_tier", 2))]
        shown = [c if c != pw else "<password>" for c in cmd]
        print("\n%s  node %d  (%s)" % (port, nid, mac))
        print("  " + " ".join(shown[1:]))
        if a.dry_run:
            continue
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print("  FAILED rc=%d" % r.returncode)
        else:
            print("  ok -- confirm the boot log says fleet=%d" % a.tdm_total)


if __name__ == "__main__":
    main()
