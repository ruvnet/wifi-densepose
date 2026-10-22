#!/usr/bin/env python3
"""Build an NVS image that fills only the cfg_trial namespace.

The recovery bank is written to the cfg_trial namespace before any live
configuration is written. Filling that namespace alone makes the bank write
fail with ESP_ERR_NVS_NOT_ENOUGH_SPACE while the WiFi namespaces keep the
room they need to boot the node.

The WiFi password and the OTA PSK come from local files and are never
printed. The generated CSV holds both in clear text and must be treated as
a secret until it is deleted.

    ./make_trial_full_nvs.py 120
    ./make_trial_full_nvs.py 120 --out-bin /tmp/nvs-trial-full.bin

The NVS image itself is produced by the `esp_idf_nvs_partition_gen` module
(Espressif, Apache-2.0, `pip install esp-idf-nvs-partition-gen`). Point
--python at an interpreter that has it, or point --nvs-gen at an
`nvs_partition_gen.py` wrapper.
"""

import argparse
import csv
import io
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SIZE = 24576


def default_out(name):
    return os.path.join(HERE, name)


def read_secret(path, what):
    if not os.path.isfile(path):
        raise SystemExit("cannot read %s file: %s" % (what, path))
    with open(path, encoding="utf-8") as fh:
        return fh.read().strip()


def build_csv(password, psk, junk_count, junk_len=64):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["key", "type", "encoding", "value"])
    # live configuration first, so it always gets its own page
    w.writerow(["csi_cfg", "namespace", "", ""])
    w.writerow(["ssid", "data", "string", "wagaya"])
    w.writerow(["password", "data", "string", password])
    w.writerow(["target_ip", "data", "string", "192.168.0.57"])
    w.writerow(["node_id", "data", "u8", "1"])
    w.writerow(["vital_win", "data", "u16", "256"])
    w.writerow(["subk_count", "data", "u8", "32"])
    w.writerow(["csi_channel", "data", "u8", "6"])
    w.writerow(["security", "namespace", "", ""])
    w.writerow(["ota_psk", "data", "string", psk])
    # then the trial namespace, packed with junk
    w.writerow(["cfg_trial", "namespace", "", ""])
    filler = "t" * junk_len
    for i in range(junk_count):
        w.writerow(["fill_%04d" % i, "data", "string", filler])
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("junk_count", nargs="?", type=int, default=60,
                    help="number of filler entries in the cfg_trial namespace")
    ap.add_argument("--wifi-pass-file",
                    default=os.environ.get("RUVIEW_WIFI_PASS",
                                           os.path.expanduser("~/.wifi-pass")))
    ap.add_argument("--ota-psk-file",
                    default=os.environ.get("RUVIEW_OTA_PSK",
                                           os.path.expanduser("~/.ruview-ota-psk")))
    ap.add_argument("--out-bin", default=default_out("nvs-trial-full.bin"))
    ap.add_argument("--out-csv", default=default_out("nvs-trial-full.csv"))
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE)
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter that can import esp_idf_nvs_partition_gen")
    ap.add_argument("--nvs-gen", default=os.environ.get("NVS_GEN_SCRIPT"),
                    help="path to an nvs_partition_gen.py wrapper "
                         "(default: python -m esp_idf_nvs_partition_gen)")
    args = ap.parse_args()

    password = read_secret(args.wifi_pass_file, "wifi password")
    psk = read_secret(args.ota_psk_file, "ota psk")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
        fh.write(build_csv(password, psk, args.junk_count))

    if args.nvs_gen:
        cmd = [args.python, args.nvs_gen, "generate", args.out_csv,
               args.out_bin, str(args.size)]
    else:
        cmd = [args.python, "-m", "esp_idf_nvs_partition_gen", "generate",
               args.out_csv, args.out_bin, str(args.size)]

    probe = subprocess.run([args.python, "-c",
                            "import esp_idf_nvs_partition_gen"],
                           capture_output=True, text=True)
    if probe.returncode != 0 and not args.nvs_gen:
        print("junk=%d exit=%d ok=False" % (args.junk_count, probe.returncode))
        print("the interpreter %s cannot import esp_idf_nvs_partition_gen"
              % args.python)
        print("install it (pip install esp-idf-nvs-partition-gen) or pass "
              "--nvs-gen /path/to/nvs_partition_gen.py")
        return 1

    result = subprocess.run(cmd, capture_output=True, text=True)
    ok = result.returncode == 0 and os.path.exists(args.out_bin)
    print("junk=%d exit=%d ok=%s" % (args.junk_count, result.returncode, ok))
    if not ok:
        print((result.stdout or "")[-400:])
        print((result.stderr or "")[-400:])
        return 1
    print("image %s (%d bytes)" % (args.out_bin, os.path.getsize(args.out_bin)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
