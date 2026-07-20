"""Regenerate results/nan_windows_mask.npy + results/big_windows_mask.npy by
scanning a PRISTINE kagglehub download of the WiFlow-STD dataset
(kaka2434/wiflow-dataset v1, csi_windows.npy, 360,000 windows of 540x20).

============================ READ THIS FIRST ===============================
This script MUST be run against an UNCLEANED copy of the dataset.

remote/clean_v2.py (and its predecessor clean_nan.py) repair the dataset by
zeroing the corrupted windows IN PLACE, with no backup. A cleaned copy
contains no non-finite values and no out-of-range amplitudes, so on a cleaned
copy this scan produces ALL-FALSE masks -- silently wrong ground truth. The
script errors out loudly in that case (see the sanity check in main()).

That irreversibility is exactly why the two committed mask files under
results/ (gitignore-negated) are the canonical ground truth: once a download
has been cleaned, the masks can NEVER be regenerated from it. Only run this
on a fresh `kagglehub.dataset_download("kaka2434/wiflow-dataset")`.
============================================================================

Criteria (per window; mirrors the original 2026-06-10 scan and the
remote/clean_v2.py repair criteria):

  nan mask: any non-finite value (NaN/Inf) anywhere in the 540x20 window
  big mask: max |finite value| > 1.5 (the data is otherwise [0,1]-normalized;
            the corrupted files contain garbage up to 3.4e38, float32 max)

Expected result on the pristine Kaggle download (RESULTS.md defect 5):
  nan: 9,070 True | big: 9,072 True | union: 9,072 -- all windows in dataset
  files 487-499 (the final 13 files), window indices 350,922-359,999.

Usage:
  PYTHONUTF8=1 .venv/Scripts/python.exe generate_corruption_masks.py \
      [--data-dir <dir containing csi_windows.npy>] [--out-dir results]
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

EXPECTED = {"nan": 9070, "big": 9072, "union": 9072,
            "files": (487, 499), "windows": (350922, 359999)}


def scan(csi_path, chunk=4000):
    """Chunked scan of the (mmap'd) windows array; returns (nan_mask, big_mask)."""
    csi = np.load(csi_path, mmap_mode="r")
    n = len(csi)
    nan_mask = np.zeros(n, dtype=bool)
    big_mask = np.zeros(n, dtype=bool)
    for i in range(0, n, chunk):
        block = np.asarray(csi[i:i + chunk])
        finite = np.isfinite(block)
        nan_mask[i:i + chunk] = (~finite).any(axis=(1, 2))
        big_mask[i:i + chunk] = (
            np.abs(np.where(finite, block, 0)).max(axis=(1, 2)) > 1.5)
        if (i // chunk) % 10 == 0:
            print(f"  scanned {min(i + chunk, n):,}/{n:,} windows "
                  f"(nan={int(nan_mask.sum()):,} big={int(big_mask.sum()):,})",
                  flush=True)
    return nan_mask, big_mask


def describe_files(data_dir, mask):
    """Map marked windows to dataset file indices via window_info.npz."""
    info = os.path.join(data_dir, "window_info.npz")
    if not os.path.exists(info):
        return None
    w2f = np.load(info)["window_to_file"]
    return np.unique(w2f[mask])


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate the corruption masks from a PRISTINE "
                    "(uncleaned) kagglehub download. See module docstring.")
    parser.add_argument("--data-dir", default=os.path.join(
        os.path.expanduser("~"), ".cache", "kagglehub", "datasets", "kaka2434",
        "wiflow-dataset", "versions", "1", "preprocessed_csi_data"),
        help="Directory containing csi_windows.npy (PRISTINE copy)")
    parser.add_argument("--out-dir", default=RESULTS,
                        help="Where to write the two .npy masks")
    parser.add_argument("--chunk", type=int, default=4000,
                        help="Windows per scan chunk (memory/speed tradeoff)")
    args = parser.parse_args()

    csi_path = os.path.join(args.data_dir, "csi_windows.npy")
    if not os.path.exists(csi_path):
        sys.exit(f"csi_windows.npy not found in {args.data_dir}")

    print(f"scanning {csi_path} (chunk={args.chunk}) ...")
    nan_mask, big_mask = scan(csi_path, args.chunk)
    union = nan_mask | big_mask
    print(f"nan: {int(nan_mask.sum()):,} | big: {int(big_mask.sum()):,} | "
          f"union: {int(union.sum()):,} of {len(union):,} windows")

    # ---- sanity check: an all-False result means a CLEANED copy ------------
    if not union.any():
        sys.exit(
            "ERROR: scan found ZERO corrupted windows.\n"
            "\n"
            "The pristine Kaggle download (kaka2434/wiflow-dataset v1) is "
            "known to contain\n"
            "9,072 corrupted windows (NaN/Inf + amplitudes up to 3.4e38) in "
            "dataset files\n"
            "487-499 (RESULTS.md, reproducibility defect 5). Finding none "
            "means this copy\n"
            "has almost certainly already been repaired by remote/clean_v2.py "
            "(or clean_nan.py),\n"
            "which zeroes the corrupted windows IN PLACE -- after that the "
            "corruption evidence\n"
            "is gone and the masks CANNOT be regenerated from this copy.\n"
            "\n"
            "Refusing to overwrite the committed ground-truth masks with "
            "all-False ones.\n"
            "Re-download the dataset (kagglehub.dataset_download("
            "'kaka2434/wiflow-dataset'))\n"
            "and point --data-dir at the fresh, uncleaned copy.")

    files = describe_files(args.data_dir, union)
    if files is not None:
        print(f"marked windows span dataset files {files.min()}-{files.max()}: "
              f"{files.tolist()}")
        lo, hi = EXPECTED["files"]
        if files.min() != lo or files.max() != hi:
            print(f"WARNING: expected marked files exactly {lo}-{hi} "
                  f"(the pristine v1 download); got {files.min()}-{files.max()}. "
                  f"Different dataset version, or a partially cleaned copy?")
    for name, mask, exp in (("nan", nan_mask, EXPECTED["nan"]),
                            ("big", big_mask, EXPECTED["big"])):
        if int(mask.sum()) != exp:
            print(f"WARNING: {name} mask has {int(mask.sum()):,} True windows; "
                  f"the pristine v1 download yields {exp:,}.")

    os.makedirs(args.out_dir, exist_ok=True)
    for name, mask in (("nan_windows_mask.npy", nan_mask),
                       ("big_windows_mask.npy", big_mask)):
        out = os.path.join(args.out_dir, name)
        np.save(out, mask)
        print(f"wrote {out} ({int(mask.sum()):,} True)")


if __name__ == "__main__":
    main()
