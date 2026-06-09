"""Command-line interface for eegspectra.

Usage
-----
Single file (human-readable report):
    python3 -m eegspectra analyze 0000002.EDF

Single file, write channel band-power CSV + per-band PSD:
    python3 -m eegspectra analyze 0000002.EDF --csv out.csv

Batch a folder / glob (one summary row per recording):
    python3 -m eegspectra batch "*.EDF" --csv cohort.csv
    python3 -m eegspectra batch . --csv cohort.csv

Just inspect headers (fast, no spectral computation):
    python3 -m eegspectra info "*.EDF" "*.BDF"
"""
import sys
import os
import glob
import csv as _csv
import argparse

from .io import read_recording, detect_format
from .montage import classify_channels, montage_report
from .bands import BANDS, channel_band_table, average_band_powers
from .indices import sport_indices, interpret


def _expand(patterns):
    files = []
    for p in patterns:
        if os.path.isdir(p):
            for ext in ("*.edf", "*.EDF", "*.bdf", "*.BDF", "*.csv", "*.CSV"):
                files.extend(glob.glob(os.path.join(p, ext)))
        elif any(ch in p for ch in "*?[]"):
            files.extend(glob.glob(p))
        elif os.path.isfile(p):
            files.append(p)
    # de-dup, keep order
    seen, out = set(), []
    for f in sorted(files):
        if f not in seen:
            seen.add(f); out.append(f)
    return out


# --------------------------------------------------------------------------- #
def cmd_info(args):
    files = _expand(args.paths)
    if not files:
        print("No files matched.", file=sys.stderr); return 1
    print("%-22s %-6s %-32s %5s %8s %9s" %
          ("file", "fmt", "device", "ch", "fs(Hz)", "dur(s)"))
    print("-" * 90)
    for f in files:
        try:
            rec = classify_channels(read_recording(f))
            print("%-22s %-6s %-32s %5d %8g %9g" % (
                os.path.basename(f), rec.source_format, rec.device,
                len(rec.eeg_channels()), rec.fs, rec.duration))
        except Exception as e:
            print("%-22s  ERROR: %s" % (os.path.basename(f), e))
    return 0


def cmd_analyze(args):
    rec = classify_channels(read_recording(args.path))
    print("=" * 72)
    print("FILE:", args.path)
    print(rec.summary())
    print(montage_report(rec))
    print("-" * 72)

    rows = channel_band_table(rec, nperseg=args.nperseg)
    avg = average_band_powers(rec, nperseg=args.nperseg)
    bands = list(BANDS.keys())

    print("Per-channel RELATIVE band power:")
    print("  %-6s " % "ch" + " ".join("%7s" % b for b in bands))
    for r in rows:
        print("  %-6s " % r["channel"] +
              " ".join("%7.3f" % r["rel"][b] for b in bands))
    print("  %-6s " % "MEAN" + " ".join("%7.3f" % avg["rel"][b] for b in bands))

    print("-" * 72)
    print("Sport spectral indices (whole-head):")
    idx = sport_indices(rec, nperseg=args.nperseg)
    for k, v in idx.items():
        print("  %-28s %8.4f" % (k, v))
    notes = interpret(idx)
    if notes:
        print("Interpretation (heuristic, non-diagnostic):")
        for n in notes:
            print("  - " + n)

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["channel", "label"] +
                       ["abs_" + b for b in bands] +
                       ["rel_" + b for b in bands])
            for r in rows:
                w.writerow([r["channel"], r["label"]] +
                           ["%.6g" % r["abs"][b] for b in bands] +
                           ["%.6g" % r["rel"][b] for b in bands])
        print("\nWrote per-channel band powers -> %s" % args.csv)
    return 0


def cmd_batch(args):
    files = _expand(args.paths)
    if not files:
        print("No files matched.", file=sys.stderr); return 1
    bands = list(BANDS.keys())
    header = (["file", "format", "device", "patient", "n_eeg", "fs_Hz", "dur_s"] +
              ["rel_" + b for b in bands] +
              ["neural_efficiency_index", "mental_fatigue_index",
               "attention_engagement_index", "theta_beta_ratio", "alpha_beta_ratio"])
    out_rows = []
    print("Processing %d file(s)..." % len(files))
    for i, f in enumerate(files, 1):
        try:
            rec = classify_channels(read_recording(f))
            avg = average_band_powers(rec, nperseg=args.nperseg)
            idx = sport_indices(rec, nperseg=args.nperseg)
            row = [os.path.basename(f), rec.source_format, rec.device,
                   rec.patient, len(rec.eeg_channels()),
                   "%g" % rec.fs, "%g" % rec.duration]
            row += ["%.5f" % avg["rel"][b] for b in bands]
            row += ["%.5f" % idx.get(k, 0.0) for k in
                    ("neural_efficiency_index", "mental_fatigue_index",
                     "attention_engagement_index", "theta_beta_ratio",
                     "alpha_beta_ratio")]
            out_rows.append(row)
            print("  [%d/%d] %-20s %-30s OK" %
                  (i, len(files), os.path.basename(f), rec.device))
        except Exception as e:
            print("  [%d/%d] %-20s ERROR: %s" % (i, len(files), os.path.basename(f), e))

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(header)
            w.writerows(out_rows)
        print("\nWrote cohort table (%d recordings) -> %s" % (len(out_rows), args.csv))
    else:
        print("\nTip: pass --csv cohort.csv to save the cohort table.")
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="eegspectra",
        description="Spectral analysis of athlete EEG (EDF/BDF/CSV, any montage).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("info", help="quick header inspection")
    pi.add_argument("paths", nargs="+", help="files, globs or folders")
    pi.set_defaults(func=cmd_info)

    pa = sub.add_parser("analyze", help="full spectral report for one file")
    pa.add_argument("path")
    pa.add_argument("--csv", help="write per-channel band powers to this CSV")
    pa.add_argument("--nperseg", type=int, default=None,
                    help="Welch segment length in samples (default ~2 s)")
    pa.set_defaults(func=cmd_analyze)

    pb = sub.add_parser("batch", help="analyze many files -> cohort CSV")
    pb.add_argument("paths", nargs="+", help="files, globs or folders")
    pb.add_argument("--csv", help="write cohort summary table to this CSV")
    pb.add_argument("--nperseg", type=int, default=None)
    pb.set_defaults(func=cmd_batch)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
