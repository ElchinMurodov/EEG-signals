"""Device detection and 10-20 channel-name normalization.

Works for arbitrary channel counts. Known device fingerprints are recognized
explicitly; everything else falls back to a generic 10-20 normalizer so the
tool still works with unknown montages.
"""
import re
from typing import List
from .core import Recording, Channel

# Canonical international 10-20 / 10-10 electrode names we recognize.
STANDARD_1020 = {
    "FP1", "FP2", "FPZ", "AF3", "AF4", "AF7", "AF8", "AFZ",
    "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "FZ",
    "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FCZ", "FT7", "FT8",
    "C1", "C2", "C3", "C4", "C5", "C6", "CZ",
    "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPZ", "TP7", "TP8",
    "T3", "T4", "T5", "T6", "T7", "T8",
    "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "PZ",
    "PO3", "PO4", "PO7", "PO8", "POZ",
    "O1", "O2", "OZ",
}

# Old (T3/T4/T5/T6) <-> new (T7/T8/P7/P8) naming bridge.
OLD_TO_NEW = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def normalize_name(label: str) -> str:
    """Extract a canonical 10-20 electrode name from a raw channel label.

    Examples:
        "EEG Fp1"   -> "Fp1"
        "Fp1-A1"    -> "Fp1"
        "EEG T3-Ref"-> "T3"
        "C3-A1"     -> "C3"
    """
    s = label.strip()
    # drop a leading modality token
    s = re.sub(r"^(EEG|EKG|ECG|EMG|EOG)\s+", "", s, flags=re.IGNORECASE)
    # take the part before a reference ("-A1", "-Ref", "-M1", "-A2")
    s = re.split(r"[-/]", s)[0].strip()
    up = s.upper()
    if up in STANDARD_1020:
        return _pretty(up)
    # try stripping trailing non-alphanumerics
    up2 = re.sub(r"[^A-Z0-9]", "", up)
    if up2 in STANDARD_1020:
        return _pretty(up2)
    return s  # unknown -> return cleaned label


def _pretty(up: str) -> str:
    """Render a canonical upper-case name in conventional casing (Fp1, Cz...)."""
    if up.startswith("FP"):
        return "Fp" + up[2:]
    if len(up) >= 1:
        return up[0] + up[1:].lower()
    return up


# Device fingerprints: (predicate, device-name).
def detect_device(rec: Recording) -> str:
    labels = [c.label for c in rec.channels]
    up = " ".join(labels).upper()
    n_eeg = len(rec.eeg_channels())
    fs = rec.fs

    # CONTEC KT-88: ear-referenced labels like "Fp1-A1", 16 EEG ch, ~100/200 Hz
    if any(re.search(r"-A[12]\b", l, re.IGNORECASE) for l in labels) and 14 <= n_eeg <= 16:
        return "CONTEC KT-88 (16-channel)"

    # Neurotech Compact-neuro: "EEG Fpz/Oz/Fz" present, 21 EEG ch, 500 Hz, has ECG/EMG/EOG
    has_midline = all(any(normalize_name(c.label).lower() == m for c in rec.eeg_channels())
                      for m in ("fpz", "oz"))
    extra_mods = {c.kind for c in rec.channels} & {"ECG", "EMG", "EOG"}
    if 19 <= n_eeg <= 21 and has_midline:
        return "Neurotech Compact-neuro (21-channel)"
    if 19 <= n_eeg <= 21 and extra_mods and fs >= 250:
        return "Neurotech Compact-neuro (21-channel)"

    # generic fallbacks
    if n_eeg:
        return "generic (%d-channel)" % n_eeg
    return "unknown"


def classify_channels(rec: Recording) -> Recording:
    """Fill in normalized 10-20 names and detect the device. Mutates & returns rec."""
    for c in rec.channels:
        if c.kind == "EEG":
            c.name = normalize_name(c.label)
        else:
            c.name = c.label
    rec.device = detect_device(rec)
    return rec


def montage_report(rec: Recording) -> str:
    eeg = rec.eeg_channels()
    names = [c.name or normalize_name(c.label) for c in eeg]
    recognized = [n for n in names if n.upper() in STANDARD_1020
                  or n.upper() in OLD_TO_NEW]
    lines = [
        "Device      : %s" % rec.device,
        "EEG channels: %d  (%s)" % (len(eeg), ", ".join(names)),
        "Recognized  : %d/%d as standard 10-20" % (len(recognized), len(eeg)),
        "Sampling    : %g Hz" % rec.fs,
    ]
    others = [c for c in rec.channels if c.kind != "EEG"]
    if others:
        lines.append("Other leads : %s" % ", ".join(
            "%s(%s)" % (c.label, c.kind) for c in others))
    return "\n".join(lines)
