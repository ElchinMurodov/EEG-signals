"""Multi-format readers producing a unified `Recording`.

Supported formats (auto-detected by extension + content sniffing):
  * EDF / EDF+   (European Data Format, 16-bit signed)
  * BDF / BDF+   (BioSemi Data Format, 24-bit signed)
  * CSV          (flexible: header row of channel names, optional time column)

EDF and BDF share an identical ASCII header; they differ only in sample width
and the version byte (BDF = 0xFF). A single parser handles both.
"""
import os
import struct
import csv as _csv
from typing import List, Optional

from .core import Recording, Channel

SUPPORTED_FORMATS = ("EDF", "BDF", "CSV")


# --------------------------------------------------------------------------- #
# Format detection
# --------------------------------------------------------------------------- #
def detect_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".edf",):
        return "EDF"
    if ext in (".bdf",):
        return "BDF"
    if ext in (".csv", ".txt", ".tsv"):
        return "CSV"
    # content sniff: BDF first byte is 0xFF, EDF first byte is '0'
    with open(path, "rb") as f:
        head = f.read(1)
    if head == b"\xff":
        return "BDF"
    if head == b"0":
        return "EDF"
    return "CSV"


# --------------------------------------------------------------------------- #
# EDF / BDF
# --------------------------------------------------------------------------- #
def _classify_label(label: str) -> str:
    up = label.upper()
    if "ANNOT" in up:
        return "ANNOT"
    if up.startswith("ECG") or up.startswith("EKG"):
        return "ECG"
    if up.startswith("EMG"):
        return "EMG"
    if up.startswith("EOG"):
        return "EOG"
    if up.startswith("EEG") or _looks_like_eeg(up):
        return "EEG"
    if "ADD_LEAD" in up or up.startswith("AUX") or up.startswith("BIP"):
        return "AUX"
    return "AUX"


_EEG_TOKENS = ("FP", "AF", "FZ", "FC", "CZ", "PZ", "OZ", "F", "C", "P", "O", "T")


def _looks_like_eeg(up: str) -> bool:
    base = up.split("-")[0].strip()
    if not base:
        return False
    # e.g. "FP1-A1", "C3-A1", "O2"
    return any(base.startswith(tok) for tok in ("FP", "AF", "FZ", "FC", "CZ",
                                                "PZ", "OZ", "F", "C", "P", "O", "T"))


def read_edf_bdf(path: str) -> Recording:
    fmt = detect_format(path)
    sample_width = 3 if fmt == "BDF" else 2

    with open(path, "rb") as f:
        main = f.read(256)

        def s(a, b):
            return main[a:b].decode("latin-1").strip()

        patient = s(8, 88)
        recording = s(88, 168)
        startdate = s(168, 176)
        starttime = s(176, 184)
        n_records = int(s(236, 244))
        rec_dur = float(s(244, 252))
        ns = int(s(252, 256))

        sig = f.read(ns * 256)

        def block(off, width):
            base = off * ns
            return [sig[base + i * width: base + (i + 1) * width].decode("latin-1").strip()
                    for i in range(ns)]

        off = 0
        labels = block(off, 16); off += 16
        _transducer = block(off, 80); off += 80
        phys_dim = block(off, 8); off += 8
        phys_min = [float(x) for x in block(off, 8)]; off += 8
        phys_max = [float(x) for x in block(off, 8)]; off += 8
        dig_min = [float(x) for x in block(off, 8)]; off += 8
        dig_max = [float(x) for x in block(off, 8)]; off += 8
        prefilter = block(off, 80); off += 80
        n_samp = [int(x) for x in block(off, 8)]; off += 8

        # scaling factors (physical = (dig - dig_min) * gain + phys_min)
        gains = []
        for i in range(ns):
            d = (dig_max[i] - dig_min[i]) or 1.0
            gains.append((phys_max[i] - phys_min[i]) / d)

        # read all data records; samples are interleaved per record per channel
        raw = f.read()

    # decode little-endian signed integers of `sample_width` bytes
    def decode_ints(buf):
        if sample_width == 2:
            return list(struct.unpack("<%dh" % (len(buf) // 2), buf))
        # 24-bit signed little-endian
        out = []
        for j in range(0, len(buf), 3):
            b0, b1, b2 = buf[j], buf[j + 1], buf[j + 2]
            v = b0 | (b1 << 8) | (b2 << 16)
            if v & 0x800000:
                v -= 0x1000000
            out.append(v)
        return out

    samples_per_record = [n_samp[i] for i in range(ns)]
    record_size = sum(samples_per_record) * sample_width

    # accumulate per-channel sample lists
    chan_data: List[List[float]] = [[] for _ in range(ns)]
    pos = 0
    for _r in range(n_records):
        for i in range(ns):
            nbytes = samples_per_record[i] * sample_width
            chunk = raw[pos:pos + nbytes]
            pos += nbytes
            if not chunk:
                continue
            ints = decode_ints(chunk)
            g = gains[i]
            pmin = phys_min[i]
            dmin = dig_min[i]
            chan_data[i].extend([(v - dmin) * g + pmin for v in ints])

    channels = []
    for i in range(ns):
        kind = _classify_label(labels[i])
        if kind == "ANNOT":
            continue  # skip EDF+ annotation channel for spectral analysis
        fs = samples_per_record[i] / rec_dur if rec_dur else 0.0
        channels.append(Channel(
            label=labels[i],
            fs=fs,
            samples=chan_data[i],
            unit=phys_dim[i] or "uV",
            kind=kind,
            prefilter=prefilter[i],
        ))

    return Recording(
        channels=channels,
        source_path=path,
        source_format=fmt,
        patient=patient,
        start_datetime=f"{startdate} {starttime}".strip(),
        meta={"recording_id": recording, "n_records": n_records,
              "record_duration_s": rec_dur},
    )


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def read_csv(path: str, fs: Optional[float] = None,
             time_col: Optional[str] = None) -> Recording:
    """Read EEG from CSV.

    Flexible layout:
      * First row = channel names (a "time"/"t"/"timestamp" column is detected
        and used to infer fs if `fs` is not given).
      * Remaining rows = samples (one row per time point).
      * Delimiter auto-detected (comma / semicolon / tab).
    """
    with open(path, "r", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = _csv.Sniffer().sniff(sample, delimiters=",;\t")
        except Exception:
            class _D(_csv.Dialect):
                delimiter = ","
                quotechar = '"'
                doublequote = True
                skipinitialspace = True
                lineterminator = "\n"
                quoting = _csv.QUOTE_MINIMAL
            dialect = _D()
        reader = _csv.reader(f, dialect)
        rows = [r for r in reader if r]

    if not rows:
        raise ValueError("Empty CSV: %s" % path)

    header = rows[0]
    # decide whether first row is a header (non-numeric) or data
    def _is_num(x):
        try:
            float(x.replace(",", "."))
            return True
        except Exception:
            return False

    has_header = not all(_is_num(c) for c in header)
    if has_header:
        names = [h.strip() for h in header]
        data_rows = rows[1:]
    else:
        names = ["Ch%d" % (i + 1) for i in range(len(header))]
        data_rows = rows

    # locate time column
    time_idx = None
    for i, n in enumerate(names):
        if time_col and n.lower() == time_col.lower():
            time_idx = i
            break
        if n.lower() in ("time", "t", "timestamp", "sec", "seconds", "ms"):
            time_idx = i

    cols = list(range(len(names)))
    times = None
    if time_idx is not None:
        times = []
        for r in data_rows:
            try:
                times.append(float(r[time_idx].replace(",", ".")))
            except Exception:
                times.append(float("nan"))
        cols = [c for c in cols if c != time_idx]

    if fs is None:
        if times and len(times) > 2:
            dt = (times[-1] - times[0]) / (len(times) - 1)
            # heuristic: if values look like milliseconds, convert
            if dt > 5:  # >5 "seconds" per sample is implausible -> ms
                dt /= 1000.0
            fs = round(1.0 / dt, 4) if dt else 0.0
        else:
            fs = 0.0  # caller must supply

    channels = []
    for c in cols:
        vals = []
        for r in data_rows:
            try:
                vals.append(float(r[c].replace(",", ".")))
            except Exception:
                vals.append(0.0)
        channels.append(Channel(
            label=names[c],
            fs=fs or 0.0,
            samples=vals,
            unit="uV",
            kind=_classify_label(names[c]),
        ))

    return Recording(
        channels=channels,
        source_path=path,
        source_format="CSV",
        meta={"fs_assumed": fs},
    )


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
def read_recording(path: str, **kwargs) -> Recording:
    """Read any supported file into a `Recording` (format auto-detected)."""
    fmt = detect_format(path)
    if fmt in ("EDF", "BDF"):
        return read_edf_bdf(path)
    if fmt == "CSV":
        return read_csv(path, **kwargs)
    raise ValueError("Unsupported format for %s" % path)
