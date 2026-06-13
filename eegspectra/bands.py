"""EEG frequency bands and per-channel band-power computation."""
from typing import Dict, List, Sequence
from .core import Recording, Channel
from .dsp import welch_psd, band_integrate

# Standard clinical/research band limits (Hz). Gamma upper bound is capped to
# the available Nyquist frequency at analysis time.
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def band_powers(samples: Sequence[float], fs: float,
                bands: Dict[str, tuple] = None,
                nperseg: int = None) -> Dict[str, float]:
    """Absolute power per band for one channel (uV^2)."""
    bands = bands or BANDS
    freqs, psd = welch_psd(samples, fs, nperseg=nperseg)
    if not freqs:
        return {b: 0.0 for b in bands}
    nyq = fs / 2.0
    out = {}
    for name, (lo, hi) in bands.items():
        out[name] = band_integrate(freqs, psd, lo, min(hi, nyq))
    return out


def relative_band_powers(abs_powers: Dict[str, float]) -> Dict[str, float]:
    """Convert absolute band powers to fractions of total band power."""
    total = sum(abs_powers.values()) or 1.0
    return {b: v / total for b, v in abs_powers.items()}


def channel_band_table(rec: Recording, nperseg: int = None) -> List[dict]:
    """Compute absolute + relative band powers for every EEG channel."""
    rows = []
    for c in rec.eeg_channels():
        absp = band_powers(c.samples, c.fs, nperseg=nperseg)
        relp = relative_band_powers(absp)
        rows.append({
            "channel": c.name or c.label,
            "label": c.label,
            "abs": absp,
            "rel": relp,
        })
    return rows


def average_band_powers(rec: Recording, nperseg: int = None) -> Dict[str, Dict[str, float]]:
    """Dataset-/recording-level mean absolute and relative band powers."""
    rows = channel_band_table(rec, nperseg=nperseg)
    if not rows:
        return {"abs": {b: 0.0 for b in BANDS}, "rel": {b: 0.0 for b in BANDS}}
    abs_mean = {b: 0.0 for b in BANDS}
    rel_mean = {b: 0.0 for b in BANDS}
    for r in rows:
        for b in BANDS:
            abs_mean[b] += r["abs"][b]
            rel_mean[b] += r["rel"][b]
    n = len(rows)
    abs_mean = {b: v / n for b, v in abs_mean.items()}
    rel_mean = {b: v / n for b, v in rel_mean.items()}
    return {"abs": abs_mean, "rel": rel_mean}
