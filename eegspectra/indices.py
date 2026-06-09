"""Sport-specific spectral indices derived from EEG band powers.

These indices operationalize concepts from sports-neuroscience literature so a
coach / sports physician gets interpretable numbers rather than raw spectra:

  * Neural Efficiency Index (NEI)
        Higher resting/alpha relative power is associated with the neural
        efficiency hypothesis (more efficient, less "noisy" cortical activity).
        NEI = relative alpha power (averaged over channels), 0..1.

  * Mental Fatigue Index (MFI)
        Slow-wave drift relative to fast activity rises with fatigue.
        MFI = (theta + alpha) / beta  (theta/beta variants also common).

  * Attention / Engagement Index (AEI)
        Classic engagement index from neuroergonomics.
        AEI = beta / (alpha + theta).

  * Theta/Beta Ratio (TBR)
        Widely used marker of attentional/arousal state.

All indices are computed from absolute band powers averaged across EEG channels,
so they work for any channel count (16, 21, ...). Region-specific variants
(frontal theta, parieto-occipital alpha) are provided when those electrodes
exist in the montage.
"""
from typing import Dict, List
from .core import Recording
from .bands import channel_band_table

FRONTAL = {"FP1", "FP2", "FPZ", "F3", "F4", "FZ", "F7", "F8", "AF3", "AF4"}
PARIETO_OCC = {"P3", "P4", "PZ", "O1", "O2", "OZ", "T5", "T6", "P7", "P8"}


def _safe_div(a, b):
    return a / b if b else 0.0


def sport_indices(rec: Recording, nperseg: int = None) -> Dict[str, float]:
    rows = channel_band_table(rec, nperseg=nperseg)
    if not rows:
        return {}

    # whole-head means of absolute power
    def mean_abs(band, subset=None):
        vals = []
        for r in rows:
            if subset is None or r["channel"].upper() in subset:
                vals.append(r["abs"][band])
        return sum(vals) / len(vals) if vals else 0.0

    def mean_rel(band, subset=None):
        vals = []
        for r in rows:
            if subset is None or r["channel"].upper() in subset:
                vals.append(r["rel"][band])
        return sum(vals) / len(vals) if vals else 0.0

    delta = mean_abs("delta"); theta = mean_abs("theta")
    alpha = mean_abs("alpha"); beta = mean_abs("beta"); gamma = mean_abs("gamma")

    idx = {
        "neural_efficiency_index": mean_rel("alpha"),
        "mental_fatigue_index": _safe_div(theta + alpha, beta),
        "attention_engagement_index": _safe_div(beta, alpha + theta),
        "theta_beta_ratio": _safe_div(theta, beta),
        "alpha_beta_ratio": _safe_div(alpha, beta),
    }

    # region-specific markers (only if those electrodes are present)
    frontal_theta = mean_abs("theta", FRONTAL)
    po_alpha = mean_abs("alpha", PARIETO_OCC)
    if frontal_theta:
        idx["frontal_midline_theta"] = frontal_theta
    if po_alpha:
        idx["parieto_occipital_alpha"] = po_alpha

    return idx


def interpret(idx: Dict[str, float]) -> List[str]:
    """Plain-language hints for a coach/physician (heuristic, not diagnostic)."""
    notes = []
    mfi = idx.get("mental_fatigue_index")
    if mfi is not None:
        if mfi > 3.0:
            notes.append("High (theta+alpha)/beta -> possible mental fatigue / low arousal.")
        elif mfi < 1.0:
            notes.append("Low (theta+alpha)/beta -> high arousal / activation.")
    nei = idx.get("neural_efficiency_index")
    if nei is not None and nei > 0.35:
        notes.append("Elevated relative alpha -> calm, efficient 'ready' state.")
    tbr = idx.get("theta_beta_ratio")
    if tbr is not None and tbr > 3.0:
        notes.append("High theta/beta ratio -> reduced sustained attention.")
    return notes
