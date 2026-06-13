"""eegspectra - pure-Python spectral analysis toolkit for athlete EEG.

Designed for heterogeneous datasets:
  * Multiple devices  (CONTEC KT-88 16-ch, Neurotech "Compact-neuro" 21-ch, ...)
  * Multiple formats  (.edf, .bdf, .csv, ...)
  * Arbitrary channel counts and sampling rates

No third-party dependencies (numpy/scipy/mne not required) so it runs anywhere.
"""

from .core import Recording, Channel
from .io import read_recording, SUPPORTED_FORMATS
from .montage import detect_device, classify_channels
from .dsp import welch_psd, bandpass
from .bands import BANDS, band_powers
from .indices import sport_indices

__version__ = "0.1.0"

__all__ = [
    "Recording",
    "Channel",
    "read_recording",
    "SUPPORTED_FORMATS",
    "detect_device",
    "classify_channels",
    "welch_psd",
    "bandpass",
    "BANDS",
    "band_powers",
    "sport_indices",
    "__version__",
]
