"""Unified in-memory data model for an EEG recording.

A `Recording` is format-agnostic: EDF, BDF and CSV readers all produce the same
object so the rest of the pipeline never needs to know where the data came from.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Channel:
    """A single signal/lead within a recording."""
    label: str                       # raw label as stored, e.g. "EEG Fp1" or "Fp1-A1"
    fs: float                        # sampling rate (Hz)
    samples: List[float]             # physical values (microvolts for EEG)
    unit: str = "uV"                 # physical dimension
    kind: str = "EEG"                # EEG / ECG / EMG / EOG / AUX / ANNOT
    name: str = ""                   # normalized 10-20 name, e.g. "Fp1" (filled by montage)
    prefilter: str = ""              # device prefilter description

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def duration(self) -> float:
        return self.n_samples / self.fs if self.fs else 0.0


@dataclass
class Recording:
    """A complete recording from any supported device/format."""
    channels: List[Channel] = field(default_factory=list)
    source_path: str = ""
    source_format: str = ""          # "EDF" / "BDF" / "CSV"
    patient: str = ""
    start_datetime: str = ""
    device: str = "unknown"          # filled by montage.detect_device
    meta: Dict = field(default_factory=dict)

    # ---- convenience accessors -------------------------------------------
    def eeg_channels(self) -> List[Channel]:
        return [c for c in self.channels if c.kind == "EEG"]

    def labels(self) -> List[str]:
        return [c.label for c in self.channels]

    def get(self, name: str) -> Optional[Channel]:
        """Lookup by normalized 10-20 name (case-insensitive), then raw label."""
        target = name.strip().lower()
        for c in self.channels:
            if c.name.lower() == target:
                return c
        for c in self.channels:
            if c.label.lower() == target:
                return c
        return None

    @property
    def fs(self) -> float:
        """Dominant EEG sampling rate."""
        eeg = self.eeg_channels()
        rates = [c.fs for c in eeg] if eeg else [c.fs for c in self.channels]
        if not rates:
            return 0.0
        return max(set(rates), key=rates.count)

    @property
    def duration(self) -> float:
        return max((c.duration for c in self.channels), default=0.0)

    def summary(self) -> str:
        eeg = self.eeg_channels()
        return (
            f"{self.source_format} | device={self.device} | "
            f"{len(eeg)} EEG ch | fs={self.fs:g} Hz | "
            f"{self.duration:g} s | patient={self.patient or '-'}"
        )
