# eegspectra — Spectral analysis toolkit for athlete EEG

A dependency-free (pure standard-library Python) toolkit for **spectral analysis
of athletes' electroencephalographic (EEG) signals**. It was built to work
directly with the heterogeneous real recordings in this repository and is
designed to generalize to other devices, formats and electrode montages.

> Research context: *"Development of an algorithm and software tool for spectral
> analysis of athletes' EEG signals."* The toolkit turns raw multi-device EEG
> into interpretable, sport-relevant spectral indices.

## Why it exists (the gap it fills)

General EEG packages (EEGLAB, MNE-Python) are powerful but assume a single
montage workflow and require programming. This toolkit:

- ingests **multiple devices** out of the box and auto-detects them,
- reads **multiple file formats** with a single API,
- supports **arbitrary channel counts** (16, 21, or any other),
- outputs **sport-specific indices** a coach / sports physician can read,
- has **zero third-party dependencies**, so it runs anywhere.

## Supported inputs

| Format | Notes |
|--------|-------|
| `.edf` / EDF+ | 16-bit; EDF+ annotation channel auto-skipped |
| `.bdf` / BDF+ | 24-bit (BioSemi); decoded natively |
| `.csv` | auto delimiter (`, ; \t`), optional `time` column → infers `fs` |

| Device (auto-detected) | Channels | Typical fs |
|------------------------|----------|------------|
| **CONTEC KT-88** | 16 EEG (ear-referenced, `Fp1-A1`…) | 100 Hz |
| **Нейротех «Компакт-нейро»** | 21 EEG (`Fp1`…`Oz`) + ECG/EMG/EOG | 500 Hz |
| generic | any count | any |

Unknown montages fall back to a generic 10-20 normalizer, so the pipeline keeps
working even on devices it has never seen.

## Processing pipeline

1. **Read** → unified `Recording` object (`io.py`).
2. **Montage** → normalize labels to 10-20 names, detect device (`montage.py`).
3. **DSP** → linear detrend → Hann window → FFT (radix-2 + Bluestein for any
   length) → **Welch PSD** (≈2 s segments, 50 % overlap) (`dsp.py`).
4. **Band powers** → δ, θ, α, β, γ absolute + relative power (`bands.py`).
5. **Sport indices** → derived metrics + plain-language hints (`indices.py`).

### Frequency bands

| Band | Hz |
|------|----|
| delta | 0.5–4 |
| theta | 4–8 |
| alpha | 8–13 |
| beta  | 13–30 |
| gamma | 30–45 (capped at Nyquist) |

### Sport spectral indices

| Index | Formula | Meaning |
|-------|---------|---------|
| Neural Efficiency Index | mean relative α | calm, efficient "ready" state |
| Mental Fatigue Index | (θ+α)/β | rises with fatigue / low arousal |
| Attention/Engagement Index | β/(α+θ) | cognitive engagement |
| Theta/Beta Ratio | θ/β | sustained-attention marker |
| Frontal midline theta | mean θ over frontal sites | attentional control |
| Parieto-occipital alpha | mean α over P/O sites | relaxation / idling |

> Indices are heuristic research markers, **not** clinical diagnoses.

## Command-line usage

```bash
# quick header inspection (no spectral computation)
python3 -m eegspectra info "*.EDF" "*.BDF"

# full per-channel + index report for one file (+ optional CSV export)
python3 -m eegspectra analyze 0000002.EDF --csv channel_bandpowers.csv

# batch a whole cohort -> one summary row per recording
python3 -m eegspectra batch "*.EDF" "*.edf" --csv cohort.csv
python3 -m eegspectra batch . --csv cohort.csv
```

## Python API

```python
from eegspectra import read_recording, classify_channels
from eegspectra.bands import average_band_powers
from eegspectra.indices import sport_indices, interpret

rec = classify_channels(read_recording("0000001.EDF"))
print(rec.summary())                      # device, channels, fs, duration
print(average_band_powers(rec)["rel"])    # mean relative band powers
idx = sport_indices(rec)
print(idx, interpret(idx))
```

## Module layout

```
eegspectra/
  core.py       Recording / Channel data model
  io.py         EDF / BDF / CSV readers + format detection
  montage.py    10-20 normalization + device detection
  dsp.py        FFT, detrend, windows, Welch PSD, band-pass filter
  bands.py      band-power computation (absolute + relative)
  indices.py    sport-specific spectral indices + interpretation
  cli.py        info / analyze / batch commands
```

## Limitations / next steps

- Artifact handling is currently limited to detrending + windowing; ICA/ASR-style
  rejection is a planned extension.
- Time-frequency (wavelet/STFT) dynamics during movement is the next algorithmic
  contribution to add on top of the static Welch estimate.
