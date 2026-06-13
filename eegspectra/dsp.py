"""Pure-Python digital signal processing core (no numpy/scipy).

Provides:
  * iterative radix-2 FFT (+ Bluestein for arbitrary lengths)
  * detrend / mean-removal
  * window functions (Hann, Hamming, rectangular)
  * Welch power-spectral-density estimate
  * Butterworth-style band-pass via cascaded biquads (RBJ) for filtering
"""
import math
import cmath
from typing import List, Tuple, Sequence


# --------------------------------------------------------------------------- #
# FFT
# --------------------------------------------------------------------------- #
def _fft_radix2(a: List[complex]) -> List[complex]:
    n = len(a)
    if n == 1:
        return a
    # bit-reversal permutation
    levels = n.bit_length() - 1
    out = a[:]
    for i in range(n):
        j = int('{:0{w}b}'.format(i, w=levels)[::-1], 2)
        if j > i:
            out[i], out[j] = out[j], out[i]
    size = 2
    while size <= n:
        half = size // 2
        step = -2j * math.pi / size
        twiddles = [cmath.exp(step * k) for k in range(half)]
        for start in range(0, n, size):
            for k in range(half):
                t = twiddles[k] * out[start + k + half]
                u = out[start + k]
                out[start + k] = u + t
                out[start + k + half] = u - t
        size *= 2
    return out


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _bluestein(a: List[complex]) -> List[complex]:
    n = len(a)
    m = _next_pow2(2 * n + 1)
    w = [cmath.exp(-1j * math.pi * (i * i % (2 * n)) / n) for i in range(n)]
    av = [a[i] * w[i] for i in range(n)] + [0j] * (m - n)
    bv = [0j] * m
    for i in range(n):
        bv[i] = w[i].conjugate()
        if i:
            bv[m - i] = w[i].conjugate()
    fa = _fft_radix2(av)
    fb = _fft_radix2(bv)
    fc = [fa[i] * fb[i] for i in range(m)]
    cc = _ifft_radix2(fc)
    return [cc[i] * w[i] for i in range(n)]


def _ifft_radix2(a: List[complex]) -> List[complex]:
    n = len(a)
    conj = [x.conjugate() for x in a]
    y = _fft_radix2(conj)
    return [x.conjugate() / n for x in y]


def fft(a: Sequence[float]) -> List[complex]:
    """FFT of a real or complex sequence of any length."""
    x = [complex(v) for v in a]
    n = len(x)
    if n == 0:
        return []
    if n & (n - 1) == 0:
        return _fft_radix2(x)
    return _bluestein(x)


# --------------------------------------------------------------------------- #
# Pre-processing helpers
# --------------------------------------------------------------------------- #
def detrend(x: Sequence[float], mode: str = "mean") -> List[float]:
    n = len(x)
    if n == 0:
        return []
    if mode == "mean":
        m = sum(x) / n
        return [v - m for v in x]
    if mode == "linear":
        # least-squares line removal
        mx = (n - 1) / 2.0
        my = sum(x) / n
        sxx = sum((i - mx) ** 2 for i in range(n))
        sxy = sum((i - mx) * (x[i] - my) for i in range(n))
        slope = sxy / sxx if sxx else 0.0
        intercept = my - slope * mx
        return [x[i] - (slope * i + intercept) for i in range(n)]
    return list(x)


def window(n: int, kind: str = "hann") -> List[float]:
    if n <= 1:
        return [1.0] * n
    if kind == "hann":
        return [0.5 - 0.5 * math.cos(2 * math.pi * i / (n - 1)) for i in range(n)]
    if kind == "hamming":
        return [0.54 - 0.46 * math.cos(2 * math.pi * i / (n - 1)) for i in range(n)]
    return [1.0] * n  # rectangular


# --------------------------------------------------------------------------- #
# Welch PSD
# --------------------------------------------------------------------------- #
def welch_psd(x: Sequence[float], fs: float,
              nperseg: int = None, overlap: float = 0.5,
              win: str = "hann", detrend_mode: str = "linear"
              ) -> Tuple[List[float], List[float]]:
    """Estimate one-sided power spectral density (Welch's method).

    Returns (freqs, psd) where psd is in units of (signal_unit^2 / Hz).
    """
    n = len(x)
    if n == 0 or fs <= 0:
        return [], []
    if nperseg is None:
        nperseg = min(n, _next_pow2(int(fs * 2)))  # ~2-second segments
    nperseg = min(nperseg, n)
    if nperseg < 8:
        nperseg = n
    step = max(1, int(nperseg * (1 - overlap)))
    w = window(nperseg, win)
    # window power normalization (for density)
    u = sum(v * v for v in w)
    scale = 1.0 / (fs * u) if (fs * u) else 0.0

    nfreq = nperseg // 2 + 1
    acc = [0.0] * nfreq
    nseg = 0
    start = 0
    while start + nperseg <= n:
        seg = detrend(x[start:start + nperseg], detrend_mode)
        seg = [seg[i] * w[i] for i in range(nperseg)]
        X = fft(seg)
        for k in range(nfreq):
            mag2 = (X[k].real * X[k].real + X[k].imag * X[k].imag)
            # one-sided: double all but DC and Nyquist
            if k != 0 and not (nperseg % 2 == 0 and k == nfreq - 1):
                mag2 *= 2.0
            acc[k] += mag2 * scale
        nseg += 1
        start += step

    if nseg == 0:  # signal shorter than one segment
        seg = detrend(list(x), detrend_mode)
        L = len(seg)
        wv = window(L, win)
        uu = sum(v * v for v in wv) or 1.0
        seg = [seg[i] * wv[i] for i in range(L)]
        X = fft(seg)
        nfreq = L // 2 + 1
        acc = [0.0] * nfreq
        sc = 1.0 / (fs * uu)
        for k in range(nfreq):
            mag2 = X[k].real ** 2 + X[k].imag ** 2
            if k != 0 and not (L % 2 == 0 and k == nfreq - 1):
                mag2 *= 2.0
            acc[k] = mag2 * sc
        freqs = [k * fs / L for k in range(nfreq)]
        return freqs, acc

    psd = [a / nseg for a in acc]
    freqs = [k * fs / nperseg for k in range(nfreq)]
    return freqs, psd


# --------------------------------------------------------------------------- #
# Band-pass filtering (RBJ biquads, zero-phase via forward-backward)
# --------------------------------------------------------------------------- #
def _biquad(x: Sequence[float], b, a) -> List[float]:
    b0, b1, b2 = b
    a0, a1, a2 = a
    b0, b1, b2 = b0 / a0, b1 / a0, b2 / a0
    a1, a2 = a1 / a0, a2 / a0
    y = [0.0] * len(x)
    x1 = x2 = y1 = y2 = 0.0
    for i, xn in enumerate(x):
        yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[i] = yn
        x2, x1 = x1, xn
        y2, y1 = y1, yn
    return y


def _bandpass_biquad(fs, f_lo, f_hi):
    f0 = math.sqrt(max(f_lo, 1e-6) * f_hi)
    bw = max(f_hi - f_lo, 1e-6)
    q = f0 / bw
    w0 = 2 * math.pi * f0 / fs
    alpha = math.sin(w0) / (2 * q)
    b = [alpha, 0.0, -alpha]
    a = [1 + alpha, -2 * math.cos(w0), 1 - alpha]
    return b, a


def bandpass(x: Sequence[float], fs: float, f_lo: float, f_hi: float,
             zero_phase: bool = True) -> List[float]:
    """Band-pass filter via an RBJ biquad; zero-phase by forward-backward pass."""
    if fs <= 0 or not x:
        return list(x)
    f_hi = min(f_hi, fs / 2 * 0.99)
    b, a = _bandpass_biquad(fs, f_lo, f_hi)
    y = _biquad(x, b, a)
    if zero_phase:
        y = _biquad(y[::-1], b, a)[::-1]
    return y


def band_integrate(freqs: Sequence[float], psd: Sequence[float],
                   f_lo: float, f_hi: float) -> float:
    """Integrate PSD over [f_lo, f_hi] using the trapezoid rule -> absolute power."""
    total = 0.0
    for i in range(1, len(freqs)):
        f0, f1 = freqs[i - 1], freqs[i]
        if f1 < f_lo or f0 > f_hi:
            continue
        lo = max(f0, f_lo)
        hi = min(f1, f_hi)
        if hi <= lo:
            continue
        # linear interpolation of psd at lo, hi
        def interp(ff):
            if f1 == f0:
                return psd[i]
            t = (ff - f0) / (f1 - f0)
            return psd[i - 1] + t * (psd[i] - psd[i - 1])
        p_lo, p_hi = interp(lo), interp(hi)
        total += (p_lo + p_hi) / 2 * (hi - lo)
    return total
