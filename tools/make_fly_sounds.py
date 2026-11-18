#!/usr/bin/env python3
"""
make_fly_sounds.py
Procedural housefly (Musca domestica) sound synthesis.

Outputs (16-bit PCM / 44.1 kHz / mono / WAV):
  buzz_hover.wav   2.0 s  seamless loop    ~190 Hz wingbeat, calm
  buzz_fly.wav     2.0 s  seamless loop    ~215 Hz wingbeat, brighter
  buzz_dart.wav    0.6 s  one-shot         pitch rise+fall, amp burst
  takeoff.wav      0.5 s  one-shot         wings spin-up 120->190 Hz
  landing.wav      0.6 s  one-shot         pitch falls 190->90 Hz + tick
  walk_steps.wav   1.5 s  seamless loop    tiny leg-ticks on glass
  groom.wav        1.2 s  seamless loop    soft dry rustle
  preview_rates.wav        one-shot         buzz_fly at 0.8/1.0/1.2/1.5x

Requires: numpy (stdlib otherwise).
"""

import os
import wave
import numpy as np

SR = 44100
SEED = 20240517


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------
def db_to_lin(db):
    return 10.0 ** (db / 20.0)


def lin_to_db(x):
    return 20.0 * np.log10(max(abs(x), 1e-12))


def normalize_peak(x, peak_db):
    peak = float(np.max(np.abs(x)))
    if peak < 1e-12:
        return x
    return x * (db_to_lin(peak_db) / peak)


def remove_dc(x):
    return x - float(np.mean(x))


def soft_clip(x, drive=1.0):
    return np.tanh(drive * x) / drive


def write_wav(path, x, sr=SR):
    x = np.clip(x, -1.0, 1.0)
    pcm = np.round(x * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        nf = w.getnframes()
        raw = w.readframes(nf)
    data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    return sr, ch, sw * 8, nf, data


# ----------------------------------------------------------------------------
# Loop-seam helper
# ----------------------------------------------------------------------------
def make_loop_seamless(x, crossfade_ms=0.0):
    """
    If requested, apply a short crossfade at the loop seam.
    Our synthesizers are designed to already be periodic; the crossfade is a
    safety net only when crossfade_ms > 0 and the jump exceeds the local slope.
    """
    if crossfade_ms <= 0.0:
        return x
    n = int(round(crossfade_ms * 1e-3 * SR))
    n = min(n, len(x) // 8)
    if n < 4:
        return x
    seam = abs(x[0] - x[-1])
    typical = float(np.median(np.abs(np.diff(x)))) + 1e-12
    if seam <= 4.0 * typical:
        return x
    fade = np.linspace(0.0, 1.0, n)
    head = x[:n].copy()
    tail = x[-n:].copy()
    x = x.copy()
    x[:n] = head * fade + tail * (1.0 - fade)
    return x


# ----------------------------------------------------------------------------
# Periodic building blocks (guaranteed period == T)
# ----------------------------------------------------------------------------
def make_periodic_lfo(T, sr, f_lo, f_hi, n_comp, rng):
    """
    Returns (lfo, lfo_integral) both exactly periodic with period T.
    lfo is roughly zero-mean and peaked near +/-1.
    lfo_integral(t) = ∫_0^t lfo(τ) dτ  (analytic, exactly 0 at t=0 and t=T).
    """
    N = int(round(T * sr))
    t = np.arange(N) / sr
    lfo = np.zeros(N)
    integral = np.zeros(N)
    comps = []
    for _ in range(n_comp):
        cycles = max(1, int(round(rng.uniform(f_lo, f_hi) * T)))
        f = cycles / T
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amp = rng.uniform(0.3, 1.0)
        comps.append((f, amp, phase))
    total = sum(a for _, a, _ in comps)
    for f, a, phase in comps:
        a_n = a / total
        arg = 2.0 * np.pi * f * t + phase
        lfo += a_n * np.sin(arg)
        # ∫ sin(2π f τ + φ) dτ  from 0 to t
        integral += a_n * (np.cos(phase) - np.cos(arg)) / (2.0 * np.pi * f)
    return lfo, integral


def make_periodic_noise(T, sr, f_lo, f_hi, rng, slope=0.0):
    """
    Band-limited noise that is exactly periodic with period T.
    Produced via inverse FFT of random-phase spectrum -> loop-safe.
    """
    N = int(round(T * sr))
    freqs = np.fft.rfftfreq(N, 1.0 / sr)
    mag = np.zeros_like(freqs)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    if slope != 0.0:
        mag[band] = (freqs[band] / max(f_lo, 1e-3)) ** slope
    else:
        mag[band] = 1.0
    phases = rng.uniform(0.0, 2.0 * np.pi, len(freqs))
    spec = mag * np.exp(1j * phases)
    spec[0] = 0.0
    if N % 2 == 0:
        spec[-1] = spec[-1].real
    noise = np.fft.irfft(spec, N)
    return noise / (np.std(noise) + 1e-12)


# ----------------------------------------------------------------------------
# Biquad band-pass (RBJ cookbook) -- direct form for one-shots
# ----------------------------------------------------------------------------
def biquad_bandpass(x, f0, Q, sr=SR):
    w0 = 2.0 * np.pi * f0 / sr
    cw = np.cos(w0)
    sw = np.sin(w0)
    alpha = sw / (2.0 * Q)
    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * cw
    a2 = 1.0 - alpha
    b0 /= a0; b1 /= a0; b2 /= a0; a1 /= a0; a2 /= a0
    N = len(x)
    y = np.zeros(N)
    x1 = x2 = 0.0
    y1 = y2 = 0.0
    for i in range(N):
        v = b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[i] = v
        x2 = x1; x1 = x[i]
        y2 = y1; y1 = v
    return y


def biquad_bp_response(f0, Q, sr, freqs):
    w0 = 2.0 * np.pi * f0 / sr
    cw = np.cos(w0)
    sw = np.sin(w0)
    alpha = sw / (2.0 * Q)
    b0 = alpha; b1 = 0.0; b2 = -alpha
    a0 = 1.0 + alpha; a1 = -2.0 * cw; a2 = 1.0 - alpha
    b0 /= a0; b1 /= a0; b2 /= a0; a1 /= a0; a2 /= a0
    zinv = np.exp(-1j * 2.0 * np.pi * freqs / sr)
    H = (b0 + b1 * zinv + b2 * zinv ** 2) / (1.0 + a1 * zinv + a2 * zinv ** 2)
    return H


def apply_resonances_loop(x, resonances, sr=SR):
    """Parallel body resonances applied in the frequency domain (exact loop)."""
    N = len(x)
    freqs = np.fft.rfftfreq(N, 1.0 / sr)
    X = np.fft.rfft(x)
    H_total = np.ones_like(X, dtype=complex)
    for f0, Q, gain in resonances:
        H_total += gain * biquad_bp_response(f0, Q, sr, freqs)
    return np.fft.irfft(X * H_total, N)


def apply_resonances_oneshot(x, resonances, sr=SR):
    """Parallel body resonances via direct biquad (transient-accurate)."""
    out = x.copy()
    for f0, Q, gain in resonances:
        out += gain * biquad_bandpass(x, f0, Q, sr)
    return out


# ----------------------------------------------------------------------------
# Harmonic oscillator
# ----------------------------------------------------------------------------
def harmonic_oscillator(phase, n_harm, rolloff, phases):
    """Sum of n_harm harmonics of a fundamental whose instantaneous phase is
    supplied (radians). Amplitudes decay as 1/n**rolloff."""
    sig = np.zeros_like(phase)
    for n in range(1, n_harm + 1):
        sig += (1.0 / n ** rolloff) * np.sin(n * phase + phases[n - 1])
    return sig


# ----------------------------------------------------------------------------
# Wingbeat synthesis
# ----------------------------------------------------------------------------
BODY_RES = [(350.0, 2.0, 0.45),
            (800.0, 2.5, 0.35),
            (1600.0, 3.0, 0.22)]


def synth_loop_wingbeat(f0_target, T, sr, seed, rolloff=1.2, n_harm=24,
                        air_db=-30.0, freq_dev=0.015, amp_dev=0.12):
    """
    Seamless loop of a two-wing buzzy oscillator + wobble + air + body resonance.
    Everything is built with frequencies that are exact integer multiples of 1/T,
    so the returned buffer is EXACTLY periodic with period T.
    """
    rng = np.random.default_rng(seed)
    N = int(round(T * sr))
    t = np.arange(N) / sr

    # Quantize fundamentals to integer cycles per T -> exact loop.
    f0 = round(f0_target * T) / T
    df = max(1, int(round(1.5 * T))) / T  # ~1.5 Hz wing-to-wing detune

    # Slow LFOs (both periodic).
    w_freq, W_freq = make_periodic_lfo(T, sr, 0.5, 6.0, 5, rng)
    w_amp, _ = make_periodic_lfo(T, sr, 0.5, 6.0, 5, rng)

    # Tiny per-harmonic phase jitter (integer-cycle slow modulators => loopable).
    jit_cycles = np.maximum(1, np.round(rng.uniform(1.0, 5.0, n_harm) * T).astype(int))
    jit_phases = rng.uniform(0.0, 2.0 * np.pi, n_harm)
    jit_amps = rng.uniform(5e-4, 2e-3, n_harm)

    def make_wing(f0_wing, seed_w):
        rng_w = np.random.default_rng(seed_w)
        phases_h = rng_w.uniform(0.0, 2.0 * np.pi, n_harm)
        sig = np.zeros(N)
        base = 2.0 * np.pi * f0_wing * t
        fm = n_harm * 0.0  # placeholder
        for n in range(1, n_harm + 1):
            # freq modulation scaled by n (so % deviation is constant per harmonic)
            ph = n * base
            ph = ph + n * 2.0 * np.pi * f0 * freq_dev * W_freq
            ph = ph + phases_h[n - 1]
            jf = jit_cycles[n - 1] / T
            ph = ph + jit_amps[n - 1] * np.sin(2.0 * np.pi * jf * t + jit_phases[n - 1])
            sig += (1.0 / n ** rolloff) * np.sin(ph)
        return sig

    wing1 = make_wing(f0, seed * 13 + 1)
    wing2 = make_wing(f0 + df, seed * 13 + 2)
    sig = wing1 + wing2
    sig *= (1.0 + amp_dev * w_amp)

    # Air turbulence: band-limited 1-6 kHz, -30 dB relative to buzz RMS.
    air = make_periodic_noise(T, sr, 1000.0, 6000.0, rng)
    rms_sig = np.sqrt(np.mean(sig ** 2)) + 1e-12
    air *= (rms_sig * db_to_lin(air_db)) / (np.std(air) + 1e-12)
    sig = sig + air

    # Body resonance (applied via FFT -> exact periodic output).
    sig = apply_resonances_loop(sig, BODY_RES, sr)

    sig = make_loop_seamless(sig, crossfade_ms=0.0)
    return sig


def synth_one_shot_wingbeat(freq_fn, amp_fn, T, sr, seed,
                            n_harm=24, rolloff=1.2, air_db=-30.0):
    """One-shot wingbeat with arbitrary pitch & amplitude envelopes."""
    rng = np.random.default_rng(seed)
    N = int(round(T * sr))
    t = np.arange(N) / sr
    dt = 1.0 / sr

    freq = freq_fn(t)
    amp = amp_fn(t)

    # Small natural wobble on pitch (short one-shot: free-form LFO is fine).
    f_wob = np.zeros(N)
    for _ in range(3):
        ff = rng.uniform(2.0, 6.0)
        f_wob += rng.uniform(0.3, 1.0) * np.sin(2.0 * np.pi * ff * t + rng.uniform(0.0, 2.0 * np.pi))
    f_wob /= 3.0
    freq = freq * (1.0 + 0.015 * f_wob)

    phase = 2.0 * np.pi * np.cumsum(freq) * dt
    phase -= phase[0]

    df = 1.5  # Hz detune between wings
    phase2 = phase + 2.0 * np.pi * df * t

    ph1 = rng.uniform(0.0, 2.0 * np.pi, n_harm)
    ph2 = rng.uniform(0.0, 2.0 * np.pi, n_harm)
    sig = harmonic_oscillator(phase, n_harm, rolloff, ph1) \
        + harmonic_oscillator(phase2, n_harm, rolloff, ph2)
    sig *= amp

    # Air noise (one-shot via zero-padded FFT bandpass).
    air = rng.standard_normal(N)
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    spec = np.fft.rfft(air, n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mask = (freqs >= 1000.0) & (freqs <= 6000.0)
    spec[~mask] = 0.0
    air = np.fft.irfft(spec, n_fft)[:N]
    rms_sig = np.sqrt(np.mean(sig ** 2)) + 1e-12
    air *= (rms_sig * db_to_lin(air_db)) / (np.std(air) + 1e-12)
    sig = sig + air

    sig = apply_resonances_oneshot(sig, BODY_RES, sr)
    return sig


# ----------------------------------------------------------------------------
# One-shot envelopes
# ----------------------------------------------------------------------------
def dart_freq(t):
    T = 0.6
    u = t / T
    return 215.0 + 75.0 * np.exp(-((u - 0.40) / 0.15) ** 2)


def dart_amp(t):
    T = 0.6
    u = t / T
    rise = 1.0 - np.exp(-20.0 * u)
    burst = np.exp(-((u - 0.35) ** 2) / (2.0 * 0.22 ** 2))
    tail = np.exp(-1.2 * np.maximum(0.0, u - 0.5))
    return rise * burst * tail


def takeoff_freq(t):
    T = 0.5
    u = t / T
    return 120.0 + 70.0 * (1.0 - np.exp(-5.0 * u))


def takeoff_amp(t):
    T = 0.5
    u = t / T
    return (1.0 - np.exp(-9.0 * u)) * (1.0 - 0.15 * u)


def landing_freq(t):
    T = 0.6
    u = t / T
    return 190.0 - 100.0 * (1.0 - np.exp(-4.0 * u))


def landing_amp(t):
    T = 0.6
    u = t / T
    return np.exp(-3.0 * u) * (1.0 - np.exp(-8.0 * u))


# ----------------------------------------------------------------------------
# Walk-steps & groom (loops)
# ----------------------------------------------------------------------------
def make_leg_tick(dur_ms, sr, rng):
    N = max(4, int(round(dur_ms * 1e-3 * sr)))
    t = np.arange(N) / sr
    env = np.exp(-t / (dur_ms * 1e-3) * 4.0)
    f = rng.uniform(1500.0, 4000.0)
    sig = env * np.sin(2.0 * np.pi * f * t + rng.uniform(0.0, 2.0 * np.pi))
    sig += 0.5 * env * rng.standard_normal(N)
    sig = np.diff(np.concatenate([[0.0], sig]))  # crude high-pass
    peak = np.max(np.abs(sig)) + 1e-12
    sig /= peak
    sig *= rng.uniform(0.3, 1.0)
    return sig


def synth_walk_steps(sr, seed):
    rng = np.random.default_rng(seed)
    T = 1.5
    N = int(round(T * sr))
    buf = np.zeros(N)

    # 6 legs + a couple of re-steps -> irregular timing.
    n_ticks = 8
    positions = np.sort(rng.uniform(0.0, 1.0, n_ticks)) * N
    positions = positions.astype(int)
    for pos in positions:
        dur_ms = rng.uniform(2.0, 6.0)
        tick = make_leg_tick(dur_ms, sr, rng)
        idx = (pos + np.arange(len(tick))) % N
        np.add.at(buf, idx, tick)

    return make_loop_seamless(buf, crossfade_ms=0.0)


def synth_groom(sr, seed):
    rng = np.random.default_rng(seed)
    T = 1.2
    base = make_periodic_noise(T, sr, 1500.0, 5000.0, rng, slope=-0.3)
    am_raw = make_periodic_noise(T, sr, 8.0, 40.0, rng)
    am_env = 0.55 + 0.45 * np.tanh(1.5 * am_raw)
    am_env = np.clip(am_env, 0.05, 1.0)
    sig = base * am_env
    return make_loop_seamless(sig, crossfade_ms=0.0)


# ----------------------------------------------------------------------------
# Preview of playback rate scaling
# ----------------------------------------------------------------------------
def resample_linear(x, rate):
    """Playback-rate resampling (rate>1 = faster & higher)."""
    n_out = int(round(len(x) / rate))
    t_out = np.arange(n_out) * rate
    t_out = t_out[t_out < len(x) - 1]
    i0 = t_out.astype(int)
    frac = t_out - i0
    return x[i0] * (1.0 - frac) + x[i0 + 1] * frac


def synth_preview_rates(sr, path_buzz_fly):
    _, _, _, _, buzz = read_wav(path_buzz_fly)
    rates = [0.8, 1.0, 1.2, 1.5]
    gap = np.zeros(int(0.15 * sr))
    parts = []
    for r in rates:
        parts.append(resample_linear(buzz, r))
        parts.append(gap)
    return np.concatenate(parts[:-1])


# ----------------------------------------------------------------------------
# Post-process + validate
# ----------------------------------------------------------------------------
def fade_edges(x, in_ms=0.0, out_ms=0.0, sr=SR):
    """short linear fades so one-shots never start/stop with a click"""
    x = x.copy()
    ni = int(round(in_ms * 1e-3 * sr))
    no = int(round(out_ms * 1e-3 * sr))
    if ni > 1:
        x[:ni] *= np.linspace(0.0, 1.0, ni)
    if no > 1:
        x[-no:] *= np.linspace(1.0, 0.0, no)
    return x


def post_process(x, peak_db, drive=1.0, fade_in_ms=0.0, fade_out_ms=0.0):
    x = fade_edges(x, fade_in_ms, fade_out_ms)
    x = remove_dc(x)
    x = normalize_peak(x, peak_db)
    x = soft_clip(x, drive=drive)
    x = remove_dc(x)
    x = normalize_peak(x, peak_db)
    return x


def validate(path, expected_dur, is_loop, target_peak_db, peak_tol_db=0.6):
    sr, ch, bits, nf, data = read_wav(path)
    dur = nf / sr
    peak = float(np.max(np.abs(data))) if nf else 0.0
    peak_db = lin_to_db(peak)
    dc = float(np.mean(data))
    seam = float(abs(data[0] - data[-1])) if is_loop and nf else 0.0
    typical = float(np.median(np.abs(np.diff(data)))) + 1e-12 if nf > 2 else 1e-12
    seam_ratio = seam / typical if is_loop else 0.0

    ok = True
    ok &= (sr == SR)
    ok &= (ch == 1)
    ok &= (bits == 16)
    ok &= (abs(dur - expected_dur) < 0.02)
    ok &= (abs(peak_db - target_peak_db) < peak_tol_db)
    ok &= (abs(dc) < 5e-4)
    if is_loop:
        ok &= (seam_ratio < 6.0)

    row = {
        "name": os.path.basename(path),
        "dur": dur,
        "sr": sr,
        "ch": ch,
        "bits": bits,
        "peak_db": peak_db,
        "dc": dc,
        "seam": seam,
        "seam_ratio": seam_ratio if is_loop else float("nan"),
        "loop": is_loop,
        "ok": ok,
    }
    return row


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    np.random.seed(SEED)

    # ---- 1. Hover buzz loop (190 Hz) ----
    hover = synth_loop_wingbeat(
        f0_target=190.0, T=2.0, sr=SR, seed=SEED + 1,
        rolloff=1.20, n_harm=24, air_db=-30.0,
    )
    hover = post_process(hover, peak_db=-3.0)
    write_wav("buzz_hover.wav", hover)

    # ---- 2. Fly buzz loop (215 Hz, brighter) ----
    fly = synth_loop_wingbeat(
        f0_target=215.0, T=2.0, sr=SR, seed=SEED + 2,
        rolloff=1.06, n_harm=26, air_db=-28.5,
    )
    fly = post_process(fly, peak_db=-3.0)
    write_wav("buzz_fly.wav", fly)

    # ---- 3. Dart one-shot ----
    dart = synth_one_shot_wingbeat(
        dart_freq, dart_amp, T=0.6, sr=SR, seed=SEED + 3,
        n_harm=26, rolloff=1.10, air_db=-28.0,
    )
    dart = post_process(dart, peak_db=-3.0, fade_in_ms=4.0, fade_out_ms=12.0)
    write_wav("buzz_dart.wav", dart)

    # ---- 4. Takeoff one-shot ----
    takeoff = synth_one_shot_wingbeat(
        takeoff_freq, takeoff_amp, T=0.5, sr=SR, seed=SEED + 4,
        n_harm=24, rolloff=1.18, air_db=-30.0,
    )
    takeoff = post_process(takeoff, peak_db=-3.0, fade_in_ms=4.0, fade_out_ms=20.0)
    write_wav("takeoff.wav", takeoff)

    # ---- 5. Landing one-shot (+ soft tick at end) ----
    landing = synth_one_shot_wingbeat(
        landing_freq, landing_amp, T=0.6, sr=SR, seed=SEED + 5,
        n_harm=22, rolloff=1.22, air_db=-32.0,
    )
    rng_l = np.random.default_rng(SEED + 55)
    tick_n = int(0.008 * SR)
    tt = np.arange(tick_n) / SR
    tick = np.exp(-tt / 0.0012) * np.sin(2.0 * np.pi * 3200.0 * tt + rng_l.uniform(0.0, 2.0 * np.pi))
    tick_peak = np.max(np.abs(landing)) + 1e-12
    landing[-tick_n:] += 0.12 * tick_peak * tick
    landing = post_process(landing, peak_db=-3.0, fade_in_ms=4.0, fade_out_ms=2.0)
    write_wav("landing.wav", landing)

    # ---- 6. Walk steps loop ----
    walk = synth_walk_steps(SR, SEED + 6)
    walk = post_process(walk, peak_db=-12.0)
    write_wav("walk_steps.wav", walk)

    # ---- 7. Groom loop ----
    groom = synth_groom(SR, SEED + 7)
    groom = post_process(groom, peak_db=-12.0)
    write_wav("groom.wav", groom)

    # ---- 8. Preview of playback-rate scaling ----
    preview = synth_preview_rates(SR, "buzz_fly.wav")
    preview = post_process(preview, peak_db=-3.0)
    write_wav("preview_rates.wav", preview)

    # ---- validation table ----
    checks = [
        ("buzz_hover.wav",  2.0, True,  -3.0),
        ("buzz_fly.wav",    2.0, True,  -3.0),
        ("buzz_dart.wav",   0.6, False, -3.0),
        ("takeoff.wav",     0.5, False, -3.0),
        ("landing.wav",     0.6, False, -3.0),
        ("walk_steps.wav",  1.5, True, -12.0),
        ("groom.wav",       1.2, True, -12.0),
    ]
    rows = [validate(n, d, l, p) for (n, d, l, p) in checks]

    header = f"{'File':<18} {'Dur(s)':>7} {'SR':>6} {'Ch':>3} {'Bit':>4} " \
             f"{'Peak dBFS':>10} {'DC':>9} {'Seam':>10} {'Seam/med':>9} {'Loop':>5} {'OK':>3}"
    print(header)
    print("-" * len(header))
    for r in rows:
        sr_str = "%.1f" % r["seam_ratio"] if r["loop"] else "-"
        print(f"{r['name']:<18} {r['dur']:>7.3f} {r['sr']:>6} {r['ch']:>3} {r['bits']:>4} "
              f"{r['peak_db']:>10.3f} {r['dc']:>9.2e} {r['seam']:>10.2e} "
              f"{sr_str:>9} {str(r['loop']):>5} {str(r['ok']):>3}")

    # Assertions
    for r in rows:
        assert r["sr"] == SR, r
        assert r["ch"] == 1, r
        assert r["bits"] == 16, r
        assert abs(r["dc"]) < 5e-4, r
        assert r["ok"], r
    # preview sanity check
    sr, ch, bits, nf, _ = read_wav("preview_rates.wav")
    assert sr == SR and ch == 1 and bits == 16 and nf > 0
    print("\nAll files validated OK.")


if __name__ == "__main__":
    main()
