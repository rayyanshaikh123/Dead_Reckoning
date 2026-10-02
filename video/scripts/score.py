import sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "../public/audio"
VO_DIR = sys.argv[2] if len(sys.argv) > 2 else "vo"
"""40 s IDR story cut: intro / problem / how it works / use case / outro (120 BPM)."""
"""Procedural score + sound design + VO mix for the 30 s IDR showcase (120 BPM)."""
import numpy as np, soundfile as sf
from scipy.signal import fftconvolve, butter, sosfilt

SR = 48000
DUR = 40.0
N = int(SR * DUR)
rng = np.random.default_rng(3)
BEAT = 0.5

def buf(): return np.zeros((N, 2))
def tt(d): return np.arange(int(d * SR)) / SR
def place(dst, sig, t0, gain=1.0, pan=0.0):
    i0 = int(t0 * SR)
    if sig.ndim == 1:
        l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        sig = np.stack([sig * l * 1.414, sig * r * 1.414], 1)
    if i0 < 0: sig = sig[-i0:]; i0 = 0
    n = min(len(sig), N - i0)
    if n > 0: dst[i0:i0 + n] += sig[:n] * gain
def filt(x, kind, f, order=2):
    sos = butter(order, f, btype=kind, fs=SR, output='sos'); return sosfilt(sos, x, axis=0)
def noise(d): return rng.standard_normal(int(d * SR))
def saw(f, t, ph=0.0): return 2 * ((f * t + ph) % 1.0) - 1
def env_ad(t, a, d): return np.minimum(t / max(a, 1e-4), 1.0) * np.exp(-np.maximum(t - a, 0) / d)

# ---- reverb impulse ----
def make_ir(dur=2.2, decay=0.55):
    t = tt(dur); e = np.exp(-t / decay)
    ir = np.stack([rng.standard_normal(len(t)) * e, rng.standard_normal(len(t)) * e], 1)
    ir = filt(ir, 'low', 6000); ir[:int(0.012 * SR)] *= np.linspace(0, 1, int(0.012 * SR))[:, None]
    return ir / np.sqrt((ir ** 2).sum(0))
IR = make_ir()
def reverb(x, mix=1.0):
    out = np.stack([fftconvolve(x[:, 0], IR[:, 0])[:N], fftconvolve(x[:, 1], IR[:, 1])[:N]], 1)
    return out * mix

# ---------------- instruments ----------------
def kick(amp=1.0, dec=0.32, f0=160, f1=42):
    t = tt(0.6); f = f1 + (f0 - f1) * np.exp(-t / 0.035)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * np.exp(-t / dec) + 0.3 * np.exp(-t / 0.004) * noise(0.6)[:len(t)] * 0.3
    return np.tanh(s * 1.6) * amp
def clap(amp=1.0):
    t = tt(0.35); n = noise(0.35)
    e = np.exp(-t / 0.09); e[:int(0.03 * SR)] *= (1 + 0.6 * np.sin(2 * np.pi * 90 * t[:int(0.03 * SR)]))
    s = filt(n, 'band', [900, 5000]) * e + 0.25 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05)
    return s * amp
def hat(amp=1.0, dec=0.035):
    t = tt(0.15); return filt(noise(0.15), 'high', 7000) * np.exp(-t / dec) * amp
def snare(amp=1.0):
    t = tt(0.25); return (filt(noise(0.25), 'band', [1200, 7000]) * np.exp(-t / 0.07) + 0.4 * np.sin(2 * np.pi * 200 * t) * np.exp(-t / 0.04)) * amp
def bass_note(f, d, amp=1.0, cutoff=500):
    t = tt(d)
    s = saw(f, t) + saw(f * 1.006, t, 0.3) + 0.6 * np.sin(2 * np.pi * f / 2 * t)
    s = filt(s, 'low', cutoff, 2) * env_ad(t, 0.004, d * 0.6)
    return np.tanh(s * 1.2) * amp
def pad(freqs, d, amp=1.0, cutoff=1400, att=0.6, rel=0.8):
    t = tt(d + rel); out = np.zeros((len(t), 2))
    for i, f in enumerate(freqs):
        for det, side in ((0.996, 0), (1.004, 1), (1.0, 0), (1.0, 1)):
            out[:, side] += saw(f * det, t, rng.random()) * 0.25
    out = filt(out, 'low', cutoff, 2)
    e = np.minimum(t / att, 1.0) * np.where(t > d, np.exp(-(t - d) / (rel / 3)), 1.0)
    return out * e[:, None] * amp / len(freqs)
def pluck(f, amp=1.0, d=0.45):
    t = tt(d); s = (saw(f, t) * 0.5 + np.sin(2 * np.pi * f * t)) * np.exp(-t / 0.12)
    return filt(s, 'low', 3500) * amp
def whoosh(dur_in=0.35, dur_out=0.3, amp=1.0, lo=200, hi=4000, deep=True):
    d = dur_in + dur_out; t = tt(d); n = noise(d)
    e = np.where(t < dur_in, (t / dur_in) ** 2.5, np.exp(-(t - dur_in) / (dur_out / 3)))
    # swept band via chunks
    out = np.zeros(len(t)); chunk = 1024
    for i in range(0, len(t), chunk):
        k = min(1.0, t[i] / dur_in) if t[i] < dur_in else max(0.0, 1 - (t[i] - dur_in) / dur_out)
        fc = lo * (hi / lo) ** k
        seg = n[max(0, i - 2048):i + chunk]
        y = filt(seg, 'band', [fc * 0.6, min(fc * 1.6, 20000)])
        out[i:i + chunk] = y[-len(n[i:i + chunk]):]
    s = out * e
    if deep:
        f = 30 + 90 * np.where(t < dur_in, t / dur_in, np.exp(-(t - dur_in) / 0.1))
        s += 0.7 * np.sin(2 * np.pi * np.cumsum(f) / SR) * e
    return s * amp
def metal_click(amp=1.0, base=2100):
    t = tt(0.12); s = np.zeros(len(t))
    for m, d in ((1.0, 0.03), (1.62, 0.025), (2.47, 0.02), (3.9, 0.012)):
        s += np.sin(2 * np.pi * base * m * t) * np.exp(-t / d)
    s += filt(noise(0.12), 'high', 3000) * np.exp(-t / 0.004) * 1.5
    return s * amp * 0.4
def snap(amp=1.0):
    t = tt(0.3)
    thump = np.sin(2 * np.pi * (60 + 120 * np.exp(-t / 0.02)) * t) * np.exp(-t / 0.06)
    out = thump * 0.9; out[:len(metal_click())] += metal_click(1.2, 1700)
    return out * amp
def boom(amp=1.0, d=2.5, f0=70, f1=28):
    t = tt(d); f = f1 + (f0 - f1) * np.exp(-t / 0.25)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.9)
    s += filt(noise(d), 'low', 300) * np.exp(-t / 0.15) * 0.6
    return np.tanh(s * 1.5) * amp
def riser(t0, t1, amp=1.0):
    d = t1 - t0; t = tt(d); k = t / d
    n = noise(d); out = np.zeros(len(t)); chunk = 2048
    for i in range(0, len(t), chunk):
        fc = 300 * (9000 / 300) ** k[i]
        seg = n[max(0, i - 4096):i + chunk]; y = filt(seg, 'band', [fc * 0.7, min(fc * 1.4, 20000)])
        out[i:i + chunk] = y[-len(n[i:i + chunk]):]
    f = 110 * 2 ** (2.0 * k)  # pitch rises two octaves
    s = out * k ** 2 * 0.8 + 0.18 * (saw(1, t) * 0 + np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.5 * saw(1, np.cumsum(f * 1.01) / SR)) * k ** 1.5
    return s * amp
def blip(amp=1.0):
    t = tt(0.25); return (np.sin(2 * np.pi * 1320 * t) * np.exp(-t / 0.05) + 0.5 * np.sin(2 * np.pi * 1980 * t) * np.exp(-t / 0.03)) * amp



drums, bass, music, sfx, send = buf(), buf(), buf(), buf(), buf()
CH = {'Dm': [146.83, 174.61, 220.0, 293.66], 'Bb': [116.54, 146.83, 174.61, 233.08], 'F': [174.61, 220.0, 261.63, 349.23], 'C': [130.81, 164.81, 196.0, 261.63],
      'Gm': [98.0, 146.83, 196.0, 233.08], 'A': [110.0, 138.59, 164.81, 220.0]}
ROOT = {'Dm': 73.42, 'Bb': 58.27, 'F': 87.31, 'C': 65.41, 'Gm': 49.0, 'A': 55.0}

# ---------- INTRO 0-5 ----------
t = tt(5.3); drone = (np.sin(2 * np.pi * 36.71 * t) + 0.5 * np.sin(2 * np.pi * 73.42 * t)) * np.minimum(t / 2.5, 1) * np.clip((5.3 - t) / 0.4, 0, 1) * 0.35
place(music, drone, 0.0)
place(music, pad(CH['Dm'], 5.0, 0.5, cutoff=700, att=2.5, rel=0.3), 0.0)
for i, bt in enumerate([3.0, 3.5, 4.0, 4.5]): place(drums, kick(0.3 + 0.1 * i, dec=0.25), bt)
place(sfx, whoosh(1.4, 1.2, 0.5, lo=300, hi=6000), 0.5); place(send, whoosh(1.4, 1.2, 0.3, lo=300, hi=6000, deep=False), 0.5)
place(sfx, blip(0.18), 3.22); place(send, blip(0.2), 3.22)

# ---------- PROBLEM 5-14 : tense, half-time, GPS dies at 8.0 ----------
for b0, ch in [(5.0, 'Dm'), (7.0, 'Bb')]:
    place(music, pad(CH[ch], 2.0 if b0 < 7 else 1.0, 0.45, cutoff=900, att=0.2, rel=0.3), b0)
for bt in [5.0, 6.0, 7.0, 7.5]: place(drums, kick(0.9), bt)
for k in range(12): place(drums, hat(0.16, 0.025), 5.0 + k * 0.25 + 0.125, pan=0.3)
for k in range(12): place(bass, bass_note(ROOT['Dm'] if k < 8 else ROOT['Bb'], 0.22, 0.45, cutoff=450), 5.0 + k * 0.25)
# satellite pings
for ts in [5.35, 6.35, 7.35]:
    tp = tt(0.6); ping = np.sin(2 * np.pi * 1760 * tp) * np.exp(-tp / 0.18) * 0.12
    place(sfx, ping, ts, pan=0.4); place(send, ping * 1.5, ts)
# GPS lost glitch at 8.0: stutter + pitch dive, then everything drops to a cold drone
g = np.zeros(int(0.7 * SR))
for i in range(7):
    tp = tt(0.05); f = 1400 * 0.82 ** i
    seg = np.sign(np.sin(2 * np.pi * f * tp)) * np.exp(-tp / 0.03) * 0.25
    i0 = int(i * 0.07 * SR); g[i0:i0 + len(seg)] += seg[:len(g) - i0]
place(sfx, filt(g, 'low', 6000), 7.98, pan=-0.2); place(send, g * 0.5, 7.98)
tp = tt(1.2); dive = np.sin(2 * np.pi * np.cumsum(220 * np.exp(-tp * 2.5) + 40) / SR) * np.exp(-tp / 0.5) * 0.35
place(sfx, dive, 8.0)
place(drums, kick(1.0, dec=0.5, f0=140, f1=36), 8.0)
t = tt(2.6); cold = (np.sin(2 * np.pi * 36.71 * t) * 0.3 + pad(CH['Dm'], 2.4, 1.0, cutoff=500, att=0.6, rel=0.3)[:len(t), 0] * 0.4) * np.minimum(t / 0.5, 1)
place(music, cold, 8.0)
for bt in [8.5, 9.5]:  # heartbeat
    place(drums, kick(0.45, dec=0.2, f0=90, f1=40), bt); place(drums, kick(0.3, dec=0.2, f0=90, f1=40), bt + 0.22)
# 10.5-13.75: drift -> tension riser, accelerating ticks (the error counter)
place(sfx, riser(10.4, 13.75, 0.5), 10.4); place(send, riser(10.4, 13.75, 0.2), 10.4)
place(music, pad(CH['Gm'], 1.6, 0.45, cutoff=1200, att=0.3, rel=0.2), 10.5)
place(music, pad(CH['A'], 1.6, 0.5, cutoff=1600, att=0.3, rel=0.2), 12.1)
tk = 10.65
while tk < 13.6:
    k = (tk - 10.6) / 3.0
    place(sfx, metal_click(0.12 + 0.2 * k, 3200), tk, pan=0.3)
    tk += 0.25 * (1 - 0.8 * k) + 0.03
for k in range(13): place(bass, bass_note(ROOT['A'] * 2 ** (k / 13 * 0.25), 0.24, 0.4 + 0.02 * k, cutoff=500 + 60 * k), 10.5 + k * 0.25)

# ---------- HOW IT WORKS 14-21.3 + SNAP 21.3-25.6 : full groove ----------
prog = ['Dm', 'Bb', 'F', 'C']
for bar in range(6):  # 14.0 .. 26.0
    b0 = 14.0 + bar * 2.0; ch = prog[bar % 4]
    if b0 >= 25.6: break
    place(music, pad(CH[ch], 2.0, 0.55, cutoff=1800, att=0.25, rel=0.5), b0)
    for k in range(4):
        if b0 + k * BEAT < 25.6:
            place(drums, kick(1.0), b0 + k * BEAT)
            if k in (1, 3): place(drums, clap(0.5), b0 + k * BEAT); place(send, clap(0.22), b0 + k * BEAT)
    for k in range(16):
        tt0 = b0 + k * BEAT / 4
        if tt0 < 25.6: place(drums, hat(0.2 if k % 2 else 0.1), tt0, pan=0.3)
    for k in range(8):
        tt0 = b0 + k * BEAT / 2
        if tt0 < 25.6: place(bass, bass_note(ROOT[ch] * (2 if k % 4 == 3 else 1), 0.24, 0.5, cutoff=850), tt0)
# data blips during THINK
for i, tb in enumerate(np.arange(17.25, 21.2, 0.125)):
    if i % 3 != 2: place(sfx, blip(0.035) * 1.0, tb, pan=-0.5 + (i % 5) * 0.25)
place(sfx, metal_click(0.5, 2600), 14.08)
for tb in [23.0, 23.85, 24.7]: place(sfx, snap(0.35), tb, pan=0.5); place(send, snap(0.15), tb)

# ---------- USE CASE 25.6-31 : build to the exit ----------
place(music, pad(CH['Gm'], 2.7, 0.5, cutoff=1500, att=0.3, rel=0.2), 25.6)
place(music, pad(CH['A'], 2.7, 0.55, cutoff=2200, att=0.3, rel=0.1), 28.3)
for k in range(10): place(drums, kick(0.95), 25.6 + k * BEAT) if 25.6 + k * BEAT < 30.3 else None
for k in range(21): place(bass, bass_note(ROOT['Gm'] if k < 11 else ROOT['A'], 0.24, 0.5, cutoff=700 + 40 * k), 25.6 + k * 0.25)
tr = 29.0
while tr < 30.85:
    pk = (tr - 29.0) / 1.85; step = 0.25 if pk < 0.4 else (0.125 if pk < 0.75 else 0.0625)
    place(drums, snare(0.25 + 0.5 * pk), tr, pan=0.1); place(send, snare(0.12 * pk), tr); tr += step
place(sfx, riser(27.6, 30.9, 0.5), 27.6); place(send, riser(27.6, 30.9, 0.2), 27.6)
# exit impact 31.0
place(drums, kick(1.25, dec=0.6, f0=200, f1=38), 31.0)
place(sfx, boom(0.55, d=1.4, f0=90, f1=34), 31.0)
place(send, filt(noise(1.2), 'band', [400, 9000]) * np.exp(-tt(1.2) / 0.25) * 0.5, 31.0)
# ---------- RESULT 31-35 : lighter groove ----------
for bar in range(2):
    b0 = 31.0 + bar * 2.0; ch = ['F', 'C'][bar]
    place(music, pad(CH[ch], 2.0, 0.55, cutoff=2000, att=0.05 if bar == 0 else 0.3, rel=0.5), b0)
    for k in range(8): place(bass, bass_note(ROOT[ch], 0.22, 0.4, cutoff=700), b0 + k * BEAT / 2)
    for k in range(8): place(drums, hat(0.14 if k % 2 else 0.07), b0 + k * BEAT / 2, pan=0.3)
    for k in range(4):
        if b0 + k * BEAT > 31.2: place(drums, kick(0.7), b0 + k * BEAT)
# ---------- OUTRO 35-40 ----------
t = tt(0.62); rev = filt(noise(0.62), 'high', 2500) * (t / 0.62) ** 3 * 0.45
place(sfx, rev, 35.62 - 0.62)
place(sfx, boom(1.25, d=3.5, f0=65, f1=26), 35.62)
place(drums, kick(1.0, dec=0.8, f0=120, f1=32), 35.62)
place(music, pad(CH['Bb'], 1.0, 0.5, cutoff=1500, att=0.2), 35.62)
place(music, pad([87.31, 174.61, 220.0, 261.63, 392.0], 3.3, 0.6, cutoff=1700, att=0.5, rel=1.0), 36.55)
arp = [349.23, 440.0, 523.25, 783.99, 523.25, 440.0]
k = 0; ta = 36.55
while ta < 39.4:
    place(music, pluck(arp[k % len(arp)], 0.16 * (1 - (ta - 36.55) / 4)), ta, pan=0.4 * np.sin(k)); place(send, pluck(arp[k % len(arp)], 0.12), ta)
    k += 1; ta += 0.25
place(drums, kick(0.5, dec=0.4), 36.55)
# whooshes on cuts
for c, a in [(5.0, 0.75), (14.0, 0.75), (17.0, 0.6), (21.3, 0.65), (25.6, 0.5), (31.0, 0.6), (35.0, 0.75)]:
    w = whoosh(0.32, 0.28, a); place(sfx, w, c - 0.32); place(send, w * 0.4, c - 0.32)

# ---------- VO ----------
vo = buf()
VO = [('v1', 1.6), ('v2', 5.3), ('v3', 7.7), ('v4', 10.5), ('v5', 14.2), ('v6', 17.1), ('v7', 21.4), ('v8', 26.0), ('v9', 31.45), ('v10', 36.45)]
duck = np.zeros(N)
for name, t0 in VO:
    x, sr = sf.read(f'{VO_DIR}/p_{name}.wav'); x = x.mean(1) if x.ndim > 1 else x
    x = x / (np.abs(x).max() + 1e-9) * 0.8
    place(vo, x, t0)
    i0, i1 = int((t0 - 0.1) * SR), min(N, int((t0 + len(x) / SR + 0.15) * SR)); duck[i0:i1] = 1
k = int(0.08 * SR); duck = np.convolve(duck, np.ones(k) / k, 'same')
vo_wet = reverb(vo, 0.12)

# ---------- mix ----------
kick_env = np.zeros(N)
for b in list(np.arange(14.0, 25.6, BEAT)) + list(np.arange(25.6, 30.3, BEAT)):
    i0 = int(b * SR); L = int(0.22 * SR); kick_env[i0:i0 + L] = np.maximum(kick_env[i0:i0 + L], np.linspace(1, 0, L) ** 2)
bass *= (1 - 0.6 * kick_env)[:, None]; music *= (1 - 0.35 * kick_env)[:, None]
bass = filt(bass, 'low', 2500)
music_bus = drums * 0.9 + bass * 0.9 + music * 0.75 + reverb(music * 0.3 + send, 0.6)
music_bus *= (1 - 0.62 * duck)[:, None]

# ---------- export stems for Remotion ----------
# bed = music + sound design (undocked); Remotion ducks it under each VO line.
bed = music_bus / (1 - 0.62 * duck)[:, None] + sfx * 0.9
full = bed * (1 - 0.62 * duck)[:, None] + vo * 2.0 + vo_wet * 1.6
G = 0.89 / np.abs(full).max()
t = np.arange(N) / SR
bed *= np.clip(t / 0.05, 0, 1)[:, None] * np.clip((40.0 - t) / 0.8, 0, 1)[:, None]
sf.write(OUT + '/score.wav', (bed * G).astype(np.float32), SR, subtype='PCM_16')
for name, t0 in VO:
    x, _ = sf.read(f'{VO_DIR}/p_{name}.wav'); x = x.mean(1) if x.ndim > 1 else x
    x = x / (np.abs(x).max() + 1e-9) * 0.8 * 2.0 * G
    sf.write(f'{OUT}/{name}.wav', x.astype(np.float32), SR, subtype='PCM_16')
print('gain', G)
