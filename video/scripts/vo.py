"""Voiceover: Kokoro TTS (voice am_onyx), then pitched down and roughened with ffmpeg.

Needs: pip install kokoro-onnx soundfile, ffmpeg, and the model files in this folder:
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
Writes vo/p_v1.wav … vo/p_v10.wav, which score.py turns into public/audio/v*.wav.
"""
import os, subprocess, soundfile as sf
from kokoro_onnx import Kokoro

LINES = {
    "v1": "Every map app has a blind spot.",
    "v2": "Tunnels. Underpasses. City canyons.",
    "v3": "The moment GPS drops, your map freezes.",
    "v4": "And phone sensors alone drift hundreds of meters.",
    "v5": "I.D.R. reads your phone's motion sensors.",
    "v6": "An on-device A.I. turns them into speed and heading, ten times a second,",
    "v7": "and keeps you on the road. No GPS. No internet. No extra hardware.",
    "v8": "Six hundred forty-eight meters with no signal.",
    "v9": "I.D.R. comes out within fourteen meters.",
    "v10": "I.D.R. Lose the signal. Not the road.",
}
# trim silence, drop pitch ~8% at the same speed, warm low end, a little grit, compress
CHAIN = ("silenceremove=start_periods=1:start_threshold=-45dB,areverse,silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
         "asetrate=24000*0.92,aresample=48000,atempo=1.087,highpass=f=55,equalizer=f=110:t=q:w=1:g=5,"
         "equalizer=f=2800:t=q:w=1.2:g=2.5,equalizer=f=7000:t=h:w=1:g=-2,volume=4dB,asoftclip=type=tanh,"
         "acompressor=threshold=-20dB:ratio=4:attack=5:release=80:makeup=4dB")
os.makedirs("vo", exist_ok=True)
k = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
for name, text in LINES.items():
    s, sr = k.create(text, voice="am_onyx", speed=0.95, lang="en-us")
    sf.write(f"vo/{name}.wav", s, sr)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", f"vo/{name}.wav", "-af", CHAIN, "-ar", "48000", "-ac", "1", f"vo/p_{name}.wav"], check=True)
    print(name)
