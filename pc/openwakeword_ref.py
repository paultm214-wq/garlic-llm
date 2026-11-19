"""Reference for chat/oww.c: every score of every wake word for a 16 kHz mono 16-bit WAV, computed by pyopen_wakeword
(TFLite, the library wyoming-openwakeword uses), fed in chunks as tests/oww_test.c feeds the C version.

Run: python pc/openwakeword_ref.py speech.wav [chunk] > ref.txt
     build/oww_test speech.wav --scores --chunk 1024 > c.txt     (same lines: word index, score; within ~1e-4)
Needs: pip install pyopen-wakeword numpy
"""
import sys
import wave

from pyopen_wakeword import Model, OpenWakeWord, OpenWakeWordFeatures

WORDS = ["okay_nabu", "hey_jarvis", "alexa", "hey_mycroft", "hey_rhasspy"]   # the order of oww.bin


def main():
    path, chunk = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    pcm = wave.open(path).readframes(10**9)
    features = OpenWakeWordFeatures.from_builtin()
    models = [OpenWakeWord.from_builtin(Model(w)) for w in WORDS]
    for i in range(0, len(pcm), chunk * 2):
        for emb in features.process_streaming(pcm[i:i + chunk * 2]):
            for k, m in enumerate(models):
                for p in m.process_streaming(emb):
                    print(k, "%.6f" % p)


if __name__ == "__main__":
    main()
