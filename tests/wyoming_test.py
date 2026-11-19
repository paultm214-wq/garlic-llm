"""Drives the Wyoming server (tests/wyoming_host.c on a PC, or the console itself) the way Home Assistant's Wyoming
integration does (homeassistant/components/wyoming: data.py, stt.py, tts.py, wake_word.py), with the wyoming library.

Run: python tests/wyoming_test.py PORT [HOST] [--wake okay_nabu.wav] [--no-wake other.wav] [--speech command.wav]
  PORT/HOST    the server (the console: 10300 and its address)
  --wake       16 kHz mono 16-bit WAV with "okay nabu" in it: must be detected
  --no-wake    a WAV without a wake word: must end with not-detected
  --speech     a WAV to transcribe (else one second of a tone); with the console, the transcript is printed
  --mock       the server is tests/wyoming_host.c (its answers are checked exactly)
Needs: pip install wyoming
"""

import asyncio
import io
import math
import struct
import sys
import time
import wave

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.error import Error
from wyoming.info import Describe, Info
from wyoming.ping import Ping, Pong
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeStopped, SynthesizeVoice
from wyoming.wake import Detect, Detection, NotDetected

ARGS = sys.argv[1:]
PORT = int(ARGS[0])
HOST = ARGS[1] if len(ARGS) > 1 and not ARGS[1].startswith("--") else "127.0.0.1"
MOCK = "--mock" in ARGS


def opt(name):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else None


fails = 0


def check(cond, what):
    global fails
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        fails += 1


def wav_pcm(path):
    w = wave.open(path)
    assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, path
    return w.readframes(w.getnframes())


async def describe():
    async with AsyncTcpClient(HOST, PORT) as c:
        await c.write_event(Describe().event())
        while True:
            e = await asyncio.wait_for(c.read_event(), 5)
            if Info.is_type(e.type):
                return Info.from_event(e)


async def stt(pcm, language="en"):
    """As stt.py: Transcribe, AudioStart, chunks, AudioStop; the first Transcript or Error is the answer."""
    async with AsyncTcpClient(HOST, PORT) as c:
        await c.write_event(Transcribe(language=language).event())
        await c.write_event(AudioStart(rate=16000, width=2, channels=1).event())
        for i in range(0, len(pcm), 1024):
            await c.write_event(AudioChunk(rate=16000, width=2, channels=1, audio=pcm[i:i + 1024]).event())
        await c.write_event(AudioStop().event())
        while True:
            e = await asyncio.wait_for(c.read_event(), 120)
            if e is None:
                return None
            if Error.is_type(e.type):
                return "ERROR: " + Error.from_event(e).text
            if Transcript.is_type(e.type):
                return Transcript.from_event(e).text


async def tts(text, voice=None):
    """As tts.py async_get_tts_audio: Synthesize, then chunks until AudioStop. Returns (rate, pcm bytes, seconds to
    the first chunk)."""
    async with AsyncTcpClient(HOST, PORT) as c:
        t0 = time.time()
        await c.write_event(Synthesize(text=text, voice=SynthesizeVoice(name=voice) if voice else None).event())
        rate, pcm, first = None, b"", None
        while True:
            e = await asyncio.wait_for(c.read_event(), 300)
            if e is None or Error.is_type(e.type):
                return None, (Error.from_event(e).text if e else "closed"), None
            if AudioStop.is_type(e.type):
                return rate, pcm, first
            if AudioChunk.is_type(e.type):
                ch = AudioChunk.from_event(e)
                rate = ch.rate
                first = first or time.time() - t0
                pcm += ch.audio


async def tts_stream(parts, voice=None):
    """As tts.py async_stream_tts_audio: SynthesizeStart, chunks, the whole Synthesize, SynthesizeStop; audio until
    SynthesizeStopped (AudioStart gives the WAV header)."""
    async with AsyncTcpClient(HOST, PORT) as c:
        v = SynthesizeVoice(name=voice) if voice else None
        await c.write_event(SynthesizeStart(voice=v).event())
        for p in parts:
            await c.write_event(SynthesizeChunk(text=p).event())
        await c.write_event(Synthesize(text="".join(parts), voice=v).event())
        await c.write_event(SynthesizeStop().event())
        started, pcm = False, b""
        while True:
            e = await asyncio.wait_for(c.read_event(), 300)
            if e is None:
                return None
            if Error.is_type(e.type):
                return "ERROR"
            if started and AudioChunk.is_type(e.type):
                pcm += AudioChunk.from_event(e).audio
            elif not started and AudioStart.is_type(e.type):
                started = True
            elif SynthesizeStopped.is_type(e.type):
                return pcm


async def wake(pcm, word="okay_nabu", stop=True):
    """As wake_word.py: Detect, AudioStart, chunks (with timestamps) while reading events; a Detection ends it."""
    async with AsyncTcpClient(HOST, PORT) as c:
        await c.write_event(Detect(names=[word] if word else None).event())
        await c.write_event(AudioStart(rate=16000, width=2, channels=1).event())
        result = None

        async def reader():
            nonlocal result
            while True:
                e = await c.read_event()
                if e is None:
                    return
                if Detection.is_type(e.type):
                    result = Detection.from_event(e)
                    return
                if NotDetected.is_type(e.type):
                    result = "not-detected"
                    return
                if Error.is_type(e.type):
                    result = "ERROR: " + Error.from_event(e).text
                    return

        task = asyncio.create_task(reader())
        ts = 0
        for i in range(0, len(pcm), 2048):   # 64 ms chunks, in real time / 4
            if task.done():
                break
            await c.write_event(AudioChunk(rate=16000, width=2, channels=1, audio=pcm[i:i + 2048], timestamp=ts).event())
            ts += 64
            await asyncio.sleep(0.016)
        if stop and not task.done():
            await c.write_event(AudioStop().event())
        await asyncio.wait_for(task, 30)
        return result


def tone(seconds, rate=16000):
    return b"".join(struct.pack("<h", int(6000 * math.sin(i * 2 * math.pi * 300 / rate))) for i in range(int(seconds * rate)))


async def main():
    info = await describe()
    asr_langs = sorted({l for a in info.asr for m in a.models for l in m.languages})
    voices = [v.name for t in info.tts for v in t.voices]
    words = [m.name for w in info.wake for m in w.models]
    print("info:", asr_langs, voices, words)
    check(any(a.installed for a in info.asr) and "en" in asr_langs and "es" in asr_langs, "speech to text offered")
    check(any(t.installed for t in info.tts) and "garlic-en" in voices, "text to speech offered")
    check(info.tts and info.tts[0].supports_synthesize_streaming, "streaming text to speech")
    check("okay_nabu" in words, "wake words offered")

    async with AsyncTcpClient(HOST, PORT) as c:
        await c.write_event(Ping(text="hi").event())
        e = await asyncio.wait_for(c.read_event(), 5)
        check(Pong.is_type(e.type) and Pong.from_event(e).text == "hi", "ping")

    speech = wav_pcm(opt("--speech")) if opt("--speech") else tone(1.0)
    t0 = time.time()
    text = await stt(speech)
    print(f"transcript ({time.time() - t0:.2f} s): {text!r}")
    if MOCK:
        check(text == f"heard {len(speech) / 32000:.2f} s", "speech to text")
        long = await stt(tone(12.5))
        check(long == "heard 10.00 s", "speech to text cut at the model's limit")
    else:
        check(text is not None and not text.startswith("ERROR"), "speech to text")

    rate, pcm, first = await tts("Hello! The kitchen lights are on now.", "garlic-en")
    check(rate is not None and len(pcm) > 0, f"text to speech ({rate} Hz, {len(pcm) / 2 / (rate or 1):.2f} s, "
                                             f"first audio after {first or 0:.2f} s)")
    if not MOCK and pcm:
        with wave.open("wyoming_tts.wav", "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
            w.writeframes(pcm)
        print("     (saved as wyoming_tts.wav)")
    pcm2 = await tts_stream(["Hola, ", "las luces ", "están encendidas."], "garlic-es")
    check(isinstance(pcm2, bytes) and len(pcm2) > 0, "streamed text to speech")
    if MOCK:
        r, err, _ = await tts("fail")
        check(r is None and err == "mock failure", "text to speech error reaches the client")

    if opt("--wake"):
        t0 = time.time()
        r = await wake(wav_pcm(opt("--wake")))
        check(isinstance(r, Detection) and r.name == "okay_nabu", f"wake word detected: {r} ({time.time() - t0:.1f} s)")
        r = await wake(wav_pcm(opt("--wake")), word="ok_nabu")
        check(isinstance(r, Detection) and r.name == "ok_nabu", "an older wake word name is answered with that name")
    if opt("--no-wake"):
        r = await wake(wav_pcm(opt("--no-wake")))
        check(r == "not-detected", f"no wake word: {r}")
    r = await wake(tone(0.5), word=None)
    check(r == "not-detected", "default wake word, nothing heard")

    # several clients at once (satellites listening while another one speaks)
    t0 = time.time()
    res = await asyncio.gather(wake(tone(2.0)), wake(tone(2.0)), stt(tone(0.5)), describe())
    check(res[0] == "not-detected" and res[1] == "not-detected" and res[2] and isinstance(res[3], Info),
          f"four clients at once ({time.time() - t0:.1f} s)")

    # garbage closes the connection, and the server keeps serving
    r, w = await asyncio.open_connection(HOST, PORT)
    w.write(b"this is not json\n")
    await w.drain()
    check(await asyncio.wait_for(r.read(), 5) == b"", "malformed event closes the connection")
    w.close()
    check(isinstance(await describe(), Info), "server still answers")
    print("FAILED" if fails else "all passed", f"({fails} failures)")
    return fails


if __name__ == "__main__":
    sys.exit(1 if asyncio.run(main()) else 0)
