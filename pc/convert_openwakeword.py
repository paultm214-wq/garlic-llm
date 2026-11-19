"""openWakeWord's models (TFLite) -> chat/assets/oww.bin, which chat/oww.c runs on the console's CPU.

openWakeWord (github.com/dscripka/openWakeWord) finds a wake word in three stages, all of them small:
  melspectrogram   1760 samples (16 kHz) -> 8 frames of 32 log-mel bands: a 512-point DFT of 400 samples with a
                   periodic Hann window, hop 160 (the model's two convolutions are exactly that, so oww.c uses an FFT
                   and only the 257 x 32 filter bank is stored), power, the filter bank, 10 log10 (floor 1e-10), and
                   at most 80 dB under the loudest value of the call; then x / 10 + 2 (pyopen_wakeword)
  embedding        76 frames x 32 bands -> 96 features (Google's speech embedding: 20 convolutions with LeakyReLU)
  one classifier per wake word: the last 16 embeddings -> a probability (fully connected layers, layer norms; hey
                   jarvis has a second network that rechecks every score of 0.5 or more)

Input: the .tflite files of pyopen-wakeword (the package wyoming-openwakeword uses; they are in its wheel, folder
pyopen_wakeword/models) or of openWakeWord's releases, in models/openwakeword/ (or the folder given). Needs numpy and
the `tflite` package (pip install numpy tflite).

oww.bin (little-endian):
  "OWW1", u32 version 1, u32 n_fft 512, u32 window 400, u32 hop 160, u32 mels 32,
  f32 amin, f32 top_db, f32 db scale (10 log10(e)), f32 [257][32] filter bank,
  program of the embedding (input 76 x 32 x 1),
  u32 number of wake words, then per word: char[32] name, char[32] phrase, u32 windows (16), u32 features (96),
  program of its classifier.
  A program: u32 number of ops, then per op: u32 type, u32 n_params, u32 n_floats, u32 params[], f32 floats[]:
    1 conv     kh kw cin cout sh sw pad_top pad_bottom pad_left pad_right | W[kh][kw][cin][cout], b[cout]
    2 leaky    | alpha
    3 max      | c (x = max(x, c))
    4 maxpool  fh fw sh sw pad_top pad_bottom pad_left pad_right
    5 dense    in out act (0 none, 1 relu) | W[in][out], b[out]
    6 norm     n act | eps, gamma[n], beta[n] (layer norm over the vector)
    7 sigmoid
    8 gate     | threshold: a score under it is the result; else the rest of the program runs on the input again

Run: python pc/convert_openwakeword.py [folder with the .tflite files]
"""

import struct
import sys
from pathlib import Path

import numpy as np
import tflite
from tflite.BuiltinOperator import BuiltinOperator
from tflite.BuiltinOptions import BuiltinOptions
from tflite.Conv2DOptions import Conv2DOptions
from tflite.FullyConnectedOptions import FullyConnectedOptions
from tflite.IfOptions import IfOptions
from tflite.LeakyReluOptions import LeakyReluOptions
from tflite.Pool2DOptions import Pool2DOptions
from tflite.AddOptions import AddOptions

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models" / "openwakeword"
OUT = ROOT / "chat" / "assets" / "oww.bin"
WORDS = ["okay_nabu", "hey_jarvis", "alexa", "hey_mycroft", "hey_rhasspy"]   # the order the server lists them in
OPNAME = {v: k for k, v in BuiltinOperator.__dict__.items() if not k.startswith("_")}
CONV, LEAKY, MAXC, MAXPOOL, DENSE, NORM, SIGMOID, GATE = range(1, 9)


class Graph:
    def __init__(self, path):
        self.buf = Path(path).read_bytes()
        self.m = tflite.Model.GetRootAsModel(self.buf, 0)

    def sub(self, i):
        return self.m.Subgraphs(i)

    def ops(self, g):
        out = []
        for k in range(g.OperatorsLength()):
            op = g.Operators(k)
            code = self.m.OperatorCodes(op.OpcodeIndex())
            out.append((OPNAME[max(code.BuiltinCode(), code.DeprecatedBuiltinCode())], op))
        return out

    def const(self, g, i):
        """The constant tensor i of subgraph g as float32 (None if it is not a constant)."""
        if i < 0:
            return None
        t = g.Tensors(i)
        b = self.m.Buffers(t.Buffer())
        if b.DataLength() == 0:
            return None
        dt = {0: np.float32, 2: np.int32}[t.Type()]
        a = np.frombuffer(b.DataAsNumpy().tobytes(), dt)
        return a.reshape(t.ShapeAsNumpy()) if t.ShapeLength() else a.reshape(())

    @staticmethod
    def shape(g, i):
        t = g.Tensors(i)
        return [int(x) for x in t.ShapeAsNumpy()] if t.ShapeLength() else []


def options(op, cls):
    o = cls()
    o.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
    return o


def same_pad(n, k, s):
    out = -(-n // s)
    total = max((out - 1) * s + k - n, 0)
    return total // 2, total - total // 2


def record(kind, params=(), floats=()):
    floats = np.concatenate([np.ravel(np.asarray(f, np.float32)) for f in floats]) if floats else np.zeros(0, np.float32)
    return struct.pack("<3I", kind, len(params), floats.size) + struct.pack(f"<{len(params)}I", *params) + floats.tobytes()


def program(recs):
    return struct.pack("<I", len(recs)) + b"".join(recs)


def mel_part(G):
    g = G.sub(0)
    ops = G.ops(g)
    names = [n for n, _ in ops]
    assert names.count("CONV_2D") == 2 and "BATCH_MATMUL" in names and "LOG" in names, names
    convs = [op for n, op in ops if n == "CONV_2D"]
    kern = [G.const(g, op.InputsAsNumpy()[1])[:, 0, :, 0] for op in convs]
    for op in convs:
        o = options(op, Conv2DOptions)
        assert (o.StrideH(), o.StrideW(), o.Padding()) == (1, 160, 1)
        assert not np.any(G.const(g, op.InputsAsNumpy()[2]))
    # the kernels are a 512-point DFT of a periodic Hann window of 400 samples (centred): checked, then not stored
    n, k = np.arange(512), np.arange(257)[:, None]
    win = np.zeros(512)
    win[56:456] = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)
    cos, sin = win * np.cos(2 * np.pi * k * n / 512), win * np.sin(2 * np.pi * k * n / 512)
    errs = sorted([np.abs(kern[0] - cos).max(), np.abs(kern[1] - cos).max()])
    assert errs[0] < 1e-6 and min(np.abs(np.abs(kern[0]) - np.abs(sin)).max(), np.abs(np.abs(kern[1]) - np.abs(sin)).max()) < 1e-6
    mm = next(op for n_, op in ops if n_ == "BATCH_MATMUL")
    fb = G.const(g, mm.InputsAsNumpy()[1])
    assert fb.shape == (257, 32)
    scalars = {}
    for n_, op in ops:
        for i in op.InputsAsNumpy():
            c = G.const(g, i)
            if c is not None and c.size == 1 and c.dtype == np.float32:
                scalars.setdefault(n_, []).append(float(c))
    amin = min(v for v in scalars["MAXIMUM"] if v > 0)
    top_db = scalars["SUB"][0]
    scale = float(np.prod(scalars["MUL"]))   # 10 * log10(e): the log is natural
    assert abs(amin - 1e-10) < 1e-15 and top_db == 80.0 and abs(scale - 10 / np.log(10)) < 1e-5, (amin, top_db, scale)
    return struct.pack("<4I3f", 512, 400, 160, 32, amin, top_db, scale) + fb.astype(np.float32).tobytes()


def embedding_program(G):
    g = G.sub(0)
    recs, pad = [], None
    for name, op in G.ops(g):
        ins, outs = op.InputsAsNumpy(), op.OutputsAsNumpy()
        if name == "PAD":
            p = G.const(g, ins[1])   # [[0,0],[top,bottom],[left,right],[0,0]]
            pad = (int(p[1][0]), int(p[1][1]), int(p[2][0]), int(p[2][1]))
        elif name == "CONV_2D":
            o = options(op, Conv2DOptions)
            assert o.FusedActivationFunction() == 0 and o.DilationHFactor() == 1 and o.DilationWFactor() == 1
            w = G.const(g, ins[1])                    # OHWI
            b = G.const(g, ins[2])
            cout, kh, kw, cin = w.shape
            ish = G.shape(g, ins[0])                  # [1, h, w, c] (before our PAD folding)
            sh, sw = o.StrideH(), o.StrideW()
            if o.Padding() == 0:                      # SAME
                pt, pb = same_pad(ish[1], kh, sh)
                pl, pr = same_pad(ish[2], kw, sw)
            else:
                pt = pb = pl = pr = 0
            if pad:                                   # a PAD before the convolution: folded into its padding
                pt, pb, pl, pr = pt + pad[0], pb + pad[1], pl + pad[2], pr + pad[3]
                pad = None
            recs.append(record(CONV, (kh, kw, cin, cout, sh, sw, pt, pb, pl, pr), (w.transpose(1, 2, 3, 0), b)))
        elif name == "LEAKY_RELU":
            recs.append(record(LEAKY, (), ([options(op, LeakyReluOptions).Alpha()],)))
        elif name == "MAXIMUM":
            c = G.const(g, ins[1])
            assert c is not None and c.size == 1
            recs.append(record(MAXC, (), ([float(c)],)))
        elif name == "MAX_POOL_2D":
            o = options(op, Pool2DOptions)
            assert o.FusedActivationFunction() == 0
            ish = G.shape(g, ins[0])
            fh, fw, sh, sw = o.FilterHeight(), o.FilterWidth(), o.StrideH(), o.StrideW()
            if o.Padding() == 0:
                pt, pb = same_pad(ish[1], fh, sh)
                pl, pr = same_pad(ish[2], fw, sw)
            else:
                pt = pb = pl = pr = 0
            recs.append(record(MAXPOOL, (fh, fw, sh, sw, pt, pb, pl, pr)))
        else:
            raise SystemExit(f"embedding model: operation {name} not supported")
    return program(recs)


def classifier_program(G, sub=0, bind=None):
    """The classifier's operations: dense layers, layer norms (MEAN SUB MUL MEAN ADD RSQRT MUL MUL ADD), sigmoid, and
    the IF of hey jarvis (score >= threshold: the other network's score). bind: for the IF's branch, its inputs as
    tensors of the main graph (the branch's weights are the IF's inputs)."""
    g = G.sub(sub)
    ops = G.ops(g)
    recs, i, thr = [], 0, None

    def const(t):
        if bind and t in bind:
            return G.const(G.sub(0), bind[t])
        return G.const(g, t)
    while i < len(ops):
        name, op = ops[i]
        ins = op.InputsAsNumpy()
        if name == "RESHAPE":
            i += 1
        elif name == "FULLY_CONNECTED":
            o = options(op, FullyConnectedOptions)
            w = const(ins[1])
            b = const(ins[2]) if len(ins) > 2 else None
            act = o.FusedActivationFunction()
            i += 1
            if b is None:   # the bias as a separate ADD (hey jarvis' second network)
                nxt, nop = ops[i]
                assert nxt == "ADD"
                b = const(nop.InputsAsNumpy()[1])
                act = options(nop, AddOptions).FusedActivationFunction()
                i += 1
            assert act in (0, 1), act
            recs.append(record(DENSE, (w.shape[1], w.shape[0], act), (w.T, b)))
        elif name == "MEAN":
            seq = [n for n, _ in ops[i:i + 9]]
            assert seq == ["MEAN", "SUB", "MUL", "MEAN", "ADD", "RSQRT", "MUL", "MUL", "ADD"], seq
            eps = float(const(ops[i + 4][1].InputsAsNumpy()[1]))
            gamma = const(ops[i + 7][1].InputsAsNumpy()[1])
            beta = const(ops[i + 8][1].InputsAsNumpy()[1])
            act = options(ops[i + 8][1], AddOptions).FusedActivationFunction()
            assert act in (0, 1), act
            recs.append(record(NORM, (gamma.size, act), ([eps], gamma, beta)))
            i += 9
        elif name == "LOGISTIC":
            recs.append(record(SIGMOID))
            i += 1
        elif name == "GREATER_EQUAL":
            thr = float(const(ins[1]))
            i += 1
        elif name == "IF":
            o = options(op, IfOptions)
            assert thr is not None
            # the else branch returns the score unchanged; the then branch is another network on the input
            recs.append(record(GATE, (), ([thr],)))
            sg = G.sub(o.ThenSubgraphIndex())
            args = list(op.InputsAsNumpy())[1:]   # (the first is the condition)
            then = classifier_program(G, o.ThenSubgraphIndex(), dict(zip(sg.InputsAsNumpy(), args)))
            return struct.pack("<I", len(recs) + struct.unpack_from("<I", then)[0]) + b"".join(recs) + then[4:]
        else:
            raise SystemExit(f"classifier: operation {name} not supported")
    return program(recs)


def phrase(name):
    return " ".join(w.capitalize() for w in name.replace("_", " ").split())


def main():
    def find(stem):
        for cand in (SRC / f"{stem}.tflite", SRC / f"{stem}_v0.1.tflite"):
            if cand.exists():
                return cand
        return None

    mel_path, emb_path = find("melspectrogram"), find("embedding_model")
    if not mel_path or not emb_path:
        raise SystemExit(f"melspectrogram.tflite and embedding_model.tflite are not in {SRC}")
    mel = mel_part(Graph(mel_path))
    out = b"OWW1" + struct.pack("<I", 1) + mel + embedding_program(Graph(emb_path))
    words = [(w, find(w)) for w in WORDS]
    extra = sorted(p for p in SRC.glob("*.tflite")
                   if p.stem not in ("melspectrogram", "embedding_model") and p.stem not in WORDS
                   and p.stem.replace("_v0.1", "") not in WORDS)
    words = [(w, p) for w, p in words if p] + [(p.stem.replace("_v0.1", ""), p) for p in extra]
    out += struct.pack("<I", len(words))
    for name, path in words:
        G = Graph(path)
        shp = G.shape(G.sub(0), G.sub(0).InputsAsNumpy()[0])
        assert len(name) < 32 and shp[2] == 96, (name, shp)
        out += struct.pack("<32s32s2I", name.encode(), phrase(name).encode()[:31], shp[1], shp[2])
        out += classifier_program(G)
        print(f"{name}: {path.name}, {shp[1]} embeddings")
    OUT.write_bytes(out)
    print(f"{OUT}: {len(out) / 2**20:.2f} MiB, {len(words)} wake words")


if __name__ == "__main__":
    main()
