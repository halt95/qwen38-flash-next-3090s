#!/usr/bin/env python3
"""Calibration traffic for the QSA K/V absmax collector (patch 0002).

Real-shape coverage: varied pseudo-random text at 2K/8K/32K/128K/200K/255K depth,
plus image-bearing chat prompts (vision tokens traverse the same QSA layers).
Server must run with VLLM_QSA_KV_COLLECT set and --enforce-eager
(calib/calib_launch.sh). This is the traffic the shipped sidecar was made with.

    python calib/qsa_calib_traffic.py [port=8140] [model=flash-next-calib]
"""
import base64
import io
import json
import random
import sys
import time
import urllib.request

try:
    import PIL  # noqa: F401
except ImportError:
    sys.exit("Pillow is required (run this under the vLLM venv's python); refusing to calibrate without the image arm")

PORT = sys.argv[1] if len(sys.argv) > 1 else "8140"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "flash-next-calib"
BASE = f"http://127.0.0.1:{PORT}/v1"

WORDS = ("harbour dawn quiet boats rest still granite ledger crimson orbit velvet "
         "thunder archive lantern mosaic quarry sable meridian copper glacier fable "
         "sextant bramble hollow zenith cinder plume rivet tundra ember lattice").split()
rng = random.Random(1949)


def text(n_words):
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def completion(prompt, max_tokens=64, ignore_eos=False):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.8, "ignore_eos": ignore_eos}
    req = urllib.request.Request(f"{BASE}/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    d = json.load(urllib.request.urlopen(req, timeout=3600))
    print(f"  text: {d['usage']['prompt_tokens']} prompt tok, "
          f"{d['usage']['completion_tokens']} gen, {time.perf_counter()-t0:.1f}s",
          flush=True)
    return d["usage"]["prompt_tokens"]


def image_chat(n_words):
    from PIL import Image  # required: a run without the image arm is not a valid calibration
    img = Image.new("RGB", (896, 896))
    px = img.load()
    r = random.Random(7)
    for y in range(0, 896, 8):
        for x in range(0, 896, 8):
            c = (r.randrange(256), r.randrange(256), r.randrange(256))
            for dy in range(8):
                for dx in range(8):
                    px[x + dx, y + dy] = c
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    body = {"model": MODEL, "max_tokens": 64, "temperature": 0.8, "messages": [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": uri}},
            {"type": "text", "text": "Describe the image. Context: " + text(n_words)},
        ]}]}
    req = urllib.request.Request(f"{BASE}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    d = json.load(urllib.request.urlopen(req, timeout=3600))
    print(f"  image: {d['usage']['prompt_tokens']} prompt tok, "
          f"{time.perf_counter()-t0:.1f}s", flush=True)


# Measured on this server: the word list tokenizes at ~1.26 tokens/word, so
# words = tokens / 1.26. The deepest arm sits just under max-model-len 262144.
RATIO = 1.26


def words_for(tokens):
    return int(tokens / RATIO)


for label, toks in (("2k", 2000), ("8k", 8000), ("32k", 32000)):
    print(f"[{label}]", flush=True)
    completion(text(words_for(toks)))
print("[image x2]", flush=True)
image_chat(words_for(2000))
image_chat(words_for(8000))
deepest = 0
for label, toks in (("128k", 128000), ("200k", 200000), ("255k", 255000)):
    print(f"[{label}]", flush=True)
    deepest = max(deepest, completion(text(words_for(toks))))
if deepest < 250000:
    sys.exit(f"deepest arm reached only {deepest} prompt tokens (< 250000): the tokens/word ratio is off for this tokenizer; fix RATIO")
# Flush: the collector persists its running maxima every 2,000 layer calls, not on
# a timer, so the deepest request's maxima may still be unpersisted when traffic
# ends. A dump boundary is at most 2,000 / 12 layers = 167 model forwards away;
# 12 completions forced to 64 tokens each (ignore_eos) are at least 768 model
# forwards (one prefill plus 63 decode forwards each), so at least four boundaries are crossed. Check every rank file's mtime is later
# than the deepest request before merging.
print("[flush]", flush=True)
for _ in range(12):
    completion(text(200), max_tokens=64, ignore_eos=True)
print("TRAFFIC DONE", flush=True)
