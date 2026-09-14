# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Record eager Gepard serving baselines (TTFP / RTF / throughput).

Prints ``SERVING_BASELINE=`` JSON for a live
``/v1/audio/speech`` server: TTFP (time to first PCM byte), RTF
(wall / audio duration), and throughput (audio-seconds per wall-second,
and requests per second).

  python measure_serving_baseline.py --url http://127.0.0.1:8091
  python measure_serving_baseline.py --url http://127.0.0.1:8091 --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import requests

SAMPLE_RATE = 22050
DEFAULT_TEXT = "Hello, this is Gepard speaking."
PROMPTS = (
    "He drinks coffee every morning.",
    "Machine learning is interesting.",
    "Please close the window before leaving.",
    "My favorite color is purple.",
)


def _stream(url: str, text: str, timeout_s: float, seed: int) -> dict:
    payload = {
        "model": "nineninesix/gepard-1.0",
        "input": text,
        "voice": "default",
        "seed": seed,
        "stream": True,
        "stream_format": "audio",
        "response_format": "pcm",
    }
    start = time.perf_counter()
    pcm = 0
    ttfp_s = None
    with requests.post(f"{url.rstrip('/')}/v1/audio/speech", json=payload, stream=True, timeout=timeout_s) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=None):
            if not chunk:
                continue
            if ttfp_s is None:
                ttfp_s = time.perf_counter() - start
            pcm += len(chunk)
    wall_s = time.perf_counter() - start
    duration_s = pcm / 2 / SAMPLE_RATE
    return {
        "ttfp_ms": None if ttfp_s is None else round(ttfp_s * 1000.0, 1),
        "wall_s": round(wall_s, 3),
        "audio_s": round(duration_s, 3),
        "rtf": None if duration_s <= 0 else round(wall_s / duration_s, 3),
        "pcm_bytes": pcm,
    }


def _report(label: str, rows: list[dict], wall_s: float) -> dict:
    ttfp = [r["ttfp_ms"] for r in rows if r["ttfp_ms"] is not None]
    rtf = [r["rtf"] for r in rows if r["rtf"] is not None]
    audio_s = sum(r["audio_s"] for r in rows)
    report = {
        "label": label,
        "n": len(rows),
        "ttfp_ms": ttfp,
        "ttfp_ms_median": round(statistics.median(ttfp), 1) if ttfp else None,
        "rtf": rtf,
        "rtf_median": round(statistics.median(rtf), 3) if rtf else None,
        "audio_s": round(audio_s, 3),
        "wall_s": round(wall_s, 3),
        "throughput_audio_x": round(audio_s / wall_s, 3) if wall_s else 0.0,
        "throughput_rps": round(len(rows) / wall_s, 3) if wall_s else 0.0,
        "rows": rows,
    }
    print(f"SERVING_BASELINE={json.dumps(report, separators=(',', ':'))}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    parser.add_argument("--concurrency", type=int, default=1, choices=(1, 2, 4))
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if args.concurrency == 1:
        t0 = time.perf_counter()
        rows = [_stream(args.url, DEFAULT_TEXT, args.timeout_s, args.seed)]
        wall_s = time.perf_counter() - t0
        label = "concurrency=1"
    else:
        texts = list(PROMPTS[: args.concurrency])
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            rows = list(pool.map(lambda text: _stream(args.url, text, args.timeout_s, args.seed), texts))
        wall_s = time.perf_counter() - t0
        label = f"concurrency={args.concurrency}"

    report = _report(label, rows, wall_s)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
