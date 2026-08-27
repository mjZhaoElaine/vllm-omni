# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""E2E online serving tests for Gepard-1.0 via ``/v1/audio/speech``.

Zero-shot default voice. Needs a GPU and the NeMo NanoCodec; the weekly
``TTS · Gepard-1.0 · L4`` step installs NeMo then runs this file.
"""

from __future__ import annotations

import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import pytest
import requests
import soundfile as sf

from tests.helpers.mark import hardware_test
from tests.helpers.media import convert_audio_bytes_to_text
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytest.importorskip("nemo.collections.tts.models")

pytestmark = [pytest.mark.slow, pytest.mark.tts]

MODEL = "nineninesix/gepard-1.0"
SAMPLE_RATE = 22050
SAMPLES_PER_FRAME = 1024
PCM16_BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2
DEFAULT_TIMEOUT_S = 180.0

_DISTINGUISHABLE_PROMPTS = {
    "He drinks coffee every morning.": "coffee",
    "Machine learning is interesting.": "learning",
    "Please close the window before leaving.": "window",
    "My favorite color is purple.": "purple",
}

tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("gepard.yaml"),
            server_args=["--trust-remote-code", "--disable-log-stats"],
            stage_init_timeout=900,
        ),
        id="gepard",
    )
]


def _base_config(omni_server, text: str, **extra) -> dict:
    cfg = {
        "model": omni_server.model,
        "input": text,
        "voice": "default",
        "timeout": DEFAULT_TIMEOUT_S,
        "seed": 7,
        # NanoCodec is quieter than the 24 kHz models the PCM HNR helper was
        # calibrated on; keep the catastrophic-failure check, not a quality gate.
        "min_hnr_db": -2.0,
        "transcript_escalation_model": "large-v3",
    }
    cfg.update(extra)
    return cfg


def _wav_sample_rate(wav_bytes: bytes) -> int:
    return struct.unpack_from("<I", wav_bytes, 24)[0]


def _wav_pcm_payload_len(wav_bytes: bytes) -> int:
    data, _sr = sf.read(BytesIO(wav_bytes), dtype="int16")
    return int(data.size) * 2


def _scan_logs_for_preemption(omni_server) -> None:
    log_paths = getattr(omni_server, "_stage_log_paths", {}) or {}
    hits: list[str] = []
    for path in log_paths.values():
        if path is None or not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        if "preempt" in text:
            hits.append(str(path))
    assert not hits, f"preemption mentioned in stage logs: {hits}"


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
@pytest.mark.parametrize("response_format", ["wav", "pcm"])
def test_text_to_audio_basic(omni_server, online_client, response_format: str, run_level: str) -> None:
    text = "Hello, this is Gepard speaking."
    # whisper-small/large-v3 mishear 22.05 kHz WAV of this prompt as
    # "Jeb Ard" / "Jeff Bard" (cosine 0.80 / 0.75). Offline uses keyword
    # containment; the PCM twin below still goes through the 0.9 ASR gate.
    if response_format == "wav":
        url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
        payload = {
            "model": omni_server.model,
            "input": text,
            "voice": "default",
            "seed": 7,
            "stream": False,
            "response_format": "wav",
        }
        r = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT_S)
        r.raise_for_status()
        audio = r.content
        assert _wav_sample_rate(audio) == SAMPLE_RATE
        pcm_len = _wav_pcm_payload_len(audio)
    else:
        [resp] = online_client.send_audio_speech_request(
            _base_config(omni_server, text, stream=False, response_format=response_format)
        )
        assert resp.audio_bytes
        pcm_len = len(resp.audio_bytes)
    assert pcm_len % PCM16_BYTES_PER_FRAME == 0
    if run_level in {"advanced_model", "full_model"}:
        duration = pcm_len / 2 / SAMPLE_RATE
        assert 0.5 < duration < 30.0, f"implausible duration {duration:.2f}s"


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_seed_determinism(omni_server, online_client) -> None:
    text = "Hello, this is Gepard speaking."
    [a] = online_client.send_audio_speech_request(_base_config(omni_server, text, seed=7, response_format="pcm"))
    [b] = online_client.send_audio_speech_request(_base_config(omni_server, text, seed=7, response_format="pcm"))
    [c] = online_client.send_audio_speech_request(_base_config(omni_server, text, seed=11, response_format="pcm"))
    assert a.audio_bytes == b.audio_bytes
    assert a.audio_bytes != c.audio_bytes


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_streaming_pcm_incremental(omni_server, online_client) -> None:
    text = "Hello, this is Gepard speaking."
    url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
    payload = {
        "model": omni_server.model,
        "input": text,
        "voice": "default",
        "seed": 7,
        "stream": True,
        "stream_format": "audio",
        "response_format": "pcm",
    }
    start = time.perf_counter()
    arrivals: list[tuple[float, int]] = []
    streamed = bytearray()
    with requests.post(url, json=payload, stream=True, timeout=DEFAULT_TIMEOUT_S) as resp:
        resp.raise_for_status()
        content_type = resp.headers.get("content-type") or ""
        assert "pcm" in content_type or "octet-stream" in content_type
        for chunk in resp.iter_content(chunk_size=None):
            if not chunk:
                continue
            arrivals.append((time.perf_counter(), len(chunk)))
            streamed.extend(chunk)
    assert len(arrivals) >= 2, f"expected incremental chunks, got {len(arrivals)}"
    assert arrivals[1][0] > arrivals[0][0]
    ttfa_ms = (arrivals[0][0] - start) * 1000.0
    cadence_ms = [(arrivals[i][0] - arrivals[i - 1][0]) * 1000.0 for i in range(1, len(arrivals))]
    print(f"[Gepard streaming] TTFA_ms={ttfa_ms:.1f} chunk_cadence_ms={cadence_ms}")

    [non_stream] = online_client.send_audio_speech_request(
        _base_config(omni_server, text, seed=7, stream=False, response_format="pcm")
    )
    assert bytes(streamed) == non_stream.audio_bytes


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_seeded_streaming_stops(omni_server) -> None:
    url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
    payload = {
        "model": omni_server.model,
        "input": "Hello, this is Gepard speaking.",
        "voice": "default",
        "seed": 7,
        "stream": True,
        "stream_format": "audio",
        "response_format": "pcm",
    }
    with requests.post(url, json=payload, stream=True, timeout=DEFAULT_TIMEOUT_S) as resp:
        resp.raise_for_status()
        audio = b"".join(chunk for chunk in resp.iter_content(chunk_size=None) if chunk)
    duration = len(audio) / 2 / SAMPLE_RATE
    assert duration < 30.0, f"seeded stream ran {duration:.1f}s — stop constraint dropped?"


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_sse_stream_emits_delta_then_done(omni_server) -> None:
    url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
    payload = {
        "model": omni_server.model,
        "input": "Hello, this is Gepard speaking.",
        "voice": "default",
        "seed": 7,
        "stream": True,
        "stream_format": "sse",
        "response_format": "pcm",
    }
    with requests.post(url, json=payload, stream=True, timeout=DEFAULT_TIMEOUT_S) as resp:
        resp.raise_for_status()
        assert resp.headers.get("content-type", "").startswith("text/event-stream")
        body = b"".join(resp.iter_content(chunk_size=None)).decode("utf-8", errors="replace")
    assert "speech.audio.delta" in body
    assert "speech.audio.done" in body


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_concurrent_requests_stay_isolated(omni_server, online_client, run_level: str) -> None:
    def _one(text: str):
        return online_client.send_audio_speech_request(_base_config(omni_server, text, seed=7, response_format="wav"))[
            0
        ]

    texts = list(_DISTINGUISHABLE_PROMPTS)
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(_one, texts))

    assert all(r.success for r in responses)
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            assert responses[i].audio_bytes != responses[j].audio_bytes

    if run_level in {"advanced_model", "full_model"}:
        for text, resp in zip(texts, responses, strict=True):
            keyword = _DISTINGUISHABLE_PROMPTS[text]
            transcript = convert_audio_bytes_to_text(resp.audio_bytes, language="en").lower()
            assert keyword in transcript, f"expected {keyword!r} in {transcript!r} for {text!r}"

    _scan_logs_for_preemption(omni_server)
