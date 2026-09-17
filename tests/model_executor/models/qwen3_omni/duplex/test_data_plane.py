# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.model_executor.models.qwen3_omni.duplex.data_plane import Qwen3OmniDataPlaneSession

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _encode_audio(audio: object, sample_rate: int, fmt: str, speed: float | None) -> str | None:
    del audio, sample_rate, fmt, speed
    return "ZmFrZQ=="


def test_data_plane_projects_thinker_text_then_audio() -> None:
    plane = Qwen3OmniDataPlaneSession(_encode_audio)
    request_id = "sess.e0.r.stage0_t0"
    plane.begin_request(request_id)

    thinker = SimpleNamespace(
        request_id=request_id,
        stage_id=0,
        finished=False,
        outputs=[SimpleNamespace(text="hello", cumulative_text="hello", multimodal_output={})],
        multimodal_output={},
    )
    text_events = list(plane.project_output(thinker))
    assert text_events == [
        {
            "stage_role": "thinker",
            "is_listen": False,
            "data_plane_request_id": request_id,
            "text": "hello",
            "end_of_turn": False,
        }
    ]

    audio = np.zeros(8, dtype=np.float32)
    wav = SimpleNamespace(
        request_id=request_id,
        stage_id=2,
        finished=True,
        outputs=[SimpleNamespace(text="", multimodal_output={"audio": audio, "sr": 24000})],
        multimodal_output={"audio": audio, "sr": 24000},
    )
    audio_events = list(plane.project_output(wav))
    assert audio_events[0]["stage_role"] == "tts"
    assert audio_events[0]["audio"] == "ZmFrZQ=="
    assert audio_events[0]["end_of_turn"] is True
    assert plane.is_terminal(request_id) is True
