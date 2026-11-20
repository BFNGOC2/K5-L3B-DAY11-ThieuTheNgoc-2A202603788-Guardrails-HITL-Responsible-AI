from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attacks import attacks
import agents.agent as agent_module
import agents.guards_agent as guards_module
import core.config as config_module
from core.config import DEMO_SECRETS, get_llm_provider, get_model_name
import main as lab_main


def _isolate_attack_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(
        attacks,
        "_attack_checkpoint_path",
        lambda target, fingerprint: tmp_path / f"{target}-{fingerprint}.json",
    )


def test_adversarial_prompts_cover_five_distinct_techniques_without_fixture_values():
    prompts = attacks.adversarial_prompts

    assert len(prompts) >= 5
    assert len({prompt["id"] for prompt in prompts}) == len(prompts)
    assert len({prompt["category"] for prompt in prompts}) >= 5
    for prompt in prompts:
        assert {"id", "category", "input"} <= prompt.keys()
        assert len(prompt["input"]) >= 100
        assert "TODO" not in prompt["input"]
        for secret in DEMO_SECRETS:
            assert secret.casefold() not in prompt["input"].casefold()


@pytest.mark.parametrize(
    "status_code,expected",
    [
        (400, "HTTP 400 invalid request"),
        (401, "HTTP 401 authentication"),
        (402, "HTTP 402 insufficient credits"),
        (404, "HTTP 404 model not found"),
        (429, "HTTP 429 rate limited"),
        (500, "HTTP 500 upstream internal error"),
        (502, "HTTP 502 upstream bad gateway"),
        (503, "HTTP 503 upstream unavailable"),
    ],
)
def test_provider_error_classification_is_status_only(status_code, expected):
    error = RuntimeError("sensitive provider response body")
    error.status_code = status_code

    assert attacks._safe_provider_error(error) == expected


def test_run_attacks_stops_on_provider_error_and_sanitizes_evidence(
    monkeypatch, caplog, tmp_path
):
    calls = []
    _isolate_attack_checkpoint(monkeypatch, tmp_path)

    async def fail_request(agent, runner, prompt):
        calls.append(prompt)
        try:
            raise RuntimeError("sensitive provider response body")
        except RuntimeError:
            logging.getLogger("google.adk.workflow._node_runner").exception(
                "Node execution failed with exception"
            )
        error = RuntimeError("untrusted response body must not enter artifacts")
        error.status_code = 500
        raise error

    async def no_wait(_):
        return None

    monkeypatch.setattr(attacks, "chat_with_agent", fail_request)
    monkeypatch.setattr(attacks.asyncio, "sleep", no_wait)
    prompts = [
        {"id": index, "category": "test", "input": f"Prompt {index}"}
        for index in range(1, 4)
    ]

    results = asyncio.run(
        attacks.run_attacks(
            object(),
            object(),
            prompts=prompts,
            target_name="red_default",
            save_json=False,
        )
    )

    assert len(calls) == 3
    assert len(results) == 1
    assert results[0]["layer"] == "error"
    assert results[0]["leaked"] is False
    assert results[0]["error"] == "HTTP 500 upstream internal error"
    assert "untrusted response body" not in json.dumps(results)
    assert "sensitive provider response body" not in caplog.text


def test_run_attacks_retries_transient_error_but_records_one_logical_result(
    monkeypatch, tmp_path
):
    calls = []
    delays = []
    _isolate_attack_checkpoint(monkeypatch, tmp_path)

    async def fail_once_then_succeed(agent, runner, prompt):
        calls.append(prompt)
        if len(calls) == 1:
            error = RuntimeError("sensitive provider body")
            error.status_code = 503
            raise error
        return "I cannot provide internal configuration.", None

    async def record_delay(delay):
        delays.append(delay)

    monkeypatch.setattr(attacks, "chat_with_agent", fail_once_then_succeed)
    monkeypatch.setattr(attacks.asyncio, "sleep", record_delay)
    monkeypatch.setattr(attacks.random, "uniform", lambda _low, _high: 0.01)

    results = asyncio.run(
        attacks.run_attacks(
            object(),
            object(),
            prompts=[{"id": 1, "category": "test", "input": "Test prompt"}],
            target_name="red_default",
            save_json=False,
        )
    )

    assert len(calls) == 2
    assert len(results) == 1
    assert results[0]["layer"] == "model_refuse"
    assert delays == [pytest.approx(0.26)]


@pytest.mark.parametrize("status_code", [400, 401, 402, 404])
def test_run_attacks_does_not_retry_permanent_provider_errors(
    monkeypatch, status_code, tmp_path
):
    calls = []
    _isolate_attack_checkpoint(monkeypatch, tmp_path)

    async def fail_request(agent, runner, prompt):
        calls.append(prompt)
        error = RuntimeError("sensitive provider body")
        error.status_code = status_code
        raise error

    monkeypatch.setattr(attacks, "chat_with_agent", fail_request)

    results = asyncio.run(
        attacks.run_attacks(
            object(),
            object(),
            prompts=[{"id": 1, "category": "test", "input": "Test prompt"}],
            target_name="red_default",
            save_json=False,
        )
    )

    assert len(calls) == len(results) == 1
    assert results[0]["layer"] == "error"
    assert "sensitive provider body" not in json.dumps(results)


def test_attack_fingerprint_changes_with_prompt_or_model(monkeypatch):
    prompts = [{"id": 1, "category": "technique", "input": "prompt"}]
    original = attacks.attack_run_fingerprint("red_default", prompts)
    changed_prompt = attacks.attack_run_fingerprint(
        "red_default", [{**prompts[0], "input": "changed prompt"}]
    )
    monkeypatch.setattr(config_module, "get_model_name", lambda: "different-model")
    changed_model = attacks.attack_run_fingerprint("red_default", prompts)

    assert original != changed_prompt
    assert original != changed_model


def test_seeded_checkpoint_ignores_error_rows_and_resumes_pending_ids(
    monkeypatch, tmp_path
):
    prompts = [
        {"id": 1, "category": "one", "input": "first prompt"},
        {"id": 2, "category": "two", "input": "second prompt"},
    ]
    monkeypatch.setattr(
        attacks,
        "_attack_checkpoint_path",
        lambda target, fingerprint: tmp_path / f"{target}-{fingerprint}.json",
    )
    rows = [
        {
            "id": 1,
            "category": "one",
            "input": "first prompt",
            "response_preview": "A verified response.",
            "leaked": False,
            "blocked_input": False,
            "blocked": False,
            "layer": "model_refuse",
            "blocked_at": "MODEL_REFUSE",
            "target": "red_default",
        },
        {
            "id": 2,
            "category": "two",
            "input": "second prompt",
            "response_preview": "HTTP 503 upstream unavailable",
            "leaked": False,
            "blocked_input": False,
            "blocked": False,
            "layer": "error",
            "blocked_at": "ERROR",
            "target": "red_default",
        },
    ]

    fingerprint, pending = attacks._seed_verified_attack_checkpoint(
        "red_default",
        prompts,
        rows,
        recovered_from="verified test fixture",
    )
    restored = attacks.load_completed_attack_results("red_default", prompts)

    assert fingerprint
    assert pending == [2]
    assert [row["id"] for row in restored] == [1]
    checkpoint_path = attacks._attack_checkpoint_path("red_default", fingerprint)
    serialized = checkpoint_path.read_text(encoding="utf-8")
    assert "HTTP 503" not in serialized
    assert "api_key" not in serialized


def test_run_attacks_resumes_successes_and_retries_only_pending_prompts(
    monkeypatch, tmp_path
):
    prompts = [
        {"id": 1, "category": "one", "input": "prompt one"},
        {"id": 2, "category": "two", "input": "prompt two"},
        {"id": 3, "category": "three", "input": "prompt three"},
    ]
    monkeypatch.setattr(
        attacks,
        "_attack_checkpoint_path",
        lambda target, fingerprint: tmp_path / f"{target}-{fingerprint}.json",
    )

    async def no_sleep(_):
        return None

    monkeypatch.setattr(attacks.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(attacks.random, "uniform", lambda _low, _high: 0.0)
    calls = []
    fail_prompt_two = [True]

    async def fake_chat(agent, runner, prompt):
        calls.append(prompt)
        if prompt == "prompt two" and fail_prompt_two[0]:
            error = RuntimeError("unpersisted provider body")
            error.status_code = 503
            raise error
        return "I cannot provide internal configuration.", None

    monkeypatch.setattr(attacks, "chat_with_agent", fake_chat)
    first = asyncio.run(
        attacks.run_attacks(
            object(),
            object(),
            prompts,
            target_name="red_default",
            output_path=tmp_path / "unsafe.json",
        )
    )
    assert [row["id"] for row in first] == [1, 2]
    assert first[1]["layer"] == "error"
    assert calls == ["prompt one", "prompt two", "prompt two", "prompt two"]

    calls.clear()
    fail_prompt_two[0] = False
    second = asyncio.run(
        attacks.run_attacks(
            object(),
            object(),
            prompts,
            target_name="red_default",
            output_path=tmp_path / "unsafe.json",
        )
    )

    assert [row["id"] for row in second] == [1, 2, 3]
    assert calls == ["prompt two", "prompt three"]
    assert all(row["layer"] != "error" for row in second)


def test_attack_results_writer_records_both_targets_and_configured_model(tmp_path):
    row = {
        "id": 1,
        "category": "test technique",
        "input": "A test prompt.",
        "response_preview": "A safe refusal.",
        "leaked": False,
        "blocked_input": True,
        "blocked": True,
        "layer": "input_injection",
        "blocked_at": "BLOCKED_INPUT",
        "target": "red_default",
    }
    output = tmp_path / "attack_results.json"

    attacks.save_attack_results(
        unsafe_results=[row],
        guards_results=[{**row, "target": "red_advance"}],
        filepath=output,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["llm_provider"] == get_llm_provider()
    assert payload["llm_model"] == get_model_name()
    assert len(payload["unsafe_attacks"]) == 1
    assert len(payload["guards_attacks"]) == 1
    assert payload["unsafe_attacks"][0]["leaked"] is False
    attacks.save_attack_results(
        unsafe_results=[row],
        guards_results=[{**row, "target": "red_advance"}],
        filepath=output,
        run_status="PARTIAL",
        pending_attack_ids={"red_default": [2]},
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["run_status"] == "PARTIAL"
    assert payload["pending_attack_ids"] == {"red_default": [2]}


def test_part4_does_not_call_red_advance_after_red_provider_error(
    monkeypatch, tmp_path
):
    saved = []
    not_run_artifacts = []

    monkeypatch.setattr(
        agent_module,
        "create_red_agent_default",
        lambda: (object(), object()),
    )
    monkeypatch.setattr(
        guards_module,
        "create_red_agent_advance",
        lambda: pytest.fail("Red Advance must not run after provider failure"),
    )

    async def fail_red(agent, runner, target_name):
        return [
            {
                "id": 1,
                "layer": "error",
                "error": "HTTP 500 upstream internal error",
            }
        ]

    monkeypatch.setattr(attacks, "run_attacks", fail_red)
    monkeypatch.setattr(attacks, "load_completed_attack_results", lambda *args: [])
    monkeypatch.setattr(attacks, "save_attack_results", lambda **kwargs: saved.append(kwargs))
    monkeypatch.setattr(lab_main, "_SRC_DIR", tmp_path / "src")
    monkeypatch.setattr(
        attacks,
        "write_run_attack_json",
        lambda results, **kwargs: not_run_artifacts.append((results, kwargs)),
    )

    result = asyncio.run(lab_main.part4_attacks())

    assert result["red_advance"] == []
    assert saved[0]["guards_results"] == []
    assert saved[0]["unsafe_results"][0]["layer"] == "error"
    assert not_run_artifacts == [([], {"target_name": "red_advance"})]


def test_main_reports_part4_incomplete_when_provider_fails(monkeypatch, capsys):
    async def partial_part4():
        return {
            "red_default": [{"layer": "error"}],
            "red_advance": [],
        }

    monkeypatch.setattr(lab_main, "setup_api_key", lambda: None)
    monkeypatch.setattr(lab_main, "part4_attacks", partial_part4)

    asyncio.run(lab_main.main(parts=[4]))

    output = capsys.readouterr().out
    assert "Lab 11 incomplete" in output
    assert "Lab 11 complete!" not in output
