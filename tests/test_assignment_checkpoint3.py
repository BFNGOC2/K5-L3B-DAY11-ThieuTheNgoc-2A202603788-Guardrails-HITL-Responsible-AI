from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import jsonschema
import openai
import pytest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.security_boundary import TRUSTED_EGRESS_HOSTS
from assignment import audit_log, monitoring, pipeline, rate_limiter
from assignment.audit_log import AuditLogPlugin
from assignment.monitoring import MonitoringAlert
from assignment.rate_limiter import RateLimitPlugin
import core.openai_runtime as openai_runtime
from core.openai_runtime import OpenAIAgent, OpenAIRunner
from guardrails.input_guardrails import InputGuardrailPlugin
from guardrails.output_guardrails import OutputGuardrailPlugin
import main as main_module


def _message(text: str = "safe banking request") -> types.Content:
    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


def test_rate_limiter_sliding_window_boundary_and_user_isolation(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(rate_limiter.time, "monotonic", lambda: now[0])
    limiter = RateLimitPlugin(max_requests=2, window_seconds=10)

    async def call(user_id: str):
        return await limiter.on_user_message_callback(
            invocation_context=SimpleNamespace(user_id=user_id),
            user_message=_message(),
        )

    assert asyncio.run(call("alice")) is None
    assert asyncio.run(call("alice")) is None
    now[0] = 109.0
    assert asyncio.run(call("alice")) is not None
    assert asyncio.run(call("bob")) is None
    now[0] = 110.0
    assert asyncio.run(call("alice")) is None
    assert limiter.total_count == 5
    assert limiter.blocked_count == 1


def test_audit_log_redacts_request_response_and_user_id(tmp_path):
    audit = AuditLogPlugin()
    request_id = "request-safe-id"
    audit.record_input(
        user_id="customer@example.com",
        text=(
            "Call 0901234567; CCCD 079204001234; "
            "password=Sensitive,Value!; key=sk-demo-private-key"
        ),
        request_id=request_id,
    )
    audit.record_output(
        user_id="customer@example.com",
        text="Internal credential admin123 at db.vinbank.internal:5432.",
        blocked=True,
        layer="input_guardrail",
        request_id=request_id,
    )

    path = tmp_path / "nested" / "audit.json"
    audit.export_json(str(path))
    records = json.loads(path.read_text(encoding="utf-8"))
    assert len(records) == 1
    record = records[0]
    serialized = json.dumps(record)
    for sensitive in (
        "customer@example.com",
        "0901234567",
        "079204001234",
        "Sensitive,Value!",
        "sk-demo-private-key",
        "admin123",
        "db.vinbank.internal",
    ):
        assert sensitive not in serialized
    assert record["status"] == "blocked"
    assert record["reason"] == "input_guardrail"
    assert record["metadata"]["request_redacted"] is True
    assert record["metadata"]["response_redacted"] is True
    assert record["latency_ms"] >= 0
    assert record["started_at"] and record["completed_at"]


def test_monitoring_counts_latency_and_refreshes_alerts(tmp_path):
    monitor = MonitoringAlert(
        block_rate_threshold=0.5,
        rate_limit_hit_threshold=1,
        judge_fail_rate_threshold=0.5,
    )
    monitor.record_request(blocked=False, latency_ms=10)
    monitor.record_request(blocked=True, rate_limited=True, latency_ms=20)
    monitor.record_request(blocked=True, latency_ms=30)
    monitor.record_request(blocked=False, error=True, latency_ms=40)
    monitor.judge_checks = 2
    monitor.judge_fails = 1

    alerts = monitor.check_metrics()
    assert {alert.metric for alert in alerts} == {
        "block_rate", "rate_limit_hits", "judge_fail_rate"
    }
    assert {alert.metric for alert in monitor.check_metrics()} == {
        "block_rate", "rate_limit_hits", "judge_fail_rate"
    }
    snapshot = monitor.snapshot()
    assert snapshot["total_requests"] == 4
    assert snapshot["allowed_requests"] == 1
    assert snapshot["blocked_requests"] == 2
    assert snapshot["error_requests"] == 1
    assert snapshot["rate_limit_hits"] == 1
    assert snapshot["average_latency_ms"] == 25

    path = tmp_path / "metrics.json"
    monitor.export_json(str(path))
    assert json.loads(path.read_text(encoding="utf-8"))["total_requests"] == 4


@pytest.mark.parametrize(
    "destination,payload,expected",
    [
        ("https://api.vinbank.example/v1/transfers", "amount=500000", True),
        ("https://cases.vinbank.example/v1/cases", "ordinary case note", True),
        ("http://api.vinbank.example/v1/transfers", "amount=500000", False),
        ("https://api.vinbank.example.evil.com/path", "ordinary", False),
        ("https://sub.api.vinbank.example/path", "ordinary", False),
        ("https://user@api.vinbank.example/path", "ordinary", False),
        ("https://api.vinbank.example@evil.example/path", "ordinary", False),
        ("https://%61pi.vinbank.example/path", "ordinary", False),
        ("https://api.vinbank.example.:443/path", "ordinary", False),
        ("https://api.vinbank.example:444/path", "ordinary", False),
        ("https://127.0.0.1/path", "ordinary", False),
        ("https://169.254.169.254/path", "ordinary", False),
        ("https://[::1]/path", "ordinary", False),
        ("https://localhost/path", "ordinary", False),
        ("https://db.vinbank.internal/path", "ordinary", False),
        ("https://api.vinbank.example/path", "password=demo-secret", False),
        ("https://api.vinbank.example/path", "sk-private-api-key-123", False),
        ("https://api.vinbank.example/path", "customer@example.com", False),
        ("https://api.vinbank.example/path", "Call 0901234567", False),
    ],
)
def test_egress_policy_is_exact_and_rejects_sensitive_payloads(
    destination, payload, expected
):
    assert pipeline.is_egress_allowed(destination, payload) is expected
    assert "api.vinbank.example" in TRUSTED_EGRESS_HOSTS


def test_production_plugin_order():
    plugins = pipeline.build_production_plugins()
    assert [type(plugin) for plugin in plugins] == [
        RateLimitPlugin,
        InputGuardrailPlugin,
        OutputGuardrailPlugin,
    ]


def test_openai_runtime_passes_user_id_and_short_circuits_blocked_calls():
    limiter = RateLimitPlugin(max_requests=1, window_seconds=60)
    calls = []

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create_completion)
            )

        def create_completion(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Safe response.")
                    )
                ]
            )

    runner = OpenAIRunner(
        app_name="rate-test",
        model="fake",
        plugins=[limiter],
    )
    runner._client = FakeClient
    agent = OpenAIAgent(name="rate-test", instruction="Banking assistant.")

    async def run():
        alice_first = await runner.chat(agent, "first", user_id="alice")
        alice_second = await runner.chat(agent, "blocked", user_id="alice")
        bob_first = await runner.chat(agent, "first", user_id="bob")
        return alice_first, alice_second, bob_first

    alice_first, alice_second, bob_first = asyncio.run(run())
    assert alice_first == bob_first == "Safe response."
    assert alice_second.startswith("Rate limit exceeded.")
    assert len(calls) == 2
    assert limiter.total_count == 3
    assert limiter.blocked_count == 1


def test_run_assignment_suite_writes_consistent_real_callback_results(
    monkeypatch, tmp_path
):
    class FakeRunner:
        def __init__(self, plugins):
            self.plugins = plugins

    model_calls = []

    async def fake_chat(agent, runner, text, session_id=None, *, user_id="student"):
        user_content = _message(text)
        context = SimpleNamespace(user_id=user_id)
        for plugin in runner.plugins:
            callback = getattr(plugin, "on_user_message_callback", None)
            if callback is not None:
                replacement = await callback(
                    invocation_context=context,
                    user_message=user_content,
                )
                if replacement is not None:
                    return "".join(part.text or "" for part in replacement.parts), None

        model_calls.append(text)
        response = LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text="Safe VinBank banking answer.")],
            )
        )
        for plugin in runner.plugins:
            callback = getattr(plugin, "after_model_callback", None)
            if callback is not None:
                replacement = await callback(
                    callback_context=None,
                    llm_response=response,
                )
                if replacement is not None:
                    response = replacement
        response_text = "".join(
            part.text or "" for part in response.content.parts
        )
        return response_text, None

    monkeypatch.setattr(
        pipeline,
        "create_blue_agent",
        lambda plugins: (object(), FakeRunner(plugins)),
    )
    monkeypatch.setattr(pipeline, "chat_with_agent", fake_chat)
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path)
    audit = AuditLogPlugin()
    monitor = MonitoringAlert()
    plugins = pipeline.build_production_plugins()

    results = asyncio.run(
        pipeline.run_assignment_suite(
            {"plugins": plugins, "audit": audit, "monitor": monitor}
        )
    )

    assert len(model_calls) == 5
    assert len(results["safe_queries"]) >= 5
    assert all(not query["blocked"] for query in results["safe_queries"])
    assert len(results["attack_queries"]) >= 7
    assert sum(query["blocked"] for query in results["attack_queries"]) >= 5
    assert len(results["edge_cases"]) >= 3
    assert results["rate_limit"] == {
        "max_requests": 10,
        "window_seconds": 60,
        "sent": 15,
        "passed": 10,
        "blocked": 5,
    }
    assert monitor.total_requests == len(audit.logs)
    assert monitor.blocked_requests == sum(row["blocked"] for row in audit.logs)

    results_path = tmp_path / "results.json"
    jsonschema.validate(
        json.loads(results_path.read_text(encoding="utf-8")),
        json.loads((ROOT / "schemas" / "results.schema.json").read_text(encoding="utf-8")),
    )
    assert (tmp_path / "audit_log.json").is_file()
    assert (tmp_path / "metrics.json").is_file()


def _upstream_error(status_code: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return openai.APIStatusError(
        "upstream response",
        response=response,
        body={"error": {"message": "upstream response"}},
    )


def _retry_test_runner(outcomes, plugins=None):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="Safe answer."))
                ]
            )

    runner = OpenAIRunner(
        app_name="retry-test",
        model="liquid/lfm-3-1.7b:free",
        provider="openrouter",
        plugins=list(plugins or []),
    )
    runner._client = lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    agent = OpenAIAgent(name="retry-test", instruction="Banking assistant.")
    return runner, agent, calls


@pytest.mark.parametrize("status_code", [429, 500, 502, 503])
def test_blue_retries_transient_status_errors_at_most_twice(
    status_code, monkeypatch
):
    sleeps = []
    monkeypatch.setattr(openai_runtime.time, "sleep", sleeps.append)
    monkeypatch.setattr(openai_runtime.random, "uniform", lambda _low, _high: 0)
    limiter = RateLimitPlugin(max_requests=10, window_seconds=60)
    runner, agent, calls = _retry_test_runner(
        [_upstream_error(status_code), SimpleNamespace()],
        plugins=[limiter],
    )

    answer = asyncio.run(runner.chat(agent, "Check my balance."))

    assert answer == "Safe answer."
    assert len(calls) == 2
    assert sleeps == [openai_runtime._BLUE_RETRY_BASE_SECONDS]
    assert limiter.total_count == 1


@pytest.mark.parametrize("status_code", [400, 401, 402, 404])
def test_blue_does_not_retry_permanent_status_errors(status_code, monkeypatch):
    sleeps = []
    monkeypatch.setattr(openai_runtime.time, "sleep", sleeps.append)
    runner, agent, calls = _retry_test_runner([_upstream_error(status_code)])

    with pytest.raises(openai.APIStatusError):
        asyncio.run(runner.chat(agent, "Check my balance."))

    assert len(calls) == 1
    assert sleeps == []


def test_blue_retries_timeouts_and_caps_total_attempts(monkeypatch):
    sleeps = []
    monkeypatch.setattr(openai_runtime.time, "sleep", sleeps.append)
    monkeypatch.setattr(openai_runtime.random, "uniform", lambda _low, _high: 0)
    timeout = openai.APITimeoutError(
        request=httpx.Request(
            "POST", "https://openrouter.ai/api/v1/chat/completions"
        )
    )
    runner, agent, calls = _retry_test_runner(
        [timeout, _upstream_error(503), _upstream_error(503), SimpleNamespace()]
    )

    with pytest.raises(openai.APIStatusError):
        asyncio.run(runner.chat(agent, "Check my balance."))

    assert len(calls) == openai_runtime._BLUE_MAX_ATTEMPTS
    assert sleeps == [
        openai_runtime._BLUE_RETRY_BASE_SECONDS,
        openai_runtime._BLUE_RETRY_BASE_SECONDS * 2,
    ]


def test_blue_client_disables_nested_sdk_retries_and_sets_timeout(monkeypatch):
    captured = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(openai, "OpenAI", fake_openai)
    runner = OpenAIRunner(
        app_name="client-test",
        model="liquid/lfm-3-1.7b:free",
        provider="openrouter",
        client_kwargs={"base_url": "https://openrouter.ai/api/v1"},
    )

    runner._client()

    assert captured["max_retries"] == 0
    assert captured["timeout"] == openai_runtime._BLUE_REQUEST_TIMEOUT_SECONDS


def test_assignment_suite_checkpoints_and_resumes_without_duplicate_requests(
    monkeypatch, tmp_path
):
    previous_audit = '[{"status":"previous"}]'
    previous_metrics = '{"total_requests":99}'
    (tmp_path / "audit_log.json").write_text(previous_audit, encoding="utf-8")
    (tmp_path / "metrics.json").write_text(previous_metrics, encoding="utf-8")

    class FakeRunner:
        def __init__(self, plugins):
            self.plugins = plugins
            self.provider = "openrouter"
            self.model = "liquid/lfm-3-1.7b:free"
            self.temperature = 0.4
            self.client_kwargs = {
                "api_key": "must-not-be-persisted",
                "base_url": "https://openrouter.ai/api/v1",
            }

    model_calls = []
    fail_second_call = [True]

    async def fake_chat(agent, runner, text, session_id=None, *, user_id="student"):
        user_content = _message(text)
        context = SimpleNamespace(user_id=user_id)
        for plugin in runner.plugins:
            callback = getattr(plugin, "on_user_message_callback", None)
            if callback is None:
                continue
            replacement = await callback(
                invocation_context=context,
                user_message=user_content,
            )
            if replacement is not None:
                return "".join(part.text or "" for part in replacement.parts), None

        model_calls.append(text)
        if fail_second_call[0] and len(model_calls) == 2:
            raise _upstream_error(500)

        response = LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text="Safe VinBank banking answer.")],
            )
        )
        for plugin in runner.plugins:
            callback = getattr(plugin, "after_model_callback", None)
            if callback is None:
                continue
            replacement = await callback(
                callback_context=None,
                llm_response=response,
            )
            if replacement is not None:
                response = replacement
        return "".join(part.text or "" for part in response.content.parts), None

    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        pipeline,
        "create_blue_agent",
        lambda plugins: (object(), FakeRunner(plugins)),
    )
    monkeypatch.setattr(pipeline, "chat_with_agent", fake_chat)

    def run_with_fresh_observability():
        plugins = pipeline.build_production_plugins()
        audit = AuditLogPlugin()
        monitor = MonitoringAlert()
        result = asyncio.run(
            pipeline.run_assignment_suite(
                {"plugins": plugins, "audit": audit, "monitor": monitor}
            )
        )
        return result, audit, monitor

    partial, partial_audit, partial_monitor = run_with_fresh_observability()
    assert partial["status"] == "INCOMPLETE"
    assert partial["complete"] is False
    assert partial["safe_queries"][0].get("status") is None
    assert partial["safe_queries"][1]["status"] == "ERROR"
    assert partial["safe_queries"][2]["status"] == "NOT_RUN"
    assert not (tmp_path / "results.json").exists()
    assert (tmp_path / "results.partial.json").is_file()
    assert (tmp_path / "audit_log.partial.json").is_file()
    assert (tmp_path / "metrics.partial.json").is_file()
    assert (tmp_path / "audit_log.json").read_text(encoding="utf-8") == previous_audit
    assert (tmp_path / "metrics.json").read_text(encoding="utf-8") == previous_metrics
    assert len(partial_audit.logs) == partial_monitor.total_requests == 2
    assert partial_monitor.error_requests == 1

    checkpoint_path = next(tmp_path.glob("results.checkpoint-*.json"))
    assert "must-not-be-persisted" not in checkpoint_path.read_text(encoding="utf-8")

    fail_second_call[0] = False
    resumed, audit, monitor = run_with_fresh_observability()

    assert len(resumed["safe_queries"]) == 5
    assert all(not query["blocked"] for query in resumed["safe_queries"])
    assert len(resumed["attack_queries"]) == 7
    assert sum(query["blocked"] for query in resumed["attack_queries"]) >= 5
    assert len(resumed["edge_cases"]) >= 3
    assert resumed["rate_limit"]["passed"] + resumed["rate_limit"]["blocked"] == 15
    assert resumed["rate_limit"]["blocked"] >= 1
    assert model_calls.count("What is the current savings interest rate at VinBank?") == 1
    assert monitor.error_requests == 0
    assert monitor.total_requests == len(audit.logs)
    assert len({row["request_id"] for row in audit.logs}) == len(audit.logs)

    results_path = tmp_path / "results.json"
    jsonschema.validate(
        json.loads(results_path.read_text(encoding="utf-8")),
        json.loads((ROOT / "schemas" / "results.schema.json").read_text(encoding="utf-8")),
    )


def test_checkpoint_fingerprint_changes_with_model_and_excludes_client_key():
    rate_limiter = RateLimitPlugin()
    groups = {"safe_queries": ["safe banking question"]}
    runner = SimpleNamespace(
        provider="openrouter",
        model="liquid/lfm-3-1.7b:free",
        temperature=0.4,
        client_kwargs={
            "api_key": "never-hash-or-persist-this",
            "base_url": "https://openrouter.ai/api/v1",
        },
        plugins=[],
    )

    original = pipeline._suite_fingerprint(runner, rate_limiter, groups)
    runner.model = "different-model"
    changed = pipeline._suite_fingerprint(runner, rate_limiter, groups)

    assert original != changed
    assert len(original) == len(changed) == 64


def test_main_does_not_report_completion_when_checkpoint3_is_incomplete(
    monkeypatch, capsys
):
    async def incomplete_suite():
        return {"complete": False, "status": "INCOMPLETE"}

    monkeypatch.setattr(main_module, "setup_api_key", lambda: None)
    monkeypatch.setattr(main_module, "part3_assignment_suite", incomplete_suite)

    asyncio.run(main_module.main(parts=[3]))

    output = capsys.readouterr().out
    assert "Lab 11 incomplete" in output
    assert "Lab 11 complete!" not in output


def test_main_reports_completion_when_checkpoint3_returns_results(
    monkeypatch, capsys
):
    async def completed_suite():
        return {"safe_queries": []}

    monkeypatch.setattr(main_module, "setup_api_key", lambda: None)
    monkeypatch.setattr(main_module, "part3_assignment_suite", completed_suite)

    asyncio.run(main_module.main(parts=[3]))

    output = capsys.readouterr().out
    assert "Lab 11 complete!" in output
    assert "Lab 11 incomplete" not in output
