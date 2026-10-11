"""Tests for ragdoc.llm: the shared LLM reliability layer.

Covers the retry policy (what is retried, what is not, backoff shape), refusal handling,
client resolution, and the structural client protocols. The client is always a mock whose
``chat.completions.parse`` is an ``AsyncMock``; sleeps are recorded by patching the module's
``_sleep`` so no test ever waits.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import BaseModel, ValidationError

openai = pytest.importorskip("openai", reason="llm extra not installed")

import ragdoc.llm
from ragdoc.config import RagdocConfig, configure
from ragdoc.llm import (
    ChatClient,
    EmbeddingsClient,
    LLMClient,
    LLMError,
    LLMNotConfiguredError,
    LLMRefusalError,
    call_structured,
    is_llm_failure,
    is_size_related_failure,
    resolve_openai_client,
    retry_llm,
    without_sdk_retries,
)


class _Out(BaseModel):
    value: str


_OK = _Out(value="ok")


class _StubLLMClient:
    # RagdocConfig validates openai_client via isinstance against the runtime-checkable
    # LLMClient protocol; on Python 3.12+ that uses getattr_static, which a bare
    # MagicMock's dynamic attributes never satisfy — the stub's must be real attributes.
    def __init__(self) -> None:
        self.chat = MagicMock()
        self.embeddings = MagicMock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _response(parsed: _Out | None) -> MagicMock:
    message = MagicMock()
    message.parsed = parsed
    choice = MagicMock()
    choice.message = message
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _client(*, parsed: _Out | None = _OK, side_effect: list | None = None) -> MagicMock:
    client = MagicMock()
    if side_effect is not None:
        client.chat.completions.parse = AsyncMock(side_effect=side_effect)
    else:
        client.chat.completions.parse = AsyncMock(return_value=_response(parsed))
    return client


def _httpx_response(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, headers=headers, request=httpx.Request("POST", "https://api.test/v1"))


def _rate_limit_error(headers: dict[str, str] | None = None) -> openai.RateLimitError:
    return openai.RateLimitError("rate limited", response=_httpx_response(429, headers), body=None)


def _server_error() -> openai.InternalServerError:
    return openai.InternalServerError("boom", response=_httpx_response(500), body=None)


def _bad_request_error() -> openai.BadRequestError:
    return openai.BadRequestError("bad request", response=_httpx_response(400), body=None)


def _auth_error() -> openai.AuthenticationError:
    return openai.AuthenticationError("bad key", response=_httpx_response(401), body=None)


def _not_found_error() -> openai.NotFoundError:
    return openai.NotFoundError("no such model", response=_httpx_response(404), body=None)


def _connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx.Request("POST", "https://api.test/v1"))


def _timeout_error() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=httpx.Request("POST", "https://api.test/v1"))


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record retry sleeps instead of actually sleeping."""
    recorded: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(ragdoc.llm, "_sleep", _fake_sleep)
    return recorded


# ---------------------------------------------------------------------------
# call_structured: success + refusal
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_returns_parsed_on_success():
    client = _client()
    result = await call_structured(client, model="m", messages=[], response_format=_Out)
    assert result is _OK


@pytest.mark.anyio
async def test_parsed_none_raises_refusal():
    client = _client(parsed=None)
    with pytest.raises(LLMRefusalError):
        await call_structured(client, model="m", messages=[], response_format=_Out)


@pytest.mark.anyio
async def test_refusal_not_retried(sleeps: list[float]):
    client = _client(parsed=None)
    with pytest.raises(LLMRefusalError):
        await call_structured(client, model="m", messages=[], response_format=_Out, max_retries=3)
    assert client.chat.completions.parse.call_count == 1
    assert sleeps == []


@pytest.mark.anyio
async def test_empty_choices_raises_refusal():
    resp = MagicMock()
    resp.choices = []
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(return_value=resp)
    with pytest.raises(LLMRefusalError):
        await call_structured(client, model="m", messages=[], response_format=_Out)


def test_refusal_is_llm_error():
    assert issubclass(LLMRefusalError, LLMError)
    assert issubclass(LLMNotConfiguredError, LLMError)


# ---------------------------------------------------------------------------
# call_structured: retry policy
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_rate_limit_retried_then_succeeds(sleeps: list[float]):
    client = _client(side_effect=[_rate_limit_error(), _rate_limit_error(), _response(_OK)])
    result = await call_structured(client, model="m", messages=[], response_format=_Out, max_retries=2)
    assert result is _OK
    assert client.chat.completions.parse.call_count == 3
    assert len(sleeps) == 2


@pytest.mark.anyio
async def test_rate_limit_honors_retry_after_header(sleeps: list[float]):
    client = _client(side_effect=[_rate_limit_error(headers={"Retry-After": "7"}), _response(_OK)])
    result = await call_structured(client, model="m", messages=[], response_format=_Out)
    assert result is _OK
    assert sleeps[0] == 7.0


@pytest.mark.anyio
async def test_retry_after_is_capped_at_backoff_max(sleeps: list[float]):
    """``Retry-After: 86400`` must not park the pipeline for a day: the wait is capped at backoff_max."""

    async def _always_rate_limited() -> None:
        raise _rate_limit_error(headers={"Retry-After": "86400"})

    with pytest.raises(openai.RateLimitError):
        await retry_llm(_always_rate_limited, max_retries=2, backoff_max=30.0)
    assert sleeps == [30.0, 30.0]


@pytest.mark.anyio
async def test_retry_after_below_cap_is_honored_exactly(sleeps: list[float]):
    client = _client(side_effect=[_rate_limit_error(headers={"Retry-After": "29.5"}), _response(_OK)])
    await call_structured(client, model="m", messages=[], response_format=_Out, backoff_max=30.0)
    assert sleeps == [29.5]


@pytest.mark.anyio
async def test_rate_limit_unparseable_retry_after_falls_back_to_backoff(sleeps: list[float]):
    client = _client(side_effect=[_rate_limit_error(headers={"Retry-After": "soon"}), _response(_OK)])
    await call_structured(client, model="m", messages=[], response_format=_Out, backoff_base=1.0)
    assert 0.5 <= sleeps[0] <= 1.5  # full-jitter around base * 2**0


@pytest.mark.anyio
async def test_connection_error_retried_with_exponential_jitter(sleeps: list[float]):
    client = _client(
        side_effect=[_connection_error(), _connection_error(), _connection_error(), _response(_OK)],
    )
    result = await call_structured(
        client, model="m", messages=[], response_format=_Out, max_retries=3, backoff_base=1.0, backoff_max=30.0
    )
    assert result is _OK
    assert len(sleeps) == 3
    for attempt, delay in enumerate(sleeps):
        expected = min(30.0, 1.0 * 2**attempt)
        assert expected * 0.5 <= delay <= expected * 1.5


@pytest.mark.anyio
async def test_backoff_capped_at_backoff_max(sleeps: list[float]):
    client = _client(side_effect=[_connection_error()] * 4 + [_response(_OK)])
    await call_structured(
        client, model="m", messages=[], response_format=_Out, max_retries=4, backoff_base=10.0, backoff_max=2.0
    )
    assert all(delay <= 2.0 * 1.5 for delay in sleeps)


@pytest.mark.anyio
async def test_timeout_error_is_retryable(sleeps: list[float]):
    client = _client(side_effect=[_timeout_error(), _response(_OK)])
    result = await call_structured(client, model="m", messages=[], response_format=_Out)
    assert result is _OK
    assert client.chat.completions.parse.call_count == 2


@pytest.mark.anyio
async def test_server_error_500_retried(sleeps: list[float]):
    client = _client(side_effect=[_server_error(), _response(_OK)])
    result = await call_structured(client, model="m", messages=[], response_format=_Out)
    assert result is _OK
    assert client.chat.completions.parse.call_count == 2


@pytest.mark.anyio
async def test_bad_request_400_not_retried(sleeps: list[float]):
    client = _client(side_effect=[_bad_request_error()])
    with pytest.raises(openai.BadRequestError):
        await call_structured(client, model="m", messages=[], response_format=_Out, max_retries=3)
    assert client.chat.completions.parse.call_count == 1
    assert sleeps == []


@pytest.mark.anyio
async def test_auth_401_not_retried(sleeps: list[float]):
    client = _client(side_effect=[_auth_error()])
    with pytest.raises(openai.AuthenticationError):
        await call_structured(client, model="m", messages=[], response_format=_Out, max_retries=3)
    assert client.chat.completions.parse.call_count == 1


@pytest.mark.anyio
async def test_not_found_404_not_retried(sleeps: list[float]):
    client = _client(side_effect=[_not_found_error()])
    with pytest.raises(openai.NotFoundError):
        await call_structured(client, model="m", messages=[], response_format=_Out, max_retries=3)
    assert client.chat.completions.parse.call_count == 1


@pytest.mark.anyio
async def test_non_openai_error_not_retried(sleeps: list[float]):
    client = _client(side_effect=[RuntimeError("api down")])
    with pytest.raises(RuntimeError, match="api down"):
        await call_structured(client, model="m", messages=[], response_format=_Out, max_retries=3)
    assert client.chat.completions.parse.call_count == 1


@pytest.mark.anyio
async def test_exhausted_retries_reraises_last_error(sleeps: list[float]):
    last = _rate_limit_error()
    client = _client(side_effect=[_rate_limit_error(), _rate_limit_error(), last])
    with pytest.raises(openai.RateLimitError) as excinfo:
        await call_structured(client, model="m", messages=[], response_format=_Out, max_retries=2)
    assert excinfo.value is last
    assert client.chat.completions.parse.call_count == 3


# ---------------------------------------------------------------------------
# call_structured: kwargs forwarding + namespace
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_timeout_param_forwarded():
    client = _client()
    await call_structured(client, model="m", messages=[], response_format=_Out, timeout=5.0)
    assert client.chat.completions.parse.call_args.kwargs["timeout"] == 5.0

    client2 = _client()
    await call_structured(client2, model="m", messages=[], response_format=_Out, timeout=None)
    assert "timeout" not in client2.chat.completions.parse.call_args.kwargs


@pytest.mark.anyio
async def test_temperature_and_model_forwarded():
    client = _client()
    await call_structured(client, model="my-model", messages=[], response_format=_Out, temperature=0.3)
    kwargs = client.chat.completions.parse.call_args.kwargs
    assert kwargs["model"] == "my-model"
    assert kwargs["temperature"] == 0.3
    assert kwargs["response_format"] is _Out


@pytest.mark.anyio
async def test_non_beta_namespace_used():
    client = _client()
    await call_structured(client, model="m", messages=[], response_format=_Out)
    client.chat.completions.parse.assert_awaited_once()
    deprecated_namespace = client.beta  # the pre-v2 structured-output namespace
    deprecated_namespace.chat.completions.parse.assert_not_called()


# ---------------------------------------------------------------------------
# One retry layer: the SDK's own retries must not stack under retry_llm
# ---------------------------------------------------------------------------


def _completion_json(content: str) -> dict[str, object]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content, "refusal": None}}
        ],
    }


def _sdk_client(handler, *, max_retries: int = 2) -> openai.AsyncOpenAI:  # type: ignore[no-untyped-def]
    """A REAL ``AsyncOpenAI`` over a fake HTTP transport — no network, but the SDK's retry code runs."""
    return openai.AsyncOpenAI(
        api_key="test",
        base_url="https://api.test/v1",
        max_retries=max_retries,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


@pytest.mark.anyio
async def test_sdk_retries_do_not_stack_under_library_retries(sleeps: list[float]):
    """A persistent 429 costs max_retries + 1 HTTP attempts — not (max_retries + 1) x (SDK retries + 1)."""
    requests: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        # retry-after-ms keeps the SDK's own (pre-fix) retry sleeps negligible.
        return httpx.Response(429, headers={"retry-after-ms": "1"}, json={"error": {"message": "quota"}})

    client = _sdk_client(_handler, max_retries=2)
    with pytest.raises(openai.RateLimitError):
        await call_structured(client, model="m", messages=[], response_format=_Out, max_retries=2)

    assert len(requests) == 3
    assert len(sleeps) == 2
    assert client.max_retries == 2  # the caller's client is never mutated


@pytest.mark.anyio
async def test_sdk_client_view_still_parses_and_forwards_timeout(sleeps: list[float]):
    """The retry-free view is a working client: 429 then a real structured completion parses."""
    seen_timeouts: list[object] = []
    responses = iter(
        [
            httpx.Response(429, json={"error": {"message": "quota"}}),
            httpx.Response(200, json=_completion_json('{"value": "ok"}')),
        ]
    )

    def _handler(request: httpx.Request) -> httpx.Response:
        seen_timeouts.append(request.extensions["timeout"]["read"])
        return next(responses)

    result = await call_structured(
        _sdk_client(_handler),
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        response_format=_Out,
        timeout=12.5,
    )
    assert result == _Out(value="ok")
    assert seen_timeouts == [12.5, 12.5]


def test_without_sdk_retries_leaves_custom_clients_untouched():
    custom = MagicMock()
    assert without_sdk_retries(custom) is custom
    custom.with_options.assert_not_called()

    sdk = openai.AsyncOpenAI(api_key="test", max_retries=5)
    view = without_sdk_retries(sdk)
    assert view is not sdk
    assert view.max_retries == 0
    assert sdk.max_retries == 5
    assert view.timeout == sdk.timeout
    assert view._client is sdk._client  # same connection pool — not a second HTTP client


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


def _context_length_error() -> openai.BadRequestError:
    body = {"message": "This model's maximum context length is 128000 tokens.", "code": "context_length_exceeded"}
    return openai.BadRequestError("too long", response=_httpx_response(400), body=body)


def _gateway_context_error() -> openai.BadRequestError:
    # An OpenAI-compatible gateway that sets no ``code`` — matched on the message.
    return openai.BadRequestError(
        "Requested 200000 tokens exceeds the context window", response=_httpx_response(400), body=None
    )


def _length_finish_error() -> openai.LengthFinishReasonError:
    return openai.LengthFinishReasonError(completion=MagicMock())


def _validation_error() -> ValidationError:
    try:
        _Out.model_validate_json('{"value": ')
    except ValidationError as exc:
        return exc
    raise AssertionError("unreachable")


def _json_error() -> json.JSONDecodeError:
    return json.JSONDecodeError("bad", "{", 1)


@pytest.mark.parametrize(
    "make_exc",
    [
        _context_length_error,
        _gateway_context_error,
        _length_finish_error,
        _validation_error,
        _json_error,
        _timeout_error,
    ],
)
def test_size_related_failures(make_exc):  # type: ignore[no-untyped-def]
    assert is_size_related_failure(make_exc())


@pytest.mark.parametrize(
    "make_exc",
    [
        _auth_error,
        _rate_limit_error,
        _connection_error,
        _server_error,
        _not_found_error,
        _bad_request_error,  # a 400 that is not about length (e.g. an invalid image)
        lambda: LLMRefusalError("refused"),
        lambda: RuntimeError("boom"),
        lambda: ValueError("not a parse failure"),
    ],
)
def test_non_size_related_failures(make_exc):  # type: ignore[no-untyped-def]
    assert not is_size_related_failure(make_exc())


def test_payload_too_large_413_is_size_related():
    exc = openai.APIStatusError("too large", response=_httpx_response(413), body=None)
    assert is_size_related_failure(exc)


@pytest.mark.parametrize(
    "make_exc",
    [
        _auth_error,
        _bad_request_error,
        _rate_limit_error,
        _connection_error,
        _timeout_error,
        _length_finish_error,
        _validation_error,
        lambda: LLMRefusalError("refused"),
    ],
)
def test_llm_failures_are_recognised(make_exc):  # type: ignore[no-untyped-def]
    assert is_llm_failure(make_exc())


@pytest.mark.parametrize("exc", [TypeError("bug"), AttributeError("bug"), KeyError("bug"), RuntimeError("boom")])
def test_programming_errors_are_not_llm_failures(exc: Exception):
    assert not is_llm_failure(exc)


def test_cancellation_is_not_an_llm_failure():
    import asyncio

    assert not is_llm_failure(asyncio.CancelledError())
    assert not is_size_related_failure(asyncio.CancelledError())


# ---------------------------------------------------------------------------
# resolve_openai_client
# ---------------------------------------------------------------------------


def test_resolve_client_explicit_wins():
    explicit = MagicMock()
    with configure(RagdocConfig(openai_client=_StubLLMClient())):
        assert resolve_openai_client(explicit) is explicit


def test_resolve_client_falls_back_to_config():
    configured = _StubLLMClient()
    with configure(RagdocConfig(openai_client=configured)):
        assert resolve_openai_client(None) is configured


def test_resolve_client_raises_llm_not_configured():
    with configure(RagdocConfig()), pytest.raises(LLMNotConfiguredError) as excinfo:
        resolve_openai_client(None)
    # The error must name both remedies.
    message = str(excinfo.value)
    assert "client=" in message
    assert "configure" in message
    assert "openai_client" in message


def test_config_accepts_chat_only_client_and_resolves_it():
    """A chat-only gateway client is a valid configured client: chat consumers need nothing more."""
    chat_only = _ChatOnlyStub()
    with configure(RagdocConfig(openai_client=chat_only)):
        assert resolve_openai_client(None) is chat_only


def test_config_accepts_embeddings_only_client_but_chat_resolution_fails_loud():
    """An embeddings-only client may be configured, but a chat consumer must not receive it."""
    with configure(RagdocConfig(openai_client=_EmbeddingsOnlyStub())), pytest.raises(LLMNotConfiguredError) as excinfo:
        resolve_openai_client(None)
    assert "chat" in str(excinfo.value)


def test_config_rejects_client_with_neither_capability():
    with pytest.raises(ValidationError):
        RagdocConfig(openai_client=object())  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_custom_client_error_surfaces_when_openai_unavailable(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
):
    """Without openai installed, a custom structural client's exception must surface unchanged.

    Regression: ``_classify`` used to call ``_openai_errors()`` unconditionally, replacing the
    real transport error from a custom ``ChatClient`` with an ImportError on base installs.
    """

    class _GatewayDown(ConnectionError):
        pass

    def _no_openai() -> ragdoc.llm._OpenAIErrorTypes:
        raise ImportError("LLM features require the 'llm' extra: pip install 'ragdoc[llm]'")

    monkeypatch.setattr(ragdoc.llm, "_openai_errors", _no_openai)
    client = _client(side_effect=[_GatewayDown("gateway down")])
    with pytest.raises(_GatewayDown, match="gateway down"):
        await call_structured(client, model="m", messages=[], response_format=_Out, max_retries=3)
    assert client.chat.completions.parse.call_count == 1  # non-retryable without openai's types
    assert sleeps == []


# ---------------------------------------------------------------------------
# retry_llm: generic engine (shared with embeddings)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_retry_llm_generic_over_embeddings(sleeps: list[float]):
    embeddings_response = MagicMock()
    create = AsyncMock(side_effect=[_rate_limit_error(), embeddings_response])

    async def _call() -> MagicMock:
        return await create()

    result = await retry_llm(_call, max_retries=2, log_prefix="OpenAIEmbedder")
    assert result is embeddings_response
    assert create.call_count == 2
    assert len(sleeps) == 1


@pytest.mark.anyio
async def test_retry_llm_returns_first_success():
    async def _call() -> str:
        return "done"

    assert await retry_llm(_call) == "done"


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class _ChatOnlyStub:
    chat = MagicMock()


class _EmbeddingsOnlyStub:
    embeddings = MagicMock()


class _FullStub:
    chat = MagicMock()
    embeddings = MagicMock()


def test_llmclient_protocol_runtime_checkable():
    assert isinstance(_FullStub(), LLMClient)
    assert isinstance(_ChatOnlyStub(), ChatClient)
    assert isinstance(_EmbeddingsOnlyStub(), EmbeddingsClient)
    assert not isinstance(_ChatOnlyStub(), LLMClient)  # missing embeddings
    assert not isinstance(_EmbeddingsOnlyStub(), ChatClient)
    assert not isinstance(object(), ChatClient)


def test_real_async_openai_type_checks_against_protocols():
    """A real ``openai.AsyncOpenAI`` must satisfy the structural protocols *statically*.

    Runs basedpyright on ``tests/typecheck_openai_clients.py`` (plain assignments of SDK clients
    to the protocols). Runtime ``isinstance`` only checks attribute presence, so only a type
    checker can catch signature drift between the protocols and the SDK.
    """
    pytest.importorskip("basedpyright", reason="basedpyright not installed")
    check_file = Path(__file__).parent / "typecheck_openai_clients.py"
    proc = subprocess.run(
        # --pythonpath pins import resolution to this interpreter; otherwise basedpyright
        # discovers one from the environment, which in CI is not the tox venv with openai.
        [sys.executable, "-m", "basedpyright", "--pythonpath", sys.executable, str(check_file)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# Exit criteria (spec case 31): no untyped clients, no deprecated namespace
# ---------------------------------------------------------------------------


def test_no_untyped_clients_or_beta_namespace_in_src():
    """Meta-test for the Phase 7 exit criteria, so regressions fail loudly in CI."""
    src = Path(__file__).parent.parent / "src" / "ragdoc"
    offenders: list[str] = []
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in ("client: Any", "beta.chat.completions"):
            if needle in text:
                offenders.append(f"{path.relative_to(src)}: {needle}")
    assert offenders == [], f"Phase 7 exit criteria violated: {offenders}"
