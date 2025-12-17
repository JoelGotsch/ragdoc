"""Tests for ragdoc.llm: the shared LLM reliability layer.

Covers the retry policy (what is retried, what is not, backoff shape), refusal handling,
client resolution, and the structural client protocols. The client is always a mock whose
``chat.completions.parse`` is an ``AsyncMock``; sleeps are recorded by patching the module's
``_sleep`` so no test ever waits.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import BaseModel

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
    resolve_openai_client,
    retry_llm,
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


# ---------------------------------------------------------------------------
# Exit criteria (spec case 31): no untyped clients, no deprecated namespace
# ---------------------------------------------------------------------------


def test_no_untyped_clients_or_beta_namespace_in_src():
    """Meta-test for the Phase 7 exit criteria, so regressions fail loudly in CI."""
    from pathlib import Path

    src = Path(__file__).parent.parent / "src" / "ragdoc"
    offenders: list[str] = []
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in ("client: Any", "beta.chat.completions"):
            if needle in text:
                offenders.append(f"{path.relative_to(src)}: {needle}")
    assert offenders == [], f"Phase 7 exit criteria violated: {offenders}"
