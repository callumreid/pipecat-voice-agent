"""Shared Coval trace instrumentation helpers for the Pipecat example agent."""

from __future__ import annotations

from typing import Any, Optional

from opentelemetry import trace as otel_trace

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.tracing.service_decorators import traced_stt


def _read_path(value: Any, *path: Any) -> Any:
    """Read nested data from objects, dicts, and lists."""
    current = value
    for segment in path:
        if current is None:
            return None
        if isinstance(segment, int):
            if isinstance(current, (list, tuple)) and 0 <= segment < len(current):
                current = current[segment]
            else:
                return None
            continue
        if isinstance(current, dict):
            current = current.get(segment)
        else:
            current = getattr(current, segment, None)
    return current


def _set_current_span_attribute(key: str, value: Any) -> None:
    """Attach an attribute to the current span when tracing is active."""
    span = otel_trace.get_current_span()
    if span.is_recording():
        span.set_attribute(key, value)


def extract_stt_confidence(result: Any) -> Optional[float]:
    """Extract the best-alternative confidence from a Deepgram STT result."""
    confidence = _read_path(result, "channel", "alternatives", 0, "confidence")
    if confidence is None:
        return None
    try:
        normalized = float(confidence)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= normalized <= 1.0:
        return None
    return round(normalized, 4)


class CovalDeepgramSTTService(DeepgramSTTService):
    """Deepgram STT service that enriches Pipecat's built-in `stt` span."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_stt_confidence: Optional[float] = None

    async def _on_message(self, message):
        is_final = bool(getattr(message, "is_final", False))
        self._current_stt_confidence = extract_stt_confidence(message) if is_final else None
        try:
            await super()._on_message(message)
        finally:
            if is_final:
                self._current_stt_confidence = None

    @traced_stt
    async def _handle_transcription(self, transcript: str, is_final: bool, language=None):
        if is_final and self._current_stt_confidence is not None:
            _set_current_span_attribute("stt.confidence", self._current_stt_confidence)


class _FinishReasonTrackingStream:
    """Wrap an OpenAI streaming response and attach finish_reason to the current span."""

    def __init__(self, stream: Any):
        self._stream = stream
        self._iter = stream.__aiter__()

    def __aiter__(self):
        return self

    async def __anext__(self):
        chunk = await self._iter.__anext__()
        finish_reason = _read_path(chunk, "choices", 0, "finish_reason")
        if finish_reason is not None:
            normalized = str(finish_reason).strip()
            if normalized:
                _set_current_span_attribute("llm.finish_reason", normalized)
        return chunk

    async def aclose(self):
        if hasattr(self._iter, "aclose"):
            await self._iter.aclose()
        elif hasattr(self._stream, "aclose"):
            await self._stream.aclose()

    async def close(self):
        if hasattr(self._stream, "close"):
            await self._stream.close()
        else:
            await self.aclose()


class CovalOpenAILLMService(OpenAILLMService):
    """OpenAI service wrapper that enriches Pipecat's built-in `llm` span."""

    async def get_chat_completions(self, params_from_context):
        stream = await super().get_chat_completions(params_from_context)
        return _FinishReasonTrackingStream(stream)
