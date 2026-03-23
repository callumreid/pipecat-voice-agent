"""
Pipecat voice agent with OpenTelemetry tracing for Coval simulation testing.

Cloud: async def bot(args) is called by Pipecat Cloud base image per session.
       args.room_url and args.token are injected by the platform.
Local: python bot.py — reads DAILY_ROOM_URL / DAILY_TOKEN from .env.local.

Tracing: DynamicCovalExporter buffers spans until simulation_id is known.
         Primary: X-Coval-Simulation-Id SIP header via on_dialin_connected event.
         Fallback: COVAL_SIMULATION_ID env var (set at startup for local testing).
"""

import asyncio
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Sequence

import requests
from dotenv import load_dotenv
from loguru import logger

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.daily.transport import DailyDialinSettings, DailyParams, DailyTransport

load_dotenv(override=True)

SYSTEM_PROMPT = """You are a helpful voice assistant used for testing Coval's trace ingestion.
Keep your responses concise and conversational. You have access to tools — use them when relevant:
- get_current_time: returns the current date and time
- get_weather: returns mock weather for a city
- search_web: searches the web for up-to-date information on any topic
- lookup_order_status: looks up a mock order by order ID"""

COVAL_TRACES_ENDPOINT = "https://api.coval.dev/v1/traces"


# ── Tracing ────────────────────────────────────────────────────────────────────

def _span_to_otlp_json(span: ReadableSpan) -> dict:
    """Convert a ReadableSpan to OTLP JSON format (resourceSpans structure)."""

    def attrs(attributes) -> list:
        if not attributes:
            return []
        result = []
        for k, v in attributes.items():
            if isinstance(v, bool):
                result.append({"key": k, "value": {"boolValue": v}})
            elif isinstance(v, int):
                result.append({"key": k, "value": {"intValue": v}})
            elif isinstance(v, float):
                result.append({"key": k, "value": {"doubleValue": v}})
            else:
                result.append({"key": k, "value": {"stringValue": str(v)}})
        return result

    def hex_id(id_int: int, length: int) -> str:
        return format(id_int, f"0{length * 2}x") if id_int else ""

    context = span.context
    span_dict = {
        "traceId": hex_id(context.trace_id, 16) if context else "",
        "spanId": hex_id(context.span_id, 8) if context else "",
        "parentSpanId": hex_id(span.parent.span_id, 8) if span.parent else "",
        "name": span.name,
        "kind": span.kind.value,
        "startTimeUnixNano": str(span.start_time) if span.start_time else "0",
        "endTimeUnixNano": str(span.end_time) if span.end_time else "0",
        "attributes": attrs(span.attributes),
        "status": {"code": span.status.status_code.value, "message": span.status.description or ""},
        "events": [],
        "links": [],
    }

    resource_attrs = attrs(span.resource.attributes) if span.resource else []
    return {
        "resourceSpans": [{
            "resource": {"attributes": resource_attrs},
            "scopeSpans": [{
                "scope": {"name": span.instrumentation_scope.name if span.instrumentation_scope else ""},
                "spans": [span_dict],
            }],
        }]
    }


class _CovalJSONExporter(SpanExporter):
    """Sends spans as OTLP JSON via plain HTTP. Avoids protobuf binary encoding
    issues with API Gateway binary media type configuration."""

    def __init__(self, api_key: str, simulation_id: str, endpoint: str, timeout: int):
        self._api_key = api_key
        self._simulation_id = simulation_id
        self._endpoint = endpoint
        self._timeout = timeout

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            try:
                payload = _span_to_otlp_json(span)
                resp = requests.post(
                    self._endpoint,
                    json=payload,
                    headers={
                        "x-api-key": self._api_key,
                        "X-Simulation-Id": self._simulation_id,
                    },
                    timeout=self._timeout,
                )
                if not resp.ok:
                    logger.error(f"Coval trace export failed {resp.status_code}: {resp.text}")
                    return SpanExportResult.FAILURE
                else:
                    logger.debug(f"Exported span '{span.name}' → {resp.status_code}")
            except Exception as error:
                logger.error(f"Coval trace export exception: {error}")
                return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def shutdown(self) -> None:
        pass


class DynamicCovalExporter(SpanExporter):
    """OTLP span exporter that buffers spans until the Coval simulation ID is known.

    Configured before the pipeline starts so Pipecat's conversation root span is
    captured from the start. When set_simulation_id() is called (on SIP dialin),
    all buffered spans are flushed to Coval and subsequent spans export normally.

    reset() clears state between sessions on the same warm process instance.
    """

    def __init__(self, api_key: str, endpoint: str = COVAL_TRACES_ENDPOINT, timeout: int = 30):
        self._api_key = api_key
        self._endpoint = endpoint
        self._timeout = timeout
        self._inner: Optional[_CovalJSONExporter] = None
        self._buffer: list[ReadableSpan] = []

    def reset(self) -> None:
        """Clear state for a new session (called at the start of each bot() invocation)."""
        self._inner = None
        self._buffer.clear()

    def set_simulation_id(self, simulation_id: str) -> None:
        self._inner = _CovalJSONExporter(
            api_key=self._api_key,
            simulation_id=simulation_id,
            endpoint=self._endpoint,
            timeout=self._timeout,
        )
        if self._buffer:
            logger.info(f"Flushing {len(self._buffer)} buffered spans to Coval")
            result = self._inner.export(self._buffer)
            logger.info(f"Buffered span flush result: {result}")
            self._buffer.clear()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if self._inner:
            return self._inner.export(spans)
        logger.debug(f"Buffering {len(spans)} spans (no simulation_id yet)")
        self._buffer.extend(spans)
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        if self._inner:
            return self._inner.force_flush(timeout_millis)
        return True

    def shutdown(self) -> None:
        if self._inner:
            self._inner.shutdown()


# Tracing is set up once per process (Pipecat Cloud reuses warm instances across calls).
# reset() is called at the start of each bot() invocation to clear per-session state.
_coval_exporter: Optional[DynamicCovalExporter] = None


def _init_tracing() -> None:
    global _coval_exporter
    api_key = os.getenv("COVAL_API_KEY")
    if not api_key:
        logger.warning("COVAL_API_KEY not set — tracing disabled")
        return
    _coval_exporter = DynamicCovalExporter(api_key=api_key)
    resource = Resource.create({SERVICE_NAME: "pipecat-voice-agent"})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(_coval_exporter))
    otel_trace.set_tracer_provider(provider)
    logger.info("Coval tracing initialized")


_init_tracing()


# ── Coval OTel span processors ─────────────────────────────────────────────────


class STTSpanProcessor(FrameProcessor):
    """Emits Coval-standard 'stt' spans for each final STT transcription.

    Span attributes:
      stt.transcription  — the transcribed text (required for STT WER metric)
      metrics.ttfb       — seconds from user started speaking to first result
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tracer = otel_trace.get_tracer("coval.stt")
        self._speech_start_time: Optional[float] = None
        self._first_result_time: Optional[float] = None

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._speech_start_time = time.time()
            self._first_result_time = None

        elif isinstance(frame, InterimTranscriptionFrame):
            if self._first_result_time is None and self._speech_start_time:
                self._first_result_time = time.time()

        elif isinstance(frame, TranscriptionFrame) and frame.text and frame.text.strip():
            now = time.time()
            if self._first_result_time is None:
                self._first_result_time = now

            ttfb = 0.0
            if self._speech_start_time:
                ttfb = self._first_result_time - self._speech_start_time

            with self._tracer.start_as_current_span("stt") as span:
                span.set_attribute("stt.transcription", frame.text)
                span.set_attribute("metrics.ttfb", round(ttfb, 4))

            self._speech_start_time = None
            self._first_result_time = None

        await self.push_frame(frame, direction)


class _LLMTiming:
    """Shared mutable timing state passed between the two LLM span processors."""

    request_start: Optional[float] = None
    first_token_time: Optional[float] = None


class LLMPreSpanProcessor(FrameProcessor):
    """Records when an LLM context frame is dispatched (placed before the LLM service).

    Works in tandem with LLMPostSpanProcessor (placed after the LLM service) via a
    shared _LLMTiming instance to compute accurate TTFB.
    """

    def __init__(self, timing: _LLMTiming, **kwargs):
        super().__init__(**kwargs)
        self._timing = timing

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMMessagesFrame) and direction == FrameDirection.DOWNSTREAM:
            self._timing.request_start = time.time()
            self._timing.first_token_time = None
        await self.push_frame(frame, direction)


class LLMPostSpanProcessor(FrameProcessor):
    """Emits Coval-standard 'llm' spans once each LLM turn completes (placed after LLM service).

    Span attributes:
      metrics.ttfb  — seconds from context dispatched to first response token
    """

    def __init__(self, timing: _LLMTiming, **kwargs):
        super().__init__(**kwargs)
        self._timing = timing
        self._tracer = otel_trace.get_tracer("coval.llm")

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            if self._timing.request_start is not None and self._timing.first_token_time is None:
                self._timing.first_token_time = time.time()

        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._timing.request_start is not None:
                ttfb = (
                    (self._timing.first_token_time - self._timing.request_start)
                    if self._timing.first_token_time
                    else 0.0
                )
                with self._tracer.start_as_current_span("llm") as span:
                    span.set_attribute("metrics.ttfb", round(ttfb, 4))
                self._timing.request_start = None
                self._timing.first_token_time = None

        await self.push_frame(frame, direction)


class TTSSpanProcessor(FrameProcessor):
    """Emits Coval-standard 'tts' spans around each TTS synthesis.

    Span attributes:
      metrics.ttfb  — seconds from text sent to first audio byte
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tracer = otel_trace.get_tracer("coval.tts")
        self._request_start: Optional[float] = None
        self._first_audio_time: Optional[float] = None

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            # TTS service is about to start synthesising — record request time.
            self._request_start = time.time()
            self._first_audio_time = None

        elif isinstance(frame, TTSStoppedFrame):
            # TTS synthesis complete — emit the span.
            if self._request_start is not None:
                ttfb = (
                    (self._first_audio_time - self._request_start)
                    if self._first_audio_time
                    else 0.0
                )
                with self._tracer.start_as_current_span("tts") as span:
                    span.set_attribute("metrics.ttfb", round(ttfb, 4))
                self._request_start = None
                self._first_audio_time = None

        await self.push_frame(frame, direction)


# ── Tools ──────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current date and time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Returns the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city, e.g. 'San Francisco'",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for up-to-date information on any topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (1-5, default 3)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_order_status",
            "description": "Looks up the status of an order by order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to look up, e.g. 'ORD-12345'",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
]

_WEATHER_CONDITIONS = ["sunny", "cloudy", "partly cloudy", "rainy", "windy", "foggy"]
_ORDER_STATUSES = ["processing", "shipped", "out for delivery", "delivered", "delayed"]


async def tool_get_current_time(function_name, tool_call_id, args, llm, context, result_callback):
    now = datetime.now()
    await result_callback({"time": now.strftime("%I:%M %p"), "date": now.strftime("%A, %B %d, %Y")})


async def tool_get_weather(function_name, tool_call_id, args, llm, context, result_callback):
    city = args.get("city", "Unknown")
    await result_callback({
        "city": city,
        "temperature_f": random.randint(45, 95),
        "condition": random.choice(_WEATHER_CONDITIONS),
        "humidity_pct": random.randint(30, 90),
    })


async def tool_search_web(function_name, tool_call_id, args, llm, context, result_callback):
    from duckduckgo_search import DDGS
    query = args.get("query", "")
    max_results = min(int(args.get("max_results", 3)), 5)
    try:
        ddgs = DDGS()
        raw = list(ddgs.text(query, max_results=max_results))
        results = [{"title": r["title"], "url": r["href"], "snippet": r["body"]} for r in raw]
        await result_callback({"query": query, "results": results})
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        await result_callback({"query": query, "error": str(e), "results": []})


async def tool_lookup_order_status(function_name, tool_call_id, args, llm, context, result_callback):
    order_id = args.get("order_id", "UNKNOWN")
    await result_callback({
        "order_id": order_id,
        "status": random.choice(_ORDER_STATUSES),
        "estimated_delivery": "Mar 1, 2026",
        "carrier": random.choice(["UPS", "FedEx", "USPS", "DHL"]),
    })


# ── Bot entry point ────────────────────────────────────────────────────────────

async def bot(args: Any) -> None:
    """
    Called by Pipecat Cloud per session (args.room_url + args.token injected by platform).
    Also callable directly for local dev — pass any object with .room_url / .token.
    """
    logger.info(f"Session started: room={args.room_url}")

    if _coval_exporter:
        _coval_exporter.reset()

    # Extract dialin_settings from body (passed by PCC's pinless dial-in webhook)
    body = getattr(args, "body", None) or {}
    dialin_settings = None
    if isinstance(body, dict):
        raw = body.get("dialin_settings")
        if raw:
            dialin_settings = DailyDialinSettings(
                call_id=raw.get("callId") or raw.get("call_id", ""),
                call_domain=raw.get("callDomain") or raw.get("call_domain", ""),
            )
            logger.info(f"Dial-in session: call_id={dialin_settings.call_id}")

    transport = DailyTransport(
        args.room_url,
        args.token,
        "Pipecat Test Agent",
        DailyParams(
            api_key=os.getenv("DAILY_API_KEY", ""),
            audio_out_enabled=True,
            transcription_enabled=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            dialin_settings=dialin_settings,
        ),
    )

    # For local testing: if COVAL_SIMULATION_ID is set, activate tracing immediately
    # (on_dialin_connected won't fire for non-SIP connections like direct room joins)
    env_simulation_id = os.getenv("COVAL_SIMULATION_ID")
    if env_simulation_id and _coval_exporter:
        _coval_exporter.set_simulation_id(env_simulation_id)
        logger.info(f"Coval tracing active from env var: simulation_id={env_simulation_id}")

    @transport.event_handler("on_dialin_connected")
    async def on_dialin_connected(transport, data):
        """Extract simulation_id from SIP headers on dial-in connections."""
        logger.info(f"Dialin connected — data: {data}")
        simulation_id = None
        sip_headers = data.get("sipHeaders") or data.get("sip_headers") or {}
        if isinstance(sip_headers, dict):
            simulation_id = (
                sip_headers.get("X-Coval-Simulation-Id")
                or sip_headers.get("x-coval-simulation-id")
            )
            if simulation_id:
                logger.info(f"Got simulation_id from SIP header: {simulation_id}")
        if not simulation_id:
            simulation_id = os.getenv("COVAL_SIMULATION_ID") or None
            if simulation_id:
                logger.info(f"Got simulation_id from env var fallback: {simulation_id}")
        if simulation_id and _coval_exporter:
            _coval_exporter.set_simulation_id(simulation_id)
            logger.info(f"Coval tracing active for simulation_id={simulation_id}")
        else:
            logger.warning("No simulation_id — spans will be discarded")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = DeepgramTTSService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

    llm.register_function("get_current_time", tool_get_current_time)
    llm.register_function("get_weather", tool_get_weather)
    llm.register_function("search_web", tool_search_web)
    llm.register_function("lookup_order_status", tool_lookup_order_status)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = OpenAILLMContext(messages, TOOLS)
    context_aggregator = llm.create_context_aggregator(context)

    llm_timing = _LLMTiming()
    pipeline = Pipeline([
        transport.input(),
        stt,
        STTSpanProcessor(),                       # Emits 'stt' spans: stt.transcription + metrics.ttfb
        context_aggregator.user(),
        LLMPreSpanProcessor(timing=llm_timing),   # Records LLM request start time
        llm,
        LLMPostSpanProcessor(timing=llm_timing),  # Emits 'llm' spans: metrics.ttfb
        tts,
        TTSSpanProcessor(),                       # Emits 'tts' spans: metrics.ttfb
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
        enable_tracing=True,
        enable_turn_tracking=True,
    )

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info(f"Participant joined: {participant.get('id')}")
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info(f"Participant left: {participant.get('id')} reason={reason}")
        await task.queue_frames([EndFrame()])

    handle_sigint = getattr(args, "handle_sigint", False)
    runner = PipelineRunner(handle_sigint=handle_sigint)
    await runner.run(task)


# ── Local dev entrypoint ───────────────────────────────────────────────────────

if __name__ == "__main__":
    @dataclass
    class _LocalArgs:
        room_url: str
        token: str = ""
        body: Any = None
        session_id: Optional[str] = None
        handle_sigint: bool = True  # handle Ctrl+C locally

    room_url = os.getenv("DAILY_ROOM_URL", "")
    if not room_url:
        raise SystemExit("DAILY_ROOM_URL not set — add it to .env.local")

    _args = _LocalArgs(room_url=room_url, token=os.getenv("DAILY_TOKEN", ""))
    logger.info(f"Local dev — joining room: {room_url}")

    try:
        asyncio.run(bot(_args))
    finally:
        otel_trace.get_tracer_provider().shutdown()
