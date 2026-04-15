"""
Pipecat voice agent with OpenTelemetry tracing configured to send spans to Coval.

This agent is used for testing Coval's trace ingestion and viewer. It automatically
emits structured spans for each conversation turn, STT, LLM, and TTS operation.

Tracing architecture:
  - DynamicCovalExporter is configured before the pipeline starts, so Pipecat's
    conversation root span is created against the correct TracerProvider from the start.
  - Spans are buffered internally until X-Coval-Simulation-Id arrives via the
    on_dialin_connected SIP event, at which point all buffered spans are flushed
    to Coval and new spans export normally.
  - Falls back to COVAL_SIMULATION_ID env var for manual testing (set it before
    the call connects and it will be used when on_dialin_connected fires).

Environment variables:
  DAILY_ROOM_URL          - Daily room URL to join
  DAILY_TOKEN             - Daily token (optional, for private rooms)
  OPENAI_API_KEY          - OpenAI API key
  DEEPGRAM_API_KEY        - Deepgram API key (STT)
  CARTESIA_API_KEY        - Cartesia API key (TTS)
  COVAL_API_KEY           - Coval organization API key
  COVAL_SIMULATION_ID     - Fallback simulation output ID for manual testing
  OTEL_CONSOLE_EXPORT     - Set to "true" to also print spans to stdout
"""

import asyncio
import os
import random
import threading
import time
from datetime import datetime
from typing import Optional, Sequence

from coval_trace_instrumentation import (
    CovalDeepgramSTTService,
    CovalOpenAILLMService,
)
from duckduckgo_search import DDGS

from dotenv import load_dotenv
from loguru import logger
import json

import requests
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SpanExportResult as OTLPSpanExportResult
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

load_dotenv(".env.local")

SYSTEM_PROMPT = """You are a helpful voice assistant used for testing Coval's trace ingestion.
Keep your responses concise and conversational. You have access to tools — use them when relevant:
- get_current_time: returns the current date and time
- get_weather: returns mock weather for a city
- search_web: searches the web for up-to-date information on any topic
- lookup_order_status: looks up a mock order by order ID"""

COVAL_TRACES_ENDPOINT = "https://api.coval.dev/v1/traces"
COVAL_API_KEYS_JSON = os.environ.get("COVAL_API_KEYS_JSON", "")
COVAL_API_KEYS_FILE = os.environ.get("COVAL_API_KEYS_FILE", "")
COVAL_API_KEYS_REFRESH_SECONDS = max(float(os.environ.get("COVAL_API_KEYS_REFRESH_SECONDS", "30")), 0.0)


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


class _ApiKeyStore:
    """Loads Coval trace API keys from file, JSON, or env vars.

    Resolution order: COVAL_API_KEYS_FILE (auto-refreshed) > COVAL_API_KEYS_JSON >
    COVAL_API_KEY_<LABEL> env vars > legacy COVAL_API_KEY (labeled "default").
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cached_items: list[tuple[str, str]] = []
        self._last_checked_at = 0.0
        self._file_mtime: Optional[float] = None

    def get_items(self) -> list[tuple[str, str]]:
        with self._lock:
            if COVAL_API_KEYS_FILE:
                self._refresh_from_file_if_needed()
                return list(self._cached_items)

            if not self._cached_items:
                self._cached_items = self._load_static_items()
            return list(self._cached_items)

    def _refresh_from_file_if_needed(self) -> None:
        now = time.time()
        if self._cached_items and now - self._last_checked_at < COVAL_API_KEYS_REFRESH_SECONDS:
            return

        self._last_checked_at = now
        try:
            stat = os.stat(COVAL_API_KEYS_FILE)
        except OSError as exc:
            if not self._cached_items:
                logger.warning(f"Unable to read COVAL_API_KEYS_FILE={COVAL_API_KEYS_FILE}: {exc}")
            return

        if self._file_mtime == stat.st_mtime and self._cached_items:
            return

        try:
            with open(COVAL_API_KEYS_FILE, "r", encoding="utf-8") as handle:
                parsed = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Failed to parse COVAL_API_KEYS_FILE={COVAL_API_KEYS_FILE}: {exc}")
            return

        loaded = self._items_from_mapping(parsed)
        if loaded:
            self._cached_items = loaded
            self._file_mtime = stat.st_mtime
            logger.info(f"Loaded {len(loaded)} Coval trace API key(s) from {COVAL_API_KEYS_FILE}")

    def _load_static_items(self) -> list[tuple[str, str]]:
        if COVAL_API_KEYS_JSON:
            try:
                parsed = json.loads(COVAL_API_KEYS_JSON)
            except json.JSONDecodeError as exc:
                logger.warning(f"Failed to parse COVAL_API_KEYS_JSON: {exc}")
            else:
                loaded = self._items_from_mapping(parsed)
                if loaded:
                    return loaded

        env_items: list[tuple[str, str]] = []
        for env_name, raw_value in sorted(os.environ.items()):
            if not env_name.startswith("COVAL_API_KEY_") or env_name.startswith("COVAL_API_KEYS_"):
                continue
            value = raw_value.strip()
            suffix = env_name[len("COVAL_API_KEY_") :].strip()
            if not suffix or not value:
                continue
            env_items.append((suffix.lower().replace("_", "-"), value))

        if env_items:
            return env_items

        api_key = os.getenv("COVAL_API_KEY", "").strip()
        if api_key:
            return [("default", api_key)]

        return []

    def _items_from_mapping(self, mapping: object) -> list[tuple[str, str]]:
        if not isinstance(mapping, dict):
            logger.warning("Ignoring Coval key config because it is not a JSON object")
            return []

        items: list[tuple[str, str]] = []
        for raw_label, raw_value in mapping.items():
            label = str(raw_label).strip()
            value = str(raw_value).strip() if raw_value is not None else ""
            if not label or not value:
                continue
            items.append((label, value))
        return items


_api_key_store = _ApiKeyStore()


def _spans_to_otlp_json(spans: Sequence[ReadableSpan]) -> dict:
    resource_spans = []
    for span in spans:
        resource_spans.extend(_span_to_otlp_json(span)["resourceSpans"])
    return {"resourceSpans": resource_spans}


class _TraceKeyRouter:
    """Selects the correct org-scoped Coval API key per simulation.

    POSTs to /v1/traces with each configured key until one returns 200, then
    caches the winning label per simulation_id so subsequent spans skip the
    fan-out. 401/403/404 are treated as 'wrong org, try next'; 429/5xx are
    transient and bubble up as failure (no key is cached).
    """

    def __init__(self, endpoint: str, timeout: int):
        self._endpoint = endpoint
        self._timeout = timeout
        self._lock = threading.RLock()
        self._selected_label_by_simulation: dict[str, str] = {}

    def has_keys(self) -> bool:
        return bool(_api_key_store.get_items())

    def export(self, spans: Sequence[ReadableSpan], simulation_id: str) -> SpanExportResult:
        payload = _spans_to_otlp_json(spans)
        if not payload["resourceSpans"]:
            return SpanExportResult.SUCCESS
        return SpanExportResult.SUCCESS if self._export_payload(payload, simulation_id) else SpanExportResult.FAILURE

    def _export_payload(self, payload: dict, simulation_id: str) -> bool:
        items = _api_key_store.get_items()
        if not items:
            logger.warning("No Coval trace API keys configured")
            return False

        configured = dict(items)
        cached_label = self._selected_label_by_simulation.get(simulation_id)
        if cached_label and cached_label in configured:
            success, outcome = self._post_payload(payload, simulation_id, cached_label, configured[cached_label])
            if success:
                return True
            if outcome != "mismatch":
                return False
            with self._lock:
                self._selected_label_by_simulation.pop(simulation_id, None)

        for label, api_key in items:
            if label == cached_label:
                continue
            success, outcome = self._post_payload(payload, simulation_id, label, api_key)
            if success:
                with self._lock:
                    self._selected_label_by_simulation[simulation_id] = label
                logger.info(f"Selected Coval trace API key '{label}' for simulation_id={simulation_id}")
                return True
            if outcome == "mismatch":
                continue
            return False

        logger.error(f"No configured Coval trace API key matched simulation_id={simulation_id}")
        return False

    def _post_payload(self, payload: dict, simulation_id: str, label: str, api_key: str) -> tuple[bool, str]:
        try:
            resp = requests.post(
                self._endpoint,
                json=payload,
                headers={"x-api-key": api_key, "X-Simulation-Id": simulation_id},
                timeout=self._timeout,
            )
        except requests.RequestException as error:
            logger.warning(f"Coval trace export exception using key '{label}': {error}")
            return False, "retry"

        if resp.ok:
            return True, "success"
        if resp.status_code in (401, 403, 404):
            return False, "mismatch"
        if resp.status_code == 429 or resp.status_code >= 500:
            logger.warning(f"Retryable Coval trace export failure {resp.status_code} using key '{label}'")
            return False, "retry"
        logger.error(f"Coval trace export failed {resp.status_code} using key '{label}': {resp.text}")
        return False, "fatal"


class DynamicCovalExporter(SpanExporter):
    """OTLP span exporter that buffers spans until the Coval simulation ID is known.

    Configured before the pipeline starts so Pipecat's conversation root span is
    captured from the start. When set_simulation_id() is called (on SIP dialin),
    all buffered spans are flushed via the multi-org router and subsequent spans
    export normally.
    """

    def __init__(self, endpoint: str = COVAL_TRACES_ENDPOINT, timeout: int = 30):
        self._endpoint = endpoint
        self._timeout = timeout
        self._simulation_id: Optional[str] = None
        self._router = _TraceKeyRouter(endpoint=endpoint, timeout=timeout)
        self._buffer: list[ReadableSpan] = []

    def reset(self) -> None:
        """Reset state for a new call in keep-alive mode."""
        self._simulation_id = None
        self._buffer.clear()

    def set_simulation_id(self, simulation_id: str) -> None:
        self._simulation_id = simulation_id
        if self._buffer:
            logger.info(f"Flushing {len(self._buffer)} buffered spans to Coval")
            result = self._router.export(self._buffer, simulation_id)
            logger.info(f"Buffered span flush result: {result}")
            self._buffer.clear()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if self._simulation_id:
            return self._router.export(spans, self._simulation_id)
        logger.debug(f"Buffering {len(spans)} spans (no simulation_id yet)")
        self._buffer.extend(spans)
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def shutdown(self) -> None:
        pass


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
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
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


async def tool_get_current_time(function_name, tool_call_id, args, llm, context, result_callback):
    now = datetime.now()
    await result_callback({"time": now.strftime("%I:%M %p"), "date": now.strftime("%A, %B %d, %Y")})


_WEATHER_CONDITIONS = ["sunny", "cloudy", "partly cloudy", "rainy", "windy", "foggy"]
_ORDER_STATUSES = ["processing", "shipped", "out for delivery", "delivered", "delayed"]


async def tool_get_weather(function_name, tool_call_id, args, llm, context, result_callback):
    city = args.get("city", "Unknown")
    await result_callback({
        "city": city,
        "temperature_f": random.randint(45, 95),
        "condition": random.choice(_WEATHER_CONDITIONS),
        "humidity_pct": random.randint(30, 90),
    })


async def tool_search_web(function_name, tool_call_id, args, llm, context, result_callback):
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
        "estimated_delivery": "Feb 24, 2026",
        "carrier": random.choice(["UPS", "FedEx", "USPS", "DHL"]),
    })


async def run_agent(room_url: str, token: str | None = None, coval_exporter: Optional[DynamicCovalExporter] = None):
    transport = DailyTransport(
        room_url,
        token,
        "Pipecat Test Agent",
        DailyParams(
            audio_out_enabled=True,
            transcription_enabled=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    @transport.event_handler("on_dialin_connected")
    async def on_dialin_connected(transport, data):
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

        if simulation_id and coval_exporter:
            coval_exporter.set_simulation_id(simulation_id)
            logger.info(f"Coval tracing active for simulation_id={simulation_id}")
        else:
            logger.warning("No simulation_id — spans will be discarded")

    stt = CovalDeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=os.getenv("CARTESIA_VOICE_ID", "79a125e8-cd45-4c13-8a67-188112f4dd22"),
    )
    llm = CovalOpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

    llm.register_function("get_current_time", tool_get_current_time)
    llm.register_function("get_weather", tool_get_weather)
    llm.register_function("search_web", tool_search_web)
    llm.register_function("lookup_order_status", tool_lookup_order_status)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = OpenAILLMContext(messages, TOOLS)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

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

    runner = PipelineRunner()
    await runner.run(task)


async def main():
    room_url = os.getenv("DAILY_ROOM_URL")
    token = os.getenv("DAILY_TOKEN")

    if not room_url:
        raise ValueError("DAILY_ROOM_URL must be set")

    # Set up tracing once — reused across restarts.
    # SimpleSpanProcessor is synchronous so spans aren't dropped on restart.
    coval_exporter: Optional[DynamicCovalExporter] = None
    if _api_key_store.get_items():
        coval_exporter = DynamicCovalExporter()
        resource = Resource.create({SERVICE_NAME: "pipecat-voice-agent"})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(SimpleSpanProcessor(coval_exporter))
        otel_trace.set_tracer_provider(provider)
        logger.info("Coval tracing initialized — waiting for simulation ID via SIP header")
    else:
        logger.warning("No Coval trace API keys configured — tracing disabled")

    try:
        while True:
            logger.info("Agent ready — waiting for call...")
            if coval_exporter:
                coval_exporter.reset()
            try:
                await run_agent(room_url, token, coval_exporter)
            except Exception as e:
                logger.error(f"Agent error: {e}")
            logger.info("Call ended. Restarting in 2 seconds...")
            await asyncio.sleep(2)
    finally:
        # Flush remaining spans on clean exit (Ctrl+C, SIGTERM).
        otel_trace.get_tracer_provider().shutdown()


if __name__ == "__main__":
    asyncio.run(main())
