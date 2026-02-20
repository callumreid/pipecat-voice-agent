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
from typing import Optional, Sequence

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
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

load_dotenv(".env.local")

SYSTEM_PROMPT = """You are a helpful voice assistant used for testing Coval's trace ingestion.
Keep your responses concise and conversational. You may be asked about anything.
If asked to call a tool you don't have, politely explain you don't have that capability."""

COVAL_TRACES_ENDPOINT = "https://api.coval.dev/v1/traces"


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
    """Sends spans as OTLP JSON via plain HTTP requests. Avoids protobuf binary
    encoding issues with API Gateway binary media type configuration."""

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
    """

    def __init__(self, api_key: str, endpoint: str = COVAL_TRACES_ENDPOINT, timeout: int = 30):
        self._api_key = api_key
        self._endpoint = endpoint
        self._timeout = timeout
        self._inner: Optional[OTLPSpanExporter] = None
        self._buffer: list[ReadableSpan] = []

    def set_simulation_id(self, simulation_id: str) -> None:
        self._simulation_id = simulation_id
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
            result = self._inner.export(spans)
            if result != SpanExportResult.SUCCESS:
                logger.error(f"Coval OTLP export failed for {len(spans)} spans: {result}")
            else:
                logger.debug(f"Exported {len(spans)} spans to Coval")
            return result
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


async def run_agent(room_url: str, token: str | None = None):
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

    # Configure tracing before the pipeline starts so Pipecat's conversation span
    # is created against this provider from the start.
    # Use SimpleSpanProcessor (synchronous) to avoid silent BatchSpanProcessor drop issues.
    api_key = os.getenv("COVAL_API_KEY")
    coval_exporter: Optional[DynamicCovalExporter] = None
    if api_key:
        coval_exporter = DynamicCovalExporter(api_key=api_key)
        resource = Resource.create({SERVICE_NAME: "pipecat-voice-agent"})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(SimpleSpanProcessor(coval_exporter))
        otel_trace.set_tracer_provider(provider)
        logger.info("Coval tracing initialized — waiting for simulation ID via SIP header")
    else:
        logger.warning("COVAL_API_KEY not set — tracing disabled")

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

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=os.getenv("CARTESIA_VOICE_ID", "79a125e8-cd45-4c13-8a67-188112f4dd22"),
    )
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = OpenAILLMContext(messages)
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

    # Flush all pending spans to Coval before the process exits.
    # BatchSpanProcessor exports asynchronously — without this, the final
    # batch (including the conversation root span) would be dropped.
    otel_trace.get_tracer_provider().shutdown()


async def main():
    room_url = os.getenv("DAILY_ROOM_URL")
    token = os.getenv("DAILY_TOKEN")

    if not room_url:
        raise ValueError("DAILY_ROOM_URL must be set")

    await run_agent(room_url, token)


if __name__ == "__main__":
    asyncio.run(main())
