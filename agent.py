"""
Pipecat voice agent for testing Coval's evaluation platform.

Local dev entrypoint: python agent.py
Reads DAILY_ROOM_URL / DAILY_TOKEN from .env.local.
"""

import asyncio
import os
import random
from datetime import datetime
from typing import Optional

from duckduckgo_search import DDGS

from dotenv import load_dotenv
from loguru import logger

from opentelemetry import trace as otel_trace

from pipecat.utils.tracing.service_decorators import traced_stt

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.service_switcher import ServiceSwitcher, ServiceSwitcherStrategyManual
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

try:
    from pipecat.services.google.stt import GoogleSTTService
    _HAS_GOOGLE_STT = True
except ImportError:
    _HAS_GOOGLE_STT = False

from coval_tracing import setup_coval_tracing, set_simulation_id, get_current_llm_span

load_dotenv(".env.local")


# ── Instrumented service subclasses ──────────────────────────────────────────
# These add stt.confidence, llm.finish_reason, and provider sub-spans (TRACE-10).


class InstrumentedDeepgramSTT(DeepgramSTTService):
    """DeepgramSTTService with real ASR confidence and provider sub-spans."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_confidence: float = 0.0

    async def _on_message(self, *args, **kwargs):
        result = kwargs.get("result")
        if result and hasattr(result, "channel"):
            alts = getattr(result.channel, "alternatives", [])
            if alts:
                self._last_confidence = getattr(alts[0], "confidence", 0.0)
        await super()._on_message(*args, **kwargs)

    @traced_stt
    async def _handle_transcription(self, transcript, is_final, language=None):
        """Override to inject stt.confidence and provider sub-spans into the active span."""
        # @traced_stt creates an 'stt' span as current before calling this body
        span = otel_trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("stt.confidence", self._last_confidence)
            # TRACE-10: provider sub-span
            tracer = otel_trace.get_tracer("coval.instrumentation")
            with tracer.start_as_current_span("stt.provider.deepgram") as p:
                p.set_attribute("stt.providerName", "deepgram")
                p.set_attribute("stt.confidence", self._last_confidence)
                ttfb = getattr(getattr(self, "_metrics", None), "ttfb", None)
                if ttfb is not None:
                    p.set_attribute("metrics.ttfb", ttfb)
            with tracer.start_as_current_span("stt.provider_selection") as sel:
                sel.set_attribute("stt.selectedProvider", "deepgram")
                sel.set_attribute("stt.fallbackAvailable", "google" if _HAS_GOOGLE_STT else "none")


class InstrumentedGoogleSTT:
    """Wrapper factory — returns a GoogleSTTService subclass with provider sub-spans."""

    @staticmethod
    def create(**kwargs):
        if not _HAS_GOOGLE_STT:
            return None

        class _InstrumentedGoogleSTT(GoogleSTTService):
            @traced_stt
            async def _handle_transcription(self, transcript, is_final, language=None):
                span = otel_trace.get_current_span()
                if span and span.is_recording():
                    span.set_attribute("stt.confidence", 0.95)  # Google doesn't expose confidence easily
                    tracer = otel_trace.get_tracer("coval.instrumentation")
                    with tracer.start_as_current_span("stt.provider.google") as p:
                        p.set_attribute("stt.providerName", "google")
                        p.set_attribute("stt.confidence", 0.95)
                        ttfb = getattr(getattr(self, "_metrics", None), "ttfb", None)
                        if ttfb is not None:
                            p.set_attribute("metrics.ttfb", ttfb)
                    with tracer.start_as_current_span("stt.provider_selection") as sel:
                        sel.set_attribute("stt.selectedProvider", "google")
                        sel.set_attribute("stt.reason", "fallback")

        return _InstrumentedGoogleSTT(**kwargs)


class InstrumentedOpenAILLM(OpenAILLMService):
    """OpenAILLMService that sets llm.finish_reason on the active LLM span."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._llm_span_ref = None

    async def start_llm_usage_metrics(self, tokens):
        # Called within the @traced_llm span — capture span ref and set default
        self._llm_span_ref = otel_trace.get_current_span()
        if self._llm_span_ref and self._llm_span_ref.is_recording():
            self._llm_span_ref.set_attribute("llm.finish_reason", "stop")
        await super().start_llm_usage_metrics(tokens)

    async def run_function_calls(self, function_calls):
        if self._llm_span_ref and self._llm_span_ref.is_recording():
            self._llm_span_ref.set_attribute("llm.finish_reason", "tool_calls")
        return await super().run_function_calls(function_calls)


SYSTEM_PROMPT = """You are a helpful voice assistant used for testing Coval's voice agent evaluation platform.
Keep your responses concise and conversational. You have access to tools — use them when relevant:
- get_current_time: returns the current date and time
- get_weather: returns mock weather for a city
- search_web: searches the web for up-to-date information on any topic
- lookup_order_status: looks up a mock order by order ID"""

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
        "estimated_delivery": "Mar 1, 2026",
        "carrier": random.choice(["UPS", "FedEx", "USPS", "DHL"]),
    })


async def run_agent(room_url: str, token: str | None = None):
    # COVAL: Initialize tracing before creating any tasks or runners
    setup_coval_tracing(service_name="pipecat-voice-agent")  # COVAL:

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
        # COVAL: Extract and set the simulation ID if present
        sim_id = (data.get('body') or {}).get('dialin_settings', {}).get('custom_context', {}).get('coval_simulation_id')  # COVAL:
        if sim_id:  # COVAL:
            set_simulation_id(sim_id)  # COVAL:
        logger.info(f"Dialin connected — data: {data}")

    # STT with fallback: Deepgram (primary) → Google (fallback) via ServiceSwitcher
    stt_deepgram = InstrumentedDeepgramSTT(api_key=os.getenv("DEEPGRAM_API_KEY"))
    if _HAS_GOOGLE_STT:
        stt_google = InstrumentedGoogleSTT.create(api_key=os.getenv("GOOGLE_API_KEY"))
        stt = ServiceSwitcher(
            services=[stt_deepgram, stt_google] if stt_google else [stt_deepgram],
            strategy_type=ServiceSwitcherStrategyManual,
        )
    else:
        stt = stt_deepgram

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=os.getenv("CARTESIA_VOICE_ID", "79a125e8-cd45-4c13-8a67-188112f4dd22"),
    )
    llm = InstrumentedOpenAILLM(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

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
            enable_tracing=True,  # COVAL: Ensure tracing is enabled
        ),
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

    try:
        while True:
            logger.info("Agent ready — waiting for call...")
            try:
                await run_agent(room_url, token)
            except Exception as e:
                logger.error(f"Agent error: {e}")
            logger.info("Call ended. Restarting in 2 seconds...")
            await asyncio.sleep(2)
    except KeyboardInterrupt:
        logger.info("Shutting down")


if __name__ == "__main__":
    asyncio.run(main())
