"""
Pipecat voice agent for testing Coval's evaluation platform.

Cloud: async def bot(args) is called by Pipecat Cloud base image per session.
       args.room_url and args.token are injected by the platform.
Local: python bot.py — reads DAILY_ROOM_URL / DAILY_TOKEN from .env.local.
"""

import asyncio
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.daily.transport import DailyDialinSettings, DailyParams, DailyTransport

load_dotenv(override=True)

SYSTEM_PROMPT = """You are a helpful voice assistant used for testing Coval's voice agent evaluation platform.
Keep your responses concise and conversational. You have access to tools — use them when relevant:
- get_current_time: returns the current date and time
- get_weather: returns mock weather for a city
- search_web: searches the web for up-to-date information on any topic
- lookup_order_status: looks up a mock order by order ID"""


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

    # Extract dialin_settings from body (passed by PCC's pinless dial-in webhook)
    body = getattr(args, "body", None) or {}
    logger.info(f"args.body = {body}")
    dialin_settings = None
    if isinstance(body, dict):
        raw = body.get("dialin_settings")
        if raw:
            dialin_settings = DailyDialinSettings(
                call_id=raw.get("callId") or raw.get("call_id", ""),
                call_domain=raw.get("callDomain") or raw.get("call_domain", ""),
            )
            logger.info(f"Dial-in session: call_id={dialin_settings.call_id}, call_domain={dialin_settings.call_domain}")

    daily_api_key = os.getenv("DAILY_API_KEY", "")

    transport = DailyTransport(
        args.room_url,
        args.token,
        "Pipecat Test Agent",
        DailyParams(
            api_key=daily_api_key,
            audio_out_enabled=True,
            transcription_enabled=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            dialin_settings=dialin_settings,
        ),
    )

    @transport.event_handler("on_dialin_ready")
    async def on_dialin_ready(transport, sip_endpoint):
        logger.info(f"Dialin READY — sip_endpoint: {sip_endpoint}")

    @transport.event_handler("on_dialin_connected")
    async def on_dialin_connected(transport, data):
        logger.info(f"Dialin CONNECTED — data: {data}")

    @transport.event_handler("on_dialin_stopped")
    async def on_dialin_stopped(transport, data):
        logger.info(f"Dialin STOPPED — data: {data}")

    @transport.event_handler("on_dialin_error")
    async def on_dialin_error(transport, data):
        logger.error(f"Dialin ERROR — data: {data}")

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

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
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
        handle_sigint: bool = True

    room_url = os.getenv("DAILY_ROOM_URL", "")
    if not room_url:
        raise SystemExit("DAILY_ROOM_URL not set — add it to .env.local")

    _args = _LocalArgs(room_url=room_url, token=os.getenv("DAILY_TOKEN", ""))
    logger.info(f"Local dev — joining room: {room_url}")
    asyncio.run(bot(_args))
