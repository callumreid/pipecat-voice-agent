"""
Pipecat voice agent with OpenTelemetry tracing configured to send spans to Coval.

This agent is used for testing Coval's trace ingestion and viewer. It automatically
emits structured spans for each conversation turn, STT, LLM, and TTS operation.

Environment variables:
  DAILY_API_KEY           - Daily.co API key (for room creation)
  DAILY_ROOM_URL          - Daily room URL to join (or set DAILY_API_KEY to auto-create)
  OPENAI_API_KEY          - OpenAI API key
  DEEPGRAM_API_KEY        - Deepgram API key (STT)
  CARTESIA_API_KEY        - Cartesia API key (TTS)
  COVAL_API_KEY           - Coval organization API key
  COVAL_SIMULATION_ID     - Simulation output ID to associate traces with
                            (set per-simulation; can be read from Daily room metadata at runtime)
"""

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.utils.tracing import setup_tracing

load_dotenv(".env.local")

SYSTEM_PROMPT = """You are a helpful voice assistant used for testing Coval's trace ingestion.
Keep your responses concise and conversational. You may be asked about anything.
If asked to call a tool you don't have, politely explain you don't have that capability."""


def build_coval_exporter() -> OTLPSpanExporter | None:
    """Build an OTLP exporter pointing at Coval's trace ingestion endpoint.

    Returns None if required env vars are not set, which disables tracing.
    """
    api_key = os.getenv("COVAL_API_KEY")
    simulation_id = os.getenv("COVAL_SIMULATION_ID")

    if not api_key:
        logger.warning("COVAL_API_KEY not set — tracing disabled")
        return None
    if not simulation_id:
        logger.warning("COVAL_SIMULATION_ID not set — tracing disabled")
        return None

    logger.info(f"Coval tracing enabled for simulation_id={simulation_id}")
    return OTLPSpanExporter(
        endpoint="https://api.coval.dev/v1/traces",
        headers={
            "X-API-Key": api_key,
            "X-Simulation-Id": simulation_id,
        },
        timeout=30,
    )


async def run_agent(room_url: str, token: str | None = None):
    transport = DailyTransport(
        room_url,
        token,
        "Pipecat Test Agent",
        DailyParams(
            audio_out_enabled=True,
            transcription_enabled=False,  # Using Deepgram STT instead
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

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
        await transport.capture_participant_transcription(participant["id"])
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    runner = PipelineRunner()
    await runner.run(task)


async def main():
    room_url = os.getenv("DAILY_ROOM_URL")
    token = os.getenv("DAILY_TOKEN")

    if not room_url:
        raise ValueError("DAILY_ROOM_URL must be set")

    # Set up Coval OTel tracing — must be called before the pipeline runs
    exporter = build_coval_exporter()
    if exporter:
        setup_tracing(
            service_name="pipecat-voice-agent",
            exporter=exporter,
            console_export=os.getenv("OTEL_CONSOLE_EXPORT", "").lower() == "true",
        )

    await run_agent(room_url, token)


if __name__ == "__main__":
    asyncio.run(main())
