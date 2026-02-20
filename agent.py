"""
Pipecat voice agent with OpenTelemetry tracing configured to send spans to Coval.

This agent is used for testing Coval's trace ingestion and viewer. It automatically
emits structured spans for each conversation turn, STT, LLM, and TTS operation.

The simulation_output_id is read dynamically from the SIP INVITE custom header
X-Coval-Simulation-Id, which Coval sends via Telnyx when dialing into the agent.
Falls back to COVAL_SIMULATION_ID env var for manual testing.

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

from dotenv import load_dotenv
from loguru import logger
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

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
from pipecat.utils.tracing.setup import setup_tracing

load_dotenv(".env.local")

SYSTEM_PROMPT = """You are a helpful voice assistant used for testing Coval's trace ingestion.
Keep your responses concise and conversational. You may be asked about anything.
If asked to call a tool you don't have, politely explain you don't have that capability."""


def extract_simulation_id(participant: dict) -> str | None:
    """Extract the simulation_output_id from a Daily SIP participant.

    Coval sends X-Coval-Simulation-Id as a custom SIP header via Telnyx.
    Daily surfaces custom SIP headers in the participant's sipHeaders field.
    Falls back to the COVAL_SIMULATION_ID env var for manual testing.
    """
    # Daily surfaces custom SIP headers under sipHeaders in the participant object.
    # The header name is lowercased by Daily: x-coval-simulation-id.
    sip_headers = participant.get("sipHeaders") or participant.get("sip_headers") or {}
    if isinstance(sip_headers, dict):
        simulation_id = (
            sip_headers.get("x-coval-simulation-id")
            or sip_headers.get("X-Coval-Simulation-Id")
        )
        if simulation_id:
            return simulation_id

    # Fallback: env var for manual testing
    return os.getenv("COVAL_SIMULATION_ID") or None


def build_coval_exporter(simulation_id: str) -> OTLPSpanExporter | None:
    """Build an OTLP exporter pointing at Coval's trace ingestion endpoint."""
    api_key = os.getenv("COVAL_API_KEY")
    if not api_key:
        logger.warning("COVAL_API_KEY not set — tracing disabled")
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
        logger.info(f"Participant joined: {participant.get('id')} | data: {participant}")

        simulation_id = extract_simulation_id(participant)
        if simulation_id:
            exporter = build_coval_exporter(simulation_id)
            if exporter:
                setup_tracing(
                    service_name="pipecat-voice-agent",
                    exporter=exporter,
                    console_export=os.getenv("OTEL_CONSOLE_EXPORT", "").lower() == "true",
                )
        else:
            logger.warning("No simulation_id found — traces will not be sent to Coval")

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

    await run_agent(room_url, token)


if __name__ == "__main__":
    asyncio.run(main())
