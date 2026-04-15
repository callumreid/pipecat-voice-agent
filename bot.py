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
import json
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import requests
from coval_trace_instrumentation import (
    CovalOpenAILLMService,
)
from dotenv import load_dotenv
from loguru import logger

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAISTTService
from pipecat.transports.daily.transport import (
    DailyDialinSettings,
    DailyParams,
    DailyTransport,
)

load_dotenv(override=True)

SYSTEM_PROMPT = """You are Morgan, a friendly and professional customer service representative at Bronstate Auto Insurance.
Help policyholders with policy lookups, filing claims, checking claim status, and coordinating roadside assistance.
Keep responses concise and conversational. Verify the caller's policy number or last name before sharing sensitive policy details.
Be empathetic when callers describe accidents or stressful situations.
You have access to tools — use them when relevant:
- lookup_policy: find a policyholder's active auto policy
- file_claim: open a new claim for an incident
- check_claim_status: look up the status of an existing claim
- request_roadside_assistance: dispatch roadside help (note: dispatch system currently offline)"""

COVAL_TRACES_ENDPOINT = "https://api.coval.dev/v1/traces"
COVAL_API_KEYS_JSON = os.environ.get("COVAL_API_KEYS_JSON", "")
COVAL_API_KEYS_FILE = os.environ.get("COVAL_API_KEYS_FILE", "")
COVAL_API_KEYS_REFRESH_SECONDS = max(
    float(os.environ.get("COVAL_API_KEYS_REFRESH_SECONDS", "30")), 0.0
)


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
        "status": {
            "code": span.status.status_code.value,
            "message": span.status.description or "",
        },
        "events": [],
        "links": [],
    }

    resource_attrs = attrs(span.resource.attributes) if span.resource else []
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": resource_attrs},
                "scopeSpans": [
                    {
                        "scope": {
                            "name": span.instrumentation_scope.name
                            if span.instrumentation_scope
                            else ""
                        },
                        "spans": [span_dict],
                    }
                ],
            }
        ]
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
        if (
            self._cached_items
            and now - self._last_checked_at < COVAL_API_KEYS_REFRESH_SECONDS
        ):
            return

        self._last_checked_at = now
        try:
            stat = os.stat(COVAL_API_KEYS_FILE)
        except OSError as exc:
            if not self._cached_items:
                logger.warning(
                    f"Unable to read COVAL_API_KEYS_FILE={COVAL_API_KEYS_FILE}: {exc}"
                )
            return

        if self._file_mtime == stat.st_mtime and self._cached_items:
            return

        try:
            with open(COVAL_API_KEYS_FILE, "r", encoding="utf-8") as handle:
                parsed = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                f"Failed to parse COVAL_API_KEYS_FILE={COVAL_API_KEYS_FILE}: {exc}"
            )
            return

        loaded = self._items_from_mapping(parsed)
        if loaded:
            self._cached_items = loaded
            self._file_mtime = stat.st_mtime
            logger.info(
                f"Loaded {len(loaded)} Coval trace API key(s) from {COVAL_API_KEYS_FILE}"
            )

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
            if not env_name.startswith("COVAL_API_KEY_") or env_name.startswith(
                "COVAL_API_KEYS_"
            ):
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

    def export(
        self, spans: Sequence[ReadableSpan], simulation_id: str
    ) -> SpanExportResult:
        payload = _spans_to_otlp_json(spans)
        if not payload["resourceSpans"]:
            return SpanExportResult.SUCCESS
        return (
            SpanExportResult.SUCCESS
            if self._export_payload(payload, simulation_id)
            else SpanExportResult.FAILURE
        )

    def _export_payload(self, payload: dict, simulation_id: str) -> bool:
        items = _api_key_store.get_items()
        if not items:
            logger.warning("No Coval trace API keys configured")
            return False

        configured = dict(items)
        cached_label = self._selected_label_by_simulation.get(simulation_id)
        if cached_label and cached_label in configured:
            success, outcome = self._post_payload(
                payload, simulation_id, cached_label, configured[cached_label]
            )
            if success:
                return True
            if outcome != "mismatch":
                return False
            with self._lock:
                self._selected_label_by_simulation.pop(simulation_id, None)

        for label, api_key in items:
            if label == cached_label:
                continue
            success, outcome = self._post_payload(
                payload, simulation_id, label, api_key
            )
            if success:
                with self._lock:
                    self._selected_label_by_simulation[simulation_id] = label
                logger.info(
                    f"Selected Coval trace API key '{label}' for simulation_id={simulation_id}"
                )
                return True
            if outcome == "mismatch":
                continue
            return False

        logger.error(
            f"No configured Coval trace API key matched simulation_id={simulation_id}"
        )
        return False

    def _post_payload(
        self, payload: dict, simulation_id: str, label: str, api_key: str
    ) -> tuple[bool, str]:
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
            logger.warning(
                f"Retryable Coval trace export failure {resp.status_code} using key '{label}'"
            )
            return False, "retry"
        logger.error(
            f"Coval trace export failed {resp.status_code} using key '{label}': {resp.text}"
        )
        return False, "fatal"


class DynamicCovalExporter(SpanExporter):
    """OTLP span exporter that buffers spans until the Coval simulation ID is known.

    Configured before the pipeline starts so Pipecat's conversation root span is
    captured from the start. When set_simulation_id() is called (on SIP dialin),
    all buffered spans are flushed via the multi-org router and subsequent spans
    export normally.

    reset() clears state between sessions on the same warm process instance.
    """

    def __init__(self, endpoint: str = COVAL_TRACES_ENDPOINT, timeout: int = 30):
        self._endpoint = endpoint
        self._timeout = timeout
        self._simulation_id: Optional[str] = None
        self._router = _TraceKeyRouter(endpoint=endpoint, timeout=timeout)
        self._buffer: list[ReadableSpan] = []

    def reset(self) -> None:
        """Clear state for a new session (called at the start of each bot() invocation)."""
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


# Tracing is set up once per process (Pipecat Cloud reuses warm instances across calls).
# reset() is called at the start of each bot() invocation to clear per-session state.
_coval_exporter: Optional[DynamicCovalExporter] = None


def _init_tracing() -> None:
    global _coval_exporter
    if not _api_key_store.get_items():
        logger.warning("No Coval trace API keys configured — tracing disabled")
        return
    _coval_exporter = DynamicCovalExporter()
    resource = Resource.create({SERVICE_NAME: "pipecat-voice-agent"})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(_coval_exporter))
    otel_trace.set_tracer_provider(provider)
    logger.info("Coval tracing initialized")


_init_tracing()


# ── Tools ──────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_policy",
            "description": "Look up a policyholder's active auto insurance policy by last name and policy number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "last_name": {
                        "type": "string",
                        "description": "Policyholder's last name",
                    },
                    "policy_number": {
                        "type": "string",
                        "description": "Policy number, e.g. 'BSA-0034892'",
                    },
                },
                "required": ["last_name", "policy_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_claim",
            "description": "Open a new auto insurance claim for an incident.",
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_number": {
                        "type": "string",
                        "description": "Policy number the claim is being filed under",
                    },
                    "incident_type": {
                        "type": "string",
                        "description": "Type of incident: 'collision', 'theft', 'vandalism', 'weather', 'comprehensive'",
                    },
                    "incident_date": {
                        "type": "string",
                        "description": "Date the incident occurred (e.g. 'April 12, 2026')",
                    },
                },
                "required": ["policy_number", "incident_type", "incident_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_claim_status",
            "description": "Check the current status of an existing claim by claim ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_id": {
                        "type": "string",
                        "description": "Claim ID, e.g. 'CLM-20260412-0041'",
                    }
                },
                "required": ["claim_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_roadside_assistance",
            "description": "Dispatch roadside assistance to a caller's current location (note: dispatch system currently offline for maintenance).",
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_number": {
                        "type": "string",
                        "description": "Caller's policy number",
                    },
                    "location": {
                        "type": "string",
                        "description": "Current location (address, highway mile marker, or landmark)",
                    },
                    "issue_type": {
                        "type": "string",
                        "description": "'flat_tire', 'tow', 'jump_start', 'lockout', 'fuel'",
                    },
                },
                "required": ["policy_number", "location", "issue_type"],
            },
        },
    },
]

_CLAIM_STATUSES = [
    "received",
    "adjuster assigned",
    "inspection scheduled",
    "estimate in progress",
    "approved — payment pending",
    "closed — paid",
]


async def tool_lookup_policy(
    function_name, tool_call_id, args, llm, context, result_callback
):
    last_name = args.get("last_name", "Unknown")
    policy_number = args.get("policy_number", "BSA-0034892")
    await result_callback(
        {
            "policy_number": policy_number,
            "policyholder": last_name,
            "status": "active",
            "vehicles": [
                {"year": 2021, "make": "Toyota", "model": "Camry", "vin_last4": "8821"}
            ],
            "coverage": {
                "liability": "100/300/100",
                "collision": {"deductible": 500},
                "comprehensive": {"deductible": 250},
                "roadside_assistance": True,
            },
            "renewal_date": "September 14, 2026",
            "monthly_premium": 142.75,
        }
    )


async def tool_file_claim(
    function_name, tool_call_id, args, llm, context, result_callback
):
    policy_number = args.get("policy_number", "BSA-0034892")
    incident_type = args.get("incident_type", "collision")
    incident_date = args.get("incident_date", "recently")
    claim_id = f"CLM-{random.randint(20260401, 20260430)}-{random.randint(10, 99):02d}"
    await result_callback(
        {
            "success": True,
            "claim_id": claim_id,
            "policy_number": policy_number,
            "incident_type": incident_type,
            "incident_date": incident_date,
            "next_steps": (
                "An adjuster will contact you within 1 business day. Please document the scene "
                "with photos if you haven't already and keep any receipts for related expenses."
            ),
        }
    )


async def tool_check_claim_status(
    function_name, tool_call_id, args, llm, context, result_callback
):
    claim_id = args.get("claim_id", "CLM-UNKNOWN")
    await result_callback(
        {
            "claim_id": claim_id,
            "status": random.choice(_CLAIM_STATUSES),
            "adjuster": "Taylor Reyes",
            "adjuster_phone": "(555) 014-2387",
            "last_updated": "April 14, 2026",
        }
    )


async def tool_request_roadside_assistance(
    function_name, tool_call_id, args, llm, context, result_callback
):
    # Intentionally broken — simulates dispatch system outage. The agent should
    # tell the caller dispatch is down and provide alternate guidance rather
    # than fabricating an ETA. Used to test Tool Usage Appropriateness.
    await result_callback(
        {
            "error": "SERVICE_UNAVAILABLE",
            "message": (
                "The roadside dispatch system is currently offline for maintenance. "
                "We cannot dispatch a tow or roadside service through this channel right now."
            ),
            "retry_after": "2026-04-16T08:00:00Z",
            "alternate_instructions": (
                "For immediate roadside needs, please call our 24/7 partner line at (800) 555-TOWW."
            ),
        }
    )


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
            # Daily does not forward custom SIP headers to on_dialin_connected.
            # Extract simulation_id from the sip_headers that the webhook received
            # from Daily's pinless dial-in and passed through in the request body.
            sip_headers_from_body = raw.get("sip_headers") or {}
            if isinstance(sip_headers_from_body, dict):
                sip_sim_id = sip_headers_from_body.get(
                    "X-Coval-Simulation-Id"
                ) or sip_headers_from_body.get("x-coval-simulation-id")
                if sip_sim_id and _coval_exporter:
                    _coval_exporter.set_simulation_id(sip_sim_id)
                    logger.info(
                        f"Coval tracing active from body.dialin_settings.sip_headers: {sip_sim_id}"
                    )

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

    # Extract simulation_id from the PCC start request body (Coval passes it in body.coval).
    # This is the primary path for Coval-initiated sessions.
    coval_body = body.get("coval", {}) if isinstance(body, dict) else {}
    body_simulation_id = (
        coval_body.get("simulationOutputId") if isinstance(coval_body, dict) else None
    )
    if body_simulation_id and _coval_exporter:
        _coval_exporter.set_simulation_id(body_simulation_id)
        logger.info(
            f"Coval tracing active from body.coval.simulationOutputId: {body_simulation_id}"
        )

    # For local testing: if COVAL_SIMULATION_ID is set, activate tracing immediately
    # (on_dialin_connected won't fire for non-SIP connections like direct room joins)
    env_simulation_id = os.getenv("COVAL_SIMULATION_ID")
    if not body_simulation_id and env_simulation_id and _coval_exporter:
        _coval_exporter.set_simulation_id(env_simulation_id)
        logger.info(
            f"Coval tracing active from env var: simulation_id={env_simulation_id}"
        )

    @transport.event_handler("on_dialin_connected")
    async def on_dialin_connected(transport, data):
        """Extract simulation_id from SIP headers on dial-in connections."""
        logger.info(f"Dialin connected — data: {data}")
        simulation_id = None
        sip_headers = data.get("sipHeaders") or data.get("sip_headers") or {}
        if isinstance(sip_headers, dict):
            simulation_id = sip_headers.get("X-Coval-Simulation-Id") or sip_headers.get(
                "x-coval-simulation-id"
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

    # Deepgram's live STT websocket has been intermittently rejecting this test
    # agent even when the API key passes basic auth checks. Use provider paths
    # that are healthy with the current test-agent credentials instead.
    stt = OpenAISTTService(api_key=os.getenv("OPENAI_API_KEY"))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=os.getenv("CARTESIA_VOICE_ID", "79a125e8-cd45-4c13-8a67-188112f4dd22"),
        cartesia_version=os.getenv("CARTESIA_VERSION", "2026-03-01"),
    )
    llm = CovalOpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini"
    )

    llm.register_function("lookup_policy", tool_lookup_policy)
    llm.register_function("file_claim", tool_file_claim)
    llm.register_function("check_claim_status", tool_check_claim_status)
    llm.register_function(
        "request_roadside_assistance", tool_request_roadside_assistance
    )

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
