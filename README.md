# Pipecat Voice Agent (Coval OTel Test Agent)

A Pipecat-based voice agent with OpenTelemetry tracing configured to send spans to Coval. Used to test and validate trace ingestion, storage, and visualization end-to-end.

## What it does

When a conversation happens, Pipecat automatically emits structured spans:

```
conversation
└── turn
    ├── stt    — transcript, stt.confidence, TTFB
    ├── llm    — model, input/output text, token counts, llm.finish_reason, TTFB
    └── tts    — voice_id, text, TTFB
```

The example wraps Pipecat's built-in traced Deepgram/OpenAI services so these attributes land on the standard `stt` and `llm` spans instead of creating duplicate custom spans.

These land in Coval's ClickHouse `otel.otel_traces` table, tagged with the simulation output ID. You can view them at:
- Internal: `https://app.coval.dev/<org>/runs/<run_id>/results/<sim_id>/traces-internal`
- Customer: `https://app.coval.dev/<org>/runs/<run_id>/results/<sim_id>/traces`

## Stack

| Component | Provider |
|---|---|
| Transport | Daily (WebRTC) |
| STT | Deepgram |
| LLM | OpenAI GPT-4o-mini |
| TTS | Cartesia |
| Tracing | OpenTelemetry → Coval |

## Setup

**1. Install uv (if not already):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Install dependencies:**
```bash
uv sync
```

**3. Create a Daily room:**

Go to [Daily dashboard](https://dashboard.daily.co/) and create a room, or use the Daily API. Copy the room URL.

**4. Configure environment:**
```bash
cp .env.example .env.local
# Fill in DAILY_ROOM_URL, OPENAI_API_KEY, DEEPGRAM_API_KEY, CARTESIA_API_KEY, COVAL_API_KEY
```

**5. Before each test run, set the simulation ID:**
```bash
# Get this from the Coval platform after starting a simulation
export COVAL_SIMULATION_ID=<simulation_output_id>
```

Or set it in `.env.local` (update it before each run).

## Running

```bash
uv run python agent.py
```

The agent will join the Daily room and wait for a participant. Connect via the Daily Prebuilt UI at your room URL, or have Coval dial in for a simulation.

## Debugging traces locally

Set `OTEL_CONSOLE_EXPORT=true` in `.env.local` to also print spans to stdout. Useful to confirm spans are being emitted before checking Coval.

## How the simulation ID works

`COVAL_SIMULATION_ID` must match an existing `SimulationOutput` record in Coval's database. The ingestion lambda validates this. For manual testing:
1. Create a simulation in Coval
2. Copy the simulation output ID from the URL or API
3. Set it in env before starting the agent

For automated simulations, this ID flows in from the SIP INVITE or room metadata (future work — dynamically reading from Daily room metadata).

## Connecting to Coval

For a Coval simulation to dial into this agent, configure the agent in Coval to point to the Daily room. Coval joins as a participant and drives the conversation. Ensure the agent is running and joined to the room before the simulation starts.
