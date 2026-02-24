"""
Webhook server for Daily pinless SIP dial-in.

When a SIP call arrives at the Daily SIP URI, Daily places the caller on hold
and POSTs the call details here. This server starts the Pipecat Cloud agent
with dialin_settings so the call is automatically patched into the room.

Deployed on Fly.io as coval-sip-webhook. Also runnable locally for development.

Local usage:
    1. Start this server:     python sip_webhook.py
    2. Expose it publicly:    ngrok http 8080
    3. Set up Daily SIP URI:  python setup_sip.py <ngrok-url>
    4. Dial the SIP URI with custom headers:
       sip:<address>@<domain>.sip.daily.co?x-coval-simulation-id=<sim-id>

Environment variables:
    PIPECAT_API_KEY    — Pipecat Cloud API key (required)
    PIPECAT_AGENT_NAME — PCC agent name (default: coval-pipecat-agent)
    PORT               — Listen port (default: 8080, set by Fly.io)
"""

import json
import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import requests
from loguru import logger

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=".env.local", override=True)
except ImportError:
    pass  # dotenv not needed in production (Fly.io sets env vars directly)

PIPECAT_API_KEY = os.getenv("PIPECAT_API_KEY", "")
PIPECAT_AGENT_NAME = os.getenv("PIPECAT_AGENT_NAME", "coval-pipecat-agent")
LISTEN_PORT = int(os.getenv("PORT", "8080"))


class DialinWebhookHandler(BaseHTTPRequestHandler):
    """Handle POST from Daily's pinless SIP dial-in."""

    def do_GET(self):
        """Health check endpoint."""
        self._respond(200, {"status": "ok"})

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON: {body}")
            self._respond(400, {"error": "Invalid JSON"})
            return

        logger.info(f"Received dial-in webhook: {json.dumps(data, indent=2)}")

        call_id = data.get("callId", "")
        call_domain = data.get("callDomain", "")
        sip_from = data.get("From", "")
        sip_to = data.get("To", "")
        sip_headers = data.get("sipHeaders", {})

        if not call_id or not call_domain:
            logger.error("Missing callId or callDomain")
            self._respond(400, {"error": "Missing callId or callDomain"})
            return

        simulation_id = (
            sip_headers.get("X-Coval-Simulation-Id")
            or sip_headers.get("x-coval-simulation-id")
            or ""
        )
        if simulation_id:
            logger.info(f"SIP header x-coval-simulation-id: {simulation_id}")
        else:
            logger.warning("No x-coval-simulation-id in SIP headers")

        # Start the PCC agent with dial-in settings
        try:
            pcc_response = self._start_pcc_agent(
                call_id=call_id,
                call_domain=call_domain,
                sip_from=sip_from,
                sip_to=sip_to,
                sip_headers=sip_headers,
            )
            logger.info(f"PCC agent started: {json.dumps(pcc_response, indent=2)}")
            self._respond(200, {"status": "ok", "pcc_response": pcc_response})
        except Exception as error:
            logger.error(f"Failed to start PCC agent: {error}")
            self._respond(500, {"error": str(error)})

    def _start_pcc_agent(
        self,
        call_id: str,
        call_domain: str,
        sip_from: str,
        sip_to: str,
        sip_headers: dict,
    ) -> dict:
        """Start a Pipecat Cloud agent session with SIP dial-in settings."""
        headers = {
            "Authorization": f"Bearer {PIPECAT_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "createDailyRoom": True,
            "dailyRoomProperties": {
                "sip": {
                    "display_name": sip_from or "sip-caller",
                    "sip_mode": "dial-in",
                    "num_endpoints": 1,
                },
                "exp": int(time.time()) + 3600,
            },
            "body": {
                "dialin_settings": {
                    "call_id": call_id,
                    "call_domain": call_domain,
                    "From": sip_from,
                    "To": sip_to,
                    "sip_headers": sip_headers,
                }
            },
        }

        logger.info(f"Starting PCC agent '{PIPECAT_AGENT_NAME}' with payload: {json.dumps(payload, indent=2)}")
        response = requests.post(
            f"https://api.pipecat.daily.co/v1/public/{PIPECAT_AGENT_NAME}/start",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def _respond(self, status_code: int, body: dict):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, format, *args):
        """Suppress default HTTP server logging (loguru handles it)."""
        pass


def main():
    if not PIPECAT_API_KEY:
        raise SystemExit("PIPECAT_API_KEY not set")

    server = HTTPServer(("0.0.0.0", LISTEN_PORT), DialinWebhookHandler)
    logger.info(f"SIP webhook server listening on port {LISTEN_PORT}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down webhook server")
        server.server_close()


if __name__ == "__main__":
    main()
