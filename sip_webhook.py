"""
Webhook server for Coval/Pipecat Cloud voice test calls.

This server has two modes:

1. Daily pinless SIP dial-in. When a SIP call arrives at the Daily SIP URI,
   Daily places the caller on hold and POSTs the call details here. The server
   starts the Pipecat Cloud agent with dialin_settings so the call is patched
   into the room.
2. Coval outbound voice trigger. Coval POSTs a persona phone number here, the
   server starts a Pipecat Cloud session with Daily dial-out enabled, and the
   bot dials Coval's persona number.

Deployed on Fly.io as coval-sip-webhook. Also runnable locally for development.

Local usage:
    1. Start this server:     python sip_webhook.py
    2. Expose it publicly:    ngrok http 8080
    3. Set up Daily SIP URI:  python setup_sip.py <ngrok-url>
    4. Dial the SIP URI with custom headers:
       sip:<address>@<domain>.sip.daily.co?x-coval-simulation-id=<sim-id>

Environment variables:
    PIPECAT_API_KEY         — Pipecat Cloud public API key (required)
    PIPECAT_AGENT_NAME      — PCC agent name (default: coval-pipecat-agent)
    DAILY_OUTBOUND_CALLER_ID — Optional Daily purchased phone-number ID for PSTN caller ID
    PORT                    — Listen port (default: 8080, set by Fly.io)
"""

import json
import os
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

import requests
from loguru import logger

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=".env.local", override=True)
except ImportError:
    pass  # dotenv not needed in production (Fly.io sets env vars directly)

PIPECAT_API_KEY = os.getenv("PIPECAT_API_KEY", "")
PIPECAT_AGENT_NAME = os.getenv("PIPECAT_AGENT_NAME", "coval-pipecat-agent")
DAILY_OUTBOUND_CALLER_ID = os.getenv("DAILY_OUTBOUND_CALLER_ID", "")
LISTEN_PORT = int(os.getenv("PORT", "8080"))
_MEDIA_READY_PATHS = {"/media-ready", "/media_ready"}
_PHONE_NUMBER_KEYS = ("phone_number", "phoneNumber", "to", "recipient_phone_number")

_sessions_by_simulation_id: dict[str, dict[str, str]] = {}
_sessions_lock = threading.RLock()


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

        if self.path.rstrip("/") in _MEDIA_READY_PATHS:
            self._handle_media_ready(data)
            return

        if data.get("callId") or data.get("callDomain"):
            self._handle_daily_pinless_dialin(data)
            return

        self._handle_coval_outbound_trigger(data)

    def _handle_daily_pinless_dialin(self, data: dict[str, Any]):
        logger.info(f"Received Daily dial-in webhook: {json.dumps(data, indent=2)}")

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
            pcc_response = self._start_pcc_dialin_agent(
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

    def _handle_coval_outbound_trigger(self, data: dict[str, Any]):
        logger.info(f"Received Coval outbound trigger: {json.dumps(data, indent=2)}")

        phone_number = self._extract_phone_number(data)
        simulation_id = self._extract_simulation_id(data)

        if not phone_number:
            logger.error("Missing phone number in Coval outbound trigger")
            self._respond(400, {"error": "Missing phone_number"})
            return
        if not simulation_id:
            logger.error("Missing simulation_output_id in Coval outbound trigger")
            self._respond(400, {"error": "Missing simulation_output_id"})
            return

        try:
            pcc_response = self._start_pcc_dialout_agent(
                phone_number=phone_number,
                simulation_id=simulation_id,
                trigger_payload=data,
            )
            session_id = self._extract_session_id(pcc_response)
            if session_id:
                with _sessions_lock:
                    _sessions_by_simulation_id[simulation_id] = {
                        "session_id": session_id,
                        "created_at": str(time.time()),
                    }
                logger.info(
                    f"PCC dial-out session started: simulation_id={simulation_id} session_id={session_id}"
                )
            else:
                logger.warning(
                    f"PCC dial-out start response did not include a session id: {json.dumps(pcc_response, indent=2)}"
                )

            self._respond(
                200,
                {
                    "status": "ok",
                    "mode": "dialout",
                    "agent_name": PIPECAT_AGENT_NAME,
                    "session_id": session_id,
                    "simulation_output_id": simulation_id,
                    "media_ready_endpoint": self._absolute_media_ready_endpoint(),
                },
            )
        except Exception as error:
            logger.error(f"Failed to start PCC dial-out agent: {error}")
            self._respond(500, {"error": str(error)})

    def _handle_media_ready(self, data: dict[str, Any]):
        simulation_id = self._extract_simulation_id(data)
        if not simulation_id:
            logger.error("Missing simulation_output_id in media-ready callback")
            self._respond(400, {"error": "Missing simulation_output_id"})
            return

        with _sessions_lock:
            session = _sessions_by_simulation_id.get(simulation_id)

        if not session:
            logger.error(f"No PCC session found for simulation_id={simulation_id}")
            self._respond(404, {"error": "No PCC session found for simulation_output_id"})
            return

        session_id = session["session_id"]
        try:
            pcc_response = self._post_pcc_session_event(session_id, "media-ready", data)
            logger.info(
                f"Forwarded media-ready callback: simulation_id={simulation_id} session_id={session_id}"
            )
            self._respond(
                200,
                {
                    "status": "ok",
                    "session_id": session_id,
                    "simulation_output_id": simulation_id,
                    "pcc_response": pcc_response,
                },
            )
        except Exception as error:
            logger.error(
                f"Failed to forward media-ready callback: simulation_id={simulation_id} "
                f"session_id={session_id} error={error}"
            )
            self._respond(502, {"error": str(error)})

    def _start_pcc_dialin_agent(
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

    def _start_pcc_dialout_agent(
        self,
        phone_number: str,
        simulation_id: str,
        trigger_payload: dict[str, Any],
    ) -> dict:
        """Start a Pipecat Cloud agent session that dials Coval's persona phone number."""
        headers = {
            "Authorization": f"Bearer {PIPECAT_API_KEY}",
            "Content-Type": "application/json",
        }

        dialout_settings: dict[str, Any] = {
            "phoneNumber": phone_number,
            "displayName": f"Coval Persona {simulation_id}",
        }
        if DAILY_OUTBOUND_CALLER_ID:
            dialout_settings["callerId"] = DAILY_OUTBOUND_CALLER_ID

        payload = {
            "createDailyRoom": True,
            "dailyRoomProperties": {
                "enable_dialout": True,
                "exp": int(time.time()) + 3600,
            },
            "body": {
                "dialout_settings": dialout_settings,
                "coval": {
                    "simulationOutputId": simulation_id,
                    "waitForMediaReady": True,
                    "triggerPayload": trigger_payload,
                },
            },
        }

        logger.info(
            f"Starting PCC dial-out agent '{PIPECAT_AGENT_NAME}' for simulation_id={simulation_id} "
            f"phone_number={phone_number}"
        )
        response = requests.post(
            f"https://api.pipecat.daily.co/v1/public/{PIPECAT_AGENT_NAME}/start",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def _post_pcc_session_event(self, session_id: str, path: str, payload: dict[str, Any]) -> dict:
        headers = {
            "Authorization": f"Bearer {PIPECAT_API_KEY}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"https://api.pipecat.daily.co/v1/public/{PIPECAT_AGENT_NAME}/sessions/{session_id}/{path}",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        try:
            return response.json()
        except json.JSONDecodeError:
            return {"raw_response": response.text}

    def _extract_phone_number(self, data: dict[str, Any]) -> str:
        for key in _PHONE_NUMBER_KEYS:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _extract_simulation_id(self, data: dict[str, Any]) -> str:
        for key in ("simulation_output_id", "simulationOutputId", "coval_simulation_output_id"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        coval = data.get("coval")
        if isinstance(coval, dict):
            value = coval.get("simulationOutputId") or coval.get("simulation_output_id")
            if isinstance(value, str) and value.strip():
                return value.strip()

        return ""

    def _extract_session_id(self, data: Any) -> str:
        if isinstance(data, dict):
            for key in ("sessionId", "session_id", "sessionID"):
                value = data.get(key)
                if isinstance(value, str) and value:
                    return value
            for value in data.values():
                session_id = self._extract_session_id(value)
                if session_id:
                    return session_id
        elif isinstance(data, list):
            for value in data:
                session_id = self._extract_session_id(value)
                if session_id:
                    return session_id
        return ""

    def _absolute_media_ready_endpoint(self) -> str:
        host = self.headers.get("X-Forwarded-Host") or self.headers.get("Host") or ""
        proto = self.headers.get("X-Forwarded-Proto") or "https"
        if host:
            return f"{proto}://{host}/media-ready"
        return "/media-ready"

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
