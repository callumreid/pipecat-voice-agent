"""
Set up Daily pinless SIP dial-in for the Pipecat agent.

Usage:
    python setup_sip.py <webhook-url>

Example:
    python setup_sip.py https://abc123.ngrok-free.app/dial

This creates a domain-dialin-config on your Daily domain, giving you a static
SIP URI. To pass a simulation_id, append it as a query parameter:

    sip:<address>@<domain>.sip.daily.co?x-coval-simulation-id=<sim-output-id>

Environment variables (from .env.local):
    DAILY_API_KEY — Daily API key (from https://dashboard.daily.co/developers)
"""

import json
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.local", override=True)

DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")


def get_existing_config() -> dict | None:
    """Check if a domain-dialin-config already exists."""
    response = requests.get(
        "https://api.daily.co/v1/domain-dialin-config",
        headers={"Authorization": f"Bearer {DAILY_API_KEY}"},
        timeout=15,
    )
    if response.status_code == 200:
        return response.json()
    return None


def create_config(webhook_url: str) -> dict:
    """Create a new pinless SIP dial-in config with the given webhook URL."""
    response = requests.post(
        "https://api.daily.co/v1/domain-dialin-config",
        headers={
            "Authorization": f"Bearer {DAILY_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "type": "pinless_dialin",
            "room_creation_api": webhook_url,
        },
        timeout=15,
    )
    if not response.ok:
        print(f"Error {response.status_code}: {response.text}")
    response.raise_for_status()
    return response.json()


def update_config(webhook_url: str, config_id: str) -> dict:
    """Update an existing domain-dialin-config with a new webhook URL."""
    response = requests.put(
        f"https://api.daily.co/v1/domain-dialin-config/{config_id}",
        headers={
            "Authorization": f"Bearer {DAILY_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"room_creation_api": webhook_url},
        timeout=15,
    )
    if not response.ok:
        print(f"Error {response.status_code}: {response.text}")
    response.raise_for_status()
    return response.json()


def main():
    if not DAILY_API_KEY:
        print("Error: DAILY_API_KEY not set — add it to .env.local")
        print("Get it from: https://dashboard.daily.co/developers")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python setup_sip.py <webhook-url>")
        print("Example: python setup_sip.py https://abc123.ngrok-free.app/dial")
        sys.exit(1)

    webhook_url = sys.argv[1].rstrip("/")
    if not webhook_url.startswith("https://"):
        print("Error: webhook URL must use HTTPS (required by Daily)")
        sys.exit(1)

    print(f"Webhook URL: {webhook_url}")
    print()

    # Check for existing config
    existing = get_existing_config()
    if existing:
        print("Existing domain-dialin-config found:")
        print(json.dumps(existing, indent=2))
        print()
        # Extract the config ID from existing pinless_dialin entries
        pinless_entries = existing.get("pinless_dialin", [])
        if pinless_entries:
            config_id = pinless_entries[0].get("id", "")
            print(f"Updating config {config_id} with new webhook URL...")
            result = update_config(webhook_url, config_id)
        else:
            print("No pinless_dialin entries found — creating new config...")
            result = create_config(webhook_url)
    else:
        print("No existing config — creating new domain-dialin-config...")
        result = create_config(webhook_url)

    print()
    print("Configuration:")
    print(json.dumps(result, indent=2))

    # Extract SIP URI — check both response formats
    sip_uri = None
    pinless_entries = result.get("pinless_dialin", [])
    if pinless_entries:
        sip_uri = pinless_entries[0].get("sip_uri", "")
    config = result.get("config", {})
    if not sip_uri and isinstance(config, dict):
        sip_uri = config.get("sip_uri", "")

    if sip_uri:
        print()
        print("=" * 70)
        print(f"  SIP URI: sip:{sip_uri}")
        print("=" * 70)
        print()
        print("To test with a simulation_id, dial:")
        print(f"  sip:{sip_uri}?x-coval-simulation-id=<your-simulation-output-id>")
        print()
        print("The x-coval-simulation-id header will appear in")
        print("on_dialin_connected → data['sipHeaders']")

        hmac_secret = config.get("hmac", "")
        if hmac_secret:
            print()
            print(f"HMAC secret (for webhook signature verification): {hmac_secret}")
    else:
        print()
        print("Warning: Could not find sip_uri in response.")
        print("Full response above — check manually.")


if __name__ == "__main__":
    main()
