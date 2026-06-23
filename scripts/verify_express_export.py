#!/usr/bin/env python3
"""Validate the Express recommender-export endpoint before bootstrap.

Usage:
  EXPRESS_INTERNAL_URL=http://express:3000 RECOMMENDER_API_KEY=secret \\
    python scripts/verify_express_export.py
"""
from __future__ import annotations

import json
import os
import sys

from src.data.express_loader import fetch_snapshot


def main() -> int:
    base_url = os.environ.get("EXPRESS_INTERNAL_URL", "").strip()
    api_key = os.environ.get("RECOMMENDER_API_KEY")
    timeout_ms = int(os.environ.get("EXPRESS_TIMEOUT_MS", "5000"))

    if not base_url:
        print("ERROR: EXPRESS_INTERNAL_URL is not set", file=sys.stderr)
        return 1

    try:
        payload = fetch_snapshot(base_url, api_key=api_key, timeout_ms=timeout_ms)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not fetch export: {exc}", file=sys.stderr)
        return 1

    users = payload.get("users") or []
    posts = payload.get("posts") or []
    interactions = payload.get("interactions") or []

    print(json.dumps({"users": len(users), "posts": len(posts), "interactions": len(interactions)}, indent=2))

    if not users:
        print(
            "ERROR: export returned zero users — recommender will 404 every /recommend.",
            file=sys.stderr,
        )
        return 1

    if not posts:
        print("WARNING: export returned zero posts", file=sys.stderr)

    sample = users[0].get("user_id")
    print(f"Sample user_id from export: {sample!r}")
    print("Ensure Express sends the same user_id format to POST /recommend.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
