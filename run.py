#!/usr/bin/env python3
"""
ThreatWatch — single-command launcher.

    python run.py                    # dashboard at http://localhost:8000
    python run.py --no-vision        # skip the camera/YOLO entirely
    python run.py --port 8080
    python run.py --reload           # development auto-reload

First run?  The server starts regardless and tells you what is missing:

    python scripts/preprocess.py                 # real WESAD -> windows (once)
    python scripts/train.py --models cnn_transformer
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the ThreatWatch dashboard")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-vision", action="store_true",
                        help="disable camera + pose detection")
    parser.add_argument("--reload", action="store_true", help="uvicorn auto-reload (dev)")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    if args.no_vision:
        os.environ["THREATWATCH_VISION"] = "0"

    from threatwatch.config import HOST, PORT

    host = args.host or HOST
    port = args.port or PORT

    # Import here so a config error surfaces as a clear message rather than
    # an opaque uvicorn import trace.
    try:
        from threatwatch.api.app import app  # noqa: F401
    except Exception as exc:  # pragma: no cover
        print(f"[run] failed to import the application: {exc}")
        print("[run] check that dependencies are installed: pip install -r requirements.txt")
        return 1

    import uvicorn

    print("=" * 62)
    print("   THREATWATCH — MULTIMODAL THREAT & DISTRESS DETECTION")
    print("=" * 62)
    print(f"   Dashboard : http://localhost:{port}")
    print(f"   Vision    : {'disabled' if args.no_vision else 'enabled'}")
    print("   Ctrl+C to stop.")
    print("=" * 62)

    uvicorn.run("threatwatch.api.app:app", host=host, port=port, reload=args.reload,
                log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
