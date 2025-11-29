#!/usr/bin/env bash
set -euo pipefail

uvicorn serving.api_server:app --host 0.0.0.0 --port 8000 --reload
