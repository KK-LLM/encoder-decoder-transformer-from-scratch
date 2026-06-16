#!/usr/bin/env bash
# Run Encoder-Decoder v0_baseline inference.
# Usage: bash scripts/run_infer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR/encoder_decoder/v0_baseline/src"
python infer_encoder_decoder.py
