#!/usr/bin/env bash
# Run Encoder-Decoder v0_baseline training.
# Usage: bash scripts/run_train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR/projects/encoder_decoder/v0_baseline/src"
python train_encoder_decoder_opus.py
