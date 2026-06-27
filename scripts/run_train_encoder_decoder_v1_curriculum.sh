#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

STAGE="${1:-stage1}"
OUTPUT_DIR="${2:-outputs/v1_curriculum/${STAGE}}"
RESUME_CHECKPOINT="${3:-}"

cd "$REPO_ROOT"

CMD=(
  python encoder_decoder/v1_curriculum/src/train_encoder_decoder_curriculum.py
  --stage "$STAGE"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "$RESUME_CHECKPOINT" ]]; then
  CMD+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

"${CMD[@]}"
