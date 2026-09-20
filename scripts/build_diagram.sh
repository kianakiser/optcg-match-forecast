#!/usr/bin/env bash
# Regenerate the FTI architecture diagram with this project's labels.
# Run after any stack change so README/proposal stay in sync.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python scripts/make_architecture_diagram.py \
  --out docs/src/architecture.png \
  --source-title "Limitless TCG" \
  --source-subtitle "tournament API" \
  --feature-sub "ingest · compute · write" \
  --training-sub "train · evaluate · register" \
  --inference-sub "load champion · predict" \
  --ui-sub "match forecast" \
  --feature-store "versioned Parquet" \
  --model-registry "versioned + aliases" \
  --feature-trigger "GitHub Actions (daily) · backfill" \
  --training-trigger "GitHub Actions (weekly)" \
  --inference-trigger "on demand (UI) / nightly"
