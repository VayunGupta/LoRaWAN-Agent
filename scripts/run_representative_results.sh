#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs/logs/representative_results

run_case() {
  local name="$1"
  shift
  echo "=== Running ${name} ==="
  .venv/bin/python -m src.run_react_agent "$@" \
    2>&1 | tee "outputs/logs/representative_results/${name}.txt"
  echo
}

run_case "01_clean_centralized_trust" \
  --attack-type none \
  --llm-mode off \
  --architecture centralized_trust

run_case "02_clean_loramas" \
  --attack-type none \
  --llm-mode off \
  --architecture loramas

run_case "03_weak_noise_loramas_osm" \
  --attack-type random_noise \
  --sensor sensor08 \
  --noise-sigma-db 1.5 \
  --llm-mode off \
  --architecture loramas \
  --use-environment-context

run_case "04_weak_noise_loramas_osm_no_environment_role" \
  --attack-type random_noise \
  --sensor sensor08 \
  --noise-sigma-db 1.5 \
  --llm-mode off \
  --architecture loramas \
  --use-environment-context \
  --disable-role environment_consistency

run_case "05_replay_centralized_trust" \
  --attack-type replay_attack \
  --gateway gatewayA \
  --llm-mode off \
  --architecture centralized_trust

run_case "06_replay_loramas" \
  --attack-type replay_attack \
  --gateway gatewayA \
  --llm-mode off \
  --architecture loramas

run_case "07_gateway_fabrication_centralized_trust" \
  --attack-type gateway_fabrication \
  --gateway gatewayA \
  --fabricate-fraction 0.08 \
  --fabricate-shift-db 9.0 \
  --seed 19 \
  --llm-mode off \
  --architecture centralized_trust

run_case "08_gateway_fabrication_loramas" \
  --attack-type gateway_fabrication \
  --gateway gatewayA \
  --fabricate-fraction 0.08 \
  --fabricate-shift-db 9.0 \
  --seed 19 \
  --llm-mode off \
  --architecture loramas

echo "Saved traces in outputs/logs/representative_results/"
