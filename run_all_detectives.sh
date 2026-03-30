#!/bin/bash
# Run all detective model × history mode combinations, 5 runs each
# Sonnet 4.6 with history already has 10 runs, skip it

set -e
PYTHON=".venv/bin/python"
BASE="$PYTHON -m algo_helpers.adversarial_helpers --save_response --output_path reports/output --config_path .env --continue_runs"

echo "=== Sonnet 4.6 + history (already have 10, ensuring 10) ==="
$BASE --models_file config_human.yaml --num_runs 10

echo "=== Sonnet 4.6 + no_history (5 runs) ==="
$BASE --models_file config_human_no_history.yaml --num_runs 5

echo "=== GPT-4.5 + history (5 runs) ==="
$BASE --models_file config_gpt45.yaml --num_runs 5

echo "=== GPT-4.5 + no_history (5 runs) ==="
$BASE --models_file config_gpt45_no_history.yaml --num_runs 5

echo "=== Gemini 2.5 Pro + history (5 runs) ==="
$BASE --models_file config_gemini25pro.yaml --num_runs 5

echo "=== Gemini 2.5 Pro + no_history (5 runs) ==="
$BASE --models_file config_gemini25pro_no_history.yaml --num_runs 5

echo "=== ALL DONE ==="
