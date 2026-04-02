#!/bin/bash
# AMA Agent latency measurement run script.
# Assumes the vLLM inference server is already running with --enable-prefix-caching.
# Questions within each episode are processed sequentially (required for KV-cache prefix chaining).

# MODEL_CONFIG="configs/qwen3-5-122B.yaml"
MODEL_CONFIG="configs/qwen3-30B.yaml"

JUDGE_CONFIG="configs/llm_judge.yaml"
EVALUATE=True

# Domain filter (optional)
# Set to one or more space-separated values to restrict evaluation to specific domains.
# Available: embodied_ai  game  text2sql  openworld_qa  web  software_engineering
# Example: DOMAIN="game web" bash scripts/run.sh
DOMAIN="embodied_ai"
# DOMAIN="embodied_ai game"

set -e

python src/run.py \
  --llm-server vllm \
  --llm-config "$MODEL_CONFIG" \
  --subset openend \
  --method ama_agent_latency \
  --num-episodes 1 \
  --method-config configs/ama_agent_latency.yaml \
  --test-dir dataset/test \
  --output-dir results/ama_agent_latency \
  --max-concurrency-episodes 1 \
  --max-concurrency-questions-per-episode 1 \
  --judge-config "$JUDGE_CONFIG" \
  --judge-server vllm \
  --evaluate "$EVALUATE" \
  ${DOMAIN:+--domain $DOMAIN}
