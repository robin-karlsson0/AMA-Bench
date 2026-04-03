#!/bin/bash
# StreamingLLM experiment run script.
# Assumes an OpenAI-compatible inference server (e.g. TensorRT-LLM container) is running.
#
# Unlike CSR, StreamingLLM does not benefit from prefix KV-cache sharing across
# requests; each call prefills the full (sink + rolling window + question) context.
# Questions within each episode are answered in parallel (memory_retrieve is
# stateless after construction, so no sequential ordering is required).
#
# rolling_window_size in configs/method_configs/streaming_llm_config.yaml must be
# set smaller than a typical episode's token count to observe eviction and the
# resulting accuracy degradation on mid-episode QA pairs.

# MODEL_CONFIG="configs/qwen3-5-122B.yaml"
MODEL_CONFIG="configs/qwen3-30B.yaml"

JUDGE_CONFIG="configs/llm_judge.yaml"
EVALUATE=False

# Domain filter (optional)
# Set to one or more space-separated values to restrict evaluation to specific domains.
# Available: embodied_ai  game  text2sql  openworld_qa  web  software_engineering
# DOMAIN="embodied_ai"
DOMAIN="embodied_ai game"

set -e

python src/run.py \
  --llm-server vllm \
  --llm-config "$MODEL_CONFIG" \
  --subset openend \
  --method streaming_llm \
  --method-config configs/method_configs/streaming_llm_config_vanilla.yaml \
  --test-dir dataset/test \
  --output-dir results/streaming_llm_vanilla \
  --max-concurrency-episodes 1 \
  --max-concurrency-questions-per-episode 1 \
  --judge-config "$JUDGE_CONFIG" \
  --judge-server vllm \
  --evaluate "$EVALUATE" \
  ${DOMAIN:+--domain $DOMAIN}

# --num-episodes 2 \