#!/bin/bash

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
VOLUME="/data/robin/huggingface"
PORT=8001
MAX_MODEL_LEN=130000
GPU_IDX=1

docker run --gpus all \
    --env CUDA_VISIBLE_DEVICES=$GPU_IDX \
    -p $PORT:8000 \
    -v $VOLUME:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    --ipc=host \
    vllm/vllm-openai:cu130-nightly $MODEL \
    --max-model-len $MAX_MODEL_LEN
