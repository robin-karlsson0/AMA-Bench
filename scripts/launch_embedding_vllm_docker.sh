#!/bin/bash

MODEL=Qwen/Qwen3-Embedding-4B
VOLUME=/home/ubuntu/synchro-gripper/huggingface
GPU_ID='"device=0"'
PORT=8003

# docker run --runtime nvidia --gpus $GPU_ID \
docker run --gpus $GPU_ID \
    -p $PORT:8000 \
    -v $VOLUME:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model $MODEL \
    --task embed \
    --gpu_memory_utilization 0.85 \
