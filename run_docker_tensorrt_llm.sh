docker run --rm -it --ipc host --gpus all \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  nvcr.io/nvidia/tensorrt-llm/release:latest bash