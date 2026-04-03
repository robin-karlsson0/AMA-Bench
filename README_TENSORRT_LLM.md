# Start docker container
docker run --rm -it --ipc host --gpus all \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  nvcr.io/nvidia/tensorrt-llm/release:latest bash

# Create streaming_config.yaml:
```yaml
kv_cache_config:
  max_attention_window: [1024]
  sink_token_length: 4
  enable_block_reuse: true
```

# Run OpenAI compatible server 
trtllm-serve Qwen/Qwen3-8B \
    --backend pytorch \
    --host 0.0.0.0 \
    --port 8000 \
    --max_batch_size 1 \
    --max_num_tokens 4096 \
    --max_seq_len 2048 \
    --kv_cache_free_gpu_memory_fraction 0.85 \
    --extra_llm_api_options streaming_config.yaml

# Test server
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "messages":[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Where is New York?"}],
        "max_tokens": 1024,
        "temperature": 0.7
    }'