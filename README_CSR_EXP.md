# CSR Experiment Procedure

This document describes how to reproduce the CSR (Cached State Representation) TTFT experiment on AMA-Bench. CSR measures how vLLM's KV-cache prefix caching eliminates the N-token prefill cost for long agentic trajectories — TTFT should scale with the question length M only, not with the trajectory length N+M.

---

## Prerequisites

- Python environment with `requirements.txt` installed
- CUDA GPUs available
- vLLM installed (`pip install vllm`)
- `transformers` installed (used by CSR for accurate token counting)

---

## Step 1 — Download the dataset

```bash
huggingface-cli download AMA-bench/AMA-bench \
  --repo-type dataset --local-dir ./dataset
```

Expected result: `dataset/test/open_end_qa_set.jsonl`

---

## Step 2 — Review configs

**Inference model** — [`configs/qwen3-5-122B.yaml`](configs/qwen3-5-122B.yaml):

```yaml
provider: "vllm"
model: "Qwen/Qwen3.5-122B-A10B-FP8"
enable_thinking: false   # disabled to prevent think tokens consuming max_tokens budget

vllm_host: "localhost"
vllm_port: 8001

vllm_launch:
  gpus: "4,5,6,7"
  max_model_len: 32000
  max_response_len: 4096
  tensor_parallel_size: 4
```

`enable_thinking: false` is important. Qwen3.5 is a thinking model that separates its output into `reasoning_content` (think tokens) and `content` (final answer). With thinking enabled, the model can exhaust `max_tokens` on `<think>` tokens before producing any answer content. Disabling it gives clean, predictable responses and eliminates noise in TTFT measurements from variable-length think budgets.

**Judge model** — [`configs/llm_judge.yaml`](configs/llm_judge.yaml): configured separately (also a vLLM endpoint).

---

## Step 3 — Launch the vLLM inference server

The server **must** be started with `--enable-prefix-caching`. Without it, CSR's warm-up calls have no effect and TTFT will reflect a full N+M prefill instead of M-only.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-122B-A10B-FP8 \
  --host localhost \
  --port 8001 \
  --max-model-len 32000 \
  --tensor-parallel-size 4 \
  --enable-prefix-caching \
  > vllm_server.log 2>&1 &

# Wait for readiness
until curl -sf http://localhost:8001/health; do sleep 2; done && echo "Ready"
```

Launch the judge server separately on its own port (see its config for port/GPU assignment).

---

## Step 4 — Run the CSR experiment

```bash
bash scripts/run_csr.sh
```

Which executes:

```bash
python src/run.py \
  --llm-server vllm \
  --llm-config configs/qwen3-5-122B.yaml \
  --subset openend \
  --method csr \
  --method-config configs/method_configs/csr_config.yaml \
  --test-dir dataset/test \
  --output-dir results/csr \
  --max-concurrency-episodes 1 \
  --max-concurrency-questions-per-episode 1 \
  --judge-config configs/llm_judge.yaml \
  --judge-server vllm \
  --evaluate False
```

**Key parameter choices:**

| Parameter | Value | Reason |
|---|---|---|
| `--max-concurrency-questions-per-episode` | `1` | **Required.** CSR must answer questions sequentially within each episode to preserve the KV-cache prefix chain. Parallel probes would share scheduler resources and inflate/corrupt TTFT measurements. |
| `--max-concurrency-episodes` | `1` | Conservative default. Each episode runs a warm-up loop of N sequential calls; raising this multiplies concurrent vLLM requests. Increase only if the server has spare capacity. |
| `--evaluate` | `False` | Separates answer generation from judge scoring. Run judge evaluation as a separate step (see Step 5) once answers are confirmed correct. |

---

## Step 5 — (Optional) Run LLM-as-judge evaluation

```bash
bash scripts/evaluate.sh \
  --answers-file results/csr/answers_<timestamp>.jsonl \
  --test-file dataset/test/open_end_qa_set.jsonl \
  --judge-config configs/llm_judge.yaml \
  --judge-server vllm
```

---

## Step 6 — Extract TTFT measurements

TTFT records are printed to stdout during the run:

```
[CSR] qa=0  TTFT: 0.375s  static_tokens: 15412  dynamic_tokens: 68
[CSR] qa=1  TTFT: 0.318s  static_tokens: 15412  dynamic_tokens: 97
```

Extract them from the log:

```bash
grep "\[CSR\]" run.log > csr_ttft.txt
```

Each line records:

| Field | Meaning |
|---|---|
| `qa` | 0-based QA index within the episode |
| `TTFT` | Wall-clock time to first token (seconds) |
| `static_tokens` | N — trajectory tokens, fully cached |
| `dynamic_tokens` | M — question tokens, prefilled fresh each query |

The core CSR hypothesis is that TTFT correlates with `dynamic_tokens` (M) and is independent of `static_tokens` (N). Plotting TTFT vs M across episodes verifies this empirically.

---

## Troubleshooting

**`content` is `None`, answers are empty**  
Thinking mode is still active. Confirm `enable_thinking: false` is set in the model config and that the config path in `run_csr.sh` is correct.

**TTFT values are unexpectedly high and scale with N**  
The vLLM server was not started with `--enable-prefix-caching`. Restart with that flag.

**Warning: `Failed to initialize embedding engine`**  
Harmless for CSR — this warning appears when `csr_config.yaml` is read but no `embedding_engine` block is present. CSR does not use embeddings.
