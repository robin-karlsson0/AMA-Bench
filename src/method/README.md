# Memory Methods

Each method under this directory implements the two-stage `BaseMethod` interface:

| Method | File | Strategy |
|---|---|---|
| `longcontext` | `longcontext.py` | Full trajectory as context, optional middle-truncation |
| `bm25` | `bm25.py` | Sparse retrieval over turn blocks |
| `embedding` | `embedding_mem.py` | Dense retrieval over turn blocks |
| `ama_agent` | `ama_agent.py` | Agentic chain-of-thought over memory |
| `csr` | `csr.py` | KV-cache prefix reuse with TTFT measurement |

---

## CSR — Cached State Representation

**File:** `csr.py`  
**Run flag:** `--method csr`

### Motivation

Standard LLM serving prefills the entire prompt from scratch on every request, so TTFT scales with total context length N. For long agentic trajectories this becomes the dominant latency cost. CSR eliminates this by structuring the prompt so the large, stable trajectory history is permanently cached as a KV-cache prefix. Only the small per-query dynamic suffix (the question, ~M tokens) requires fresh prefill, making TTFT ≈ f(M) independent of N.

### Prompt structure

```
T = X_static ⊕ X_task

X_static (cached)
├── X_pre    — task description + three grounding constraints
└── X_chunks — one block per trajectory turn, strictly appended

X_task (fresh each query, appended by the AMA-Bench framework)
└── question + answer-format instructions
```

### Execution flow

```
memory_construction(traj_text, task)
  1. Build X_pre from task string and constraints.
  2. Parse traj_text into x_chunks (one entry per "Step N:" block).
  3. Warm-up loop: for k in 0..N-1, send X_pre ⊕ x_chunks[:k+1]
     to the server with max_tokens=1.  Each call extends the cached
     prefix by one turn without invalidating previous blocks.

memory_retrieve(memory, question)          ← called once per QA pair
  1. Assemble X_static = X_pre ⊕ X_chunks.
  2. Issue streaming 1-token probe of (X_static + question).
     Clock stops on first non-empty chunk → TTFT.
  3. Record (qa_index, ttft, static_tokens, dynamic_tokens) in
     memory.inference_records.
  4. Return X_static; framework appends question + answer format
     and issues full generation (also hits warm cache).
```

### Measured quantities

Each `CSRInferenceRecord` stores:

| Field | Meaning |
|---|---|
| `qa_index` | 0-based QA pair index within the episode |
| `ttft` | Wall-clock time to first token (seconds) |
| `static_tokens` | N — tokens in X_static (cached, zero prefill cost) |
| `dynamic_tokens` | M — tokens in question (must be prefilled fresh) |

The paper's core claim is TTFT ≈ f(M), not f(N+M). Having N and M recorded separately in every record makes this directly verifiable in post-hoc analysis.

### Infrastructure requirements

- vLLM must be started with `--enable-prefix-caching`.
- Questions within an episode are answered **sequentially** (enforced by `MemoryQAInterface.process_episode`). Concurrent probes would share scheduler resources and inflate TTFT measurements.
- The `transformers` tokenizer for the deployed model is loaded lazily on the first probe and cached for the run; accurate token counts depend on it being available.

### Accessing records after a run

`CSRMemory.inference_records` is not surfaced in the output JSONL files. To capture it, instrument `process_episode` or run episodes inside a custom loop that retains the memory object:

```python
memory = interface.memory_construction(trajectory, task)
for i, qa_pair in enumerate(qa_pairs):
    context = method.memory_retrieve(memory, qa_pair["question"])
    # ...

records = memory.inference_records  # List[CSRInferenceRecord]
```

TTFT values are also printed to stdout as:
```
[CSR] qa=0  TTFT: 0.031s  static_tokens: 8412  dynamic_tokens: 47
```
and can be extracted with `grep "\[CSR\]"` from the run log.
