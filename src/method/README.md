# Memory Methods

Each method under this directory implements the two-stage `BaseMethod` interface:

| Method | File | Strategy |
|---|---|---|
| `longcontext` | `longcontext.py` | Full trajectory as context, optional middle-truncation |
| `bm25` | `bm25.py` | Sparse retrieval over turn blocks |
| `embedding` | `embedding_mem.py` | Dense retrieval over turn blocks |
| `ama_agent` | `ama_agent.py` | Agentic chain-of-thought over memory |
| `csr` | `csr.py` | KV-cache prefix reuse with TTFT measurement |
| `streaming_llm` | `streaming_llm.py` | Attention sink + rolling window eviction with TTFT measurement |

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

---

## StreamingLLM — Attention Sink + Rolling Window

**File:** `streaming_llm.py`  
**Run flag:** `--method streaming_llm`

### Motivation

StreamingLLM enables infinite-horizon operation by preventing OOM errors from unbounded KV cache growth. Rather than growing the cache indefinitely or recomputing from scratch on eviction, it permanently pins the first few tokens (the *attention sink*) and retains only the most recent `rolling_window_size` tokens in a sliding window. This is included in AMA-Bench as a baseline to empirically demonstrate the accuracy cost of its fundamental limitation: any trajectory content that falls outside the window is permanently forgotten.

### Memory structure after eviction

```
Full sequence (N tokens, N > sink_size + rolling_window_size)
├── [0 : sink_size]                   — attention sink, permanently retained
└── [N - rolling_window_size : N]     — rolling window, most recent turns only
    ~~~ middle tokens permanently evicted ~~~

x_static = decode(sink_ids) + decode(rolling_ids)
```

When `N ≤ sink_size + rolling_window_size` no eviction occurs and the method is equivalent to `longcontext`.

### Execution flow

```
memory_construction(traj_text, task)
  1. Build X_pre from task string and constraints (identical to CSR).
  2. Tokenize full sequence: x_pre_ids + traj_section_ids.
  3. If total > sink_size + rolling_window_size:
       keep full_ids[:sink_size] ⊕ full_ids[-rolling_window_size:]
       evict everything in between.
  4. Decode surviving token IDs back to x_static.
  5. Print eviction summary (total_tokens, tokens_evicted, eviction_ratio).

memory_retrieve(memory, question)          ← called once per QA pair
  1. Issue streaming 1-token probe of (x_static + question).
     Clock stops on first non-empty chunk → TTFT.
     Note: no prefix KV-cache sharing across requests — the server
     prefills (sink_size + rolling_window_size + dynamic_tokens) tokens
     on every call, unlike CSR which prefills only dynamic_tokens.
  2. Record (qa_index, ttft, static_tokens, dynamic_tokens,
             total_tokens_original, tokens_evicted, eviction_ratio).
  3. Return x_static.
```

### Measured quantities

Each `StreamingLLMInferenceRecord` stores:

| Field | Meaning |
|---|---|
| `qa_index` | 0-based QA pair index within the episode |
| `ttft` | Wall-clock time to first token (seconds) |
| `static_tokens` | Tokens in x_static = sink_size + rolling_window_size (or N if no eviction) |
| `dynamic_tokens` | M — tokens in question (prefilled fresh each call) |
| `total_tokens_original` | N — untruncated episode token count |
| `tokens_evicted` | N − (sink_size + rolling_window_size), 0 if no eviction |
| `eviction_ratio` | tokens_evicted / total_tokens_original |

`eviction_ratio` is the primary accuracy-degradation signal: it quantifies what fraction of the episode the model cannot access when answering questions. QA pairs whose answers depend on evicted mid-episode content will degrade proportionally.

### TTFT interpretation and comparison with CSR

In practice, vLLM's prefix caching causes the first QA probe in each episode to pay the full `static_tokens + dynamic_tokens` prefill cost, while subsequent probes within the same episode hit the cached `x_static` prefix and pay only `dynamic_tokens`. This matches CSR's post-warmup behaviour, making steady-state TTFT comparable between the two methods. The structural difference is in **accuracy**, not latency: CSR retains the full trajectory; StreamingLLM does not.

For a latency comparison that reflects the true per-step cost a real robot would pay (no persistent KV session across requests), use only `qa_index == 0` records from sessions where the server was already warm.

### Infrastructure requirements

- Any OpenAI-compatible inference server (vLLM, TensorRT-LLM, etc.); prefix caching is not required and confers no benefit across episodes since each episode has a distinct `x_static`.
- `rolling_window_size` **must** be set in config; there is no default to prevent silent equivalence with `longcontext`.
- Questions within an episode may be answered in parallel (`--max-concurrency-questions-per-episode > 1` is safe) since `memory_retrieve` is stateless after construction. Use concurrency=1 for clean per-QA TTFT measurements.
- The `transformers` tokenizer is used for accurate token-level slicing; without it the fallback is whitespace splitting, which produces approximate eviction boundaries.

### Accessing records after a run

```python
memory = method.memory_construction(traj_text, task)
for i, qa_pair in enumerate(qa_pairs):
    context = method.memory_retrieve(memory, qa_pair["question"])
    # ...

records = memory.inference_records  # List[StreamingLLMInferenceRecord]
```

TTFT and eviction values are also printed to stdout as:
```
[StreamingLLM] qa=0  TTFT: 0.224s  static_tokens: 517  dynamic_tokens: 60  evicted: 21099 (97.6%)
```
and can be extracted with `grep "\[StreamingLLM\]"` from the run log.
