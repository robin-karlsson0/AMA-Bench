"""
Cached State Representation (CSR) — AMA-Bench adapter.

CSR eliminates TTFT latency by structuring the LLM prompt so that the
large, slowly-changing trajectory history (X_static) can be fully cached
as a KV-cache prefix on the serving infrastructure, while only the small,
per-step query (X_task) requires fresh prefill computation.

Prompt structure
----------------
  T = X_static ⊕ X_task

  X_static (immutable, cached):
    X_pre    — task description + grounding constraints
    X_chunks — trajectory turns, strictly appended in chronological order

  X_task (dynamic, recomputed each query):
    question injected by the AMA-Bench framework after memory_retrieve returns

AMA-Bench integration
---------------------
memory_construction(traj_text, task)
  Builds X_pre and X_chunks, then replays the trajectory as N sequential
  1-token generation calls to warm the server's prefix KV-cache block-by-block.
  After this step the full X_static is resident in cache.

memory_retrieve(memory, question)
  Issues a streaming 1-token probe of (X_static + question) to measure TTFT
  against the warm cache — only ~M question tokens are prefilled fresh.
  Records (qa_index, ttft, static_tokens, dynamic_tokens) in
  memory.inference_records for post-hoc N/M analysis.
  Returns X_static; the framework appends the question and answer format,
  then issues the full generation call which also hits the warm cache.

Requirements
------------
  - vLLM launched with --enable-prefix-caching (or equivalent on other backends)
  - Single-episode, sequential question answering (enforced in memory_interface)
  - transformers installed for accurate token counting
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from src.method.base_method import BaseMethod


@dataclass
class CSRInferenceRecord:
    """
    Per-query latency record captured during memory_retrieve.

    Separating static_tokens (N) from dynamic_tokens (M) enables direct
    empirical verification of the CSR claim: under full prefix caching,
    TTFT should scale with M alone, independent of N.

    Attributes
    ----------
    qa_index : int
        Zero-based position of this QA pair within the episode.  Because
        CSR forces sequential execution, this equals the append order in
        memory.inference_records.
    ttft : float
        Wall-clock time in seconds from sending the streaming probe request
        to receiving the first non-empty response token.
    static_tokens : int
        Token count of X_static (X_pre ⊕ X_chunks).  This is N — fully
        covered by the warm cache; contributes zero prefill cost at query time.
    dynamic_tokens : int
        Token count of the question string.  This is M — the only tokens
        the server must prefill fresh.  The expected relationship is
        TTFT ≈ f(M), not f(N + M).
    """

    qa_index: int
    ttft: float
    static_tokens: int
    dynamic_tokens: int


@dataclass
class CSRMemory:
    """
    State container produced by memory_construction and consumed by memory_retrieve.

    Attributes
    ----------
    x_pre : str
        Immutable static prefix assembled from task description and grounding
        constraints.  Fixed for the lifetime of the episode.
    x_chunks : List[str]
        Append-only list of trajectory turn blocks in chronological order.
        Each entry corresponds to one ``Step N:`` block from the trajectory.
        Must never be modified after construction — any in-place edit would
        invalidate the server's KV-cache from that position forward.
    inference_records : List[CSRInferenceRecord]
        Grows by one entry per memory_retrieve call (i.e. one per QA pair).
        Preserves insertion order because CSR forces sequential execution.
    """

    x_pre: str
    x_chunks: List[str]
    inference_records: List[CSRInferenceRecord] = field(default_factory=list)


class CSRMethod(BaseMethod):
    """
    AMA-Bench method adapter for the Cached State Representation framework.

    Implements the two-stage BaseMethod interface:
      memory_construction — builds X_static and warms the server KV-cache
      memory_retrieve     — measures TTFT, returns X_static as context

    TTFT measurements are only meaningful when the inference server has
    block-level prefix caching active (e.g. vLLM --enable-prefix-caching).
    Without it the warm-up calls still run correctly but TTFT will reflect
    a full N+M prefill, erasing the CSR advantage.
    """

    _CONSTRAINTS = (
        "(1) Do not invent facts not supported by the provided turns.\n"
        "(2) Keep rationales short and grounded.\n"
        "(3) Use consistent objective names across turns.")

    def __init__(
        self,
        config_path: Optional[str] = None,
        client: Any = None,
        embedding_engine: Any = None,
        output_file: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        config_path : str, optional
            Path to a YAML/JSON config file containing CSR parameters
            (e.g. output_file for incremental record export).
        client : ModelClient, optional
            AMA-Bench ModelClient instance used for warm-up calls and TTFT
            measurement. Must be provided for CSR functionality.
        embedding_engine : Any, optional
            Accepted for registry-kwarg compatibility; not used.
        output_file : str, optional
            Path to a JSON file where inference records will be written
            incrementally (one record per line in JSONL format). If None,
            records are only kept in memory. Can also be specified in config_path.
        """
        # Load config if provided
        if config_path:
            config = self._load_config(config_path)
            output_file = config.get('output_file', output_file)

        # Stamp output_file with current time so each run produces a unique file
        if output_file is not None:
            p = Path(output_file)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = str(p.with_stem(f'{p.stem}_{timestamp}'))

        self.client = client  # ModelClient instance
        self._tokenizer = None  # Loaded lazily on first probe
        self.output_file = output_file  # Optional JSON Lines output file

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_x_pre(self, task: str) -> str:
        """Assemble the immutable static prefix from task string and constraints."""
        parts = []
        if task:
            parts.append(f"# Task\n{task}")
        parts.append(f"# Constraints\n{self._CONSTRAINTS}")
        return "\n\n".join(parts)

    def _parse_chunks(self, traj_text: str) -> List[str]:
        """
        Split the step-formatted trajectory text into individual turn blocks.

        Expected input format (produced by MemoryQAInterface._trajectory_to_text):
            Step 0:
              Action: <action>
              Observation: <observation>
            Step 1:
              ...
        """
        chunks: List[str] = []
        current: List[str] = []
        for line in traj_text.split("\n"):
            if line.startswith("Step ") and current:
                chunks.append("\n".join(current))
                current = []
            current.append(line)
        if current:
            chunks.append("\n".join(current))
        return chunks

    def _assemble_x_static(self, x_pre: str, x_chunks: List[str]) -> str:
        """Concatenate X_pre and X_chunks into the serialised X_static string."""
        if not x_chunks:
            return x_pre
        traj_text = "\n".join(x_chunks)
        return f"{x_pre}\n\n# Agent Trajectory\n{traj_text}"

    def _warm_up_cache(self, x_pre: str, x_chunks: List[str]) -> None:
        """
        Replay the trajectory as N sequential 1-token calls to populate the KV-cache.

        Call k submits X_pre ⊕ X_chunks[:k+1], extending the cached prefix by
        exactly one turn block.  Because every call is a strict prefix extension
        of the previous one, no cached block is ever invalidated.  After N calls
        the entire X_static is resident in the server's prefix cache.

        Implementation note: the underlying SDK client is called directly,
        bypassing ModelClient.query's max_retries=5 exponential-backoff loop.
        A server error during warm-up should surface immediately as an exception
        rather than stalling silently for several minutes on a long trajectory.
        """
        if self.client is None:
            return
        provider = getattr(self.client, "provider", "")
        if provider not in ("vllm", "openai", "deepseek"):
            # Non-streaming providers: fall back to ModelClient.query with
            # retries since warm-up is best-effort for these backends.
            for k in range(len(x_chunks)):
                prompt = self._assemble_x_static(x_pre, x_chunks[:k + 1])
                self.client.query(prompt, max_tokens=1, temperature=0.0)
            return
        underlying = self.client.client
        for k in range(len(x_chunks)):
            prompt = self._assemble_x_static(x_pre, x_chunks[:k + 1])
            underlying.chat.completions.create(
                model=self.client.model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                max_tokens=1,
                temperature=0.0,
            )

    def _get_tokenizer(self) -> Any:
        """
        Return the model tokenizer, loading from HuggingFace Hub on first call.

        Uses client.model (e.g. ``"Qwen/Qwen3-32B"``) as the identifier.
        The instance is cached so the download happens at most once per run.
        Returns None if the client is unavailable; callers fall back to
        whitespace splitting.
        """
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            model_name = getattr(self.client, "model",
                                 "") if self.client else ""
            if model_name:
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        return self._tokenizer

    def _count_tokens(self, text: str) -> int:
        """Token count via the model tokenizer; falls back to whitespace split."""

        tokenizer = self._get_tokenizer()
        if tokenizer is not None:
            return len(tokenizer.encode(text, add_special_tokens=False))
        return len(text.split())

    def _write_record_to_file(self, record: CSRInferenceRecord) -> None:
        """
        Write a single inference record to the output file in JSONL format.

        Each record is written as a single JSON object on its own line.
        If output_file is not set, this is a no-op.
        """
        if self.output_file is None:
            return
        try:
            Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_file, "a") as f:
                json.dump(asdict(record), f)
                f.write("\n")
        except Exception as e:
            print(
                f"[CSR] Warning: failed to write record to {self.output_file}: {e}"
            )

    def _measure_ttft(
        self,
        x_static: str,
        question: str,
        qa_index: int,
    ) -> CSRInferenceRecord:
        """
        Issue a streaming 1-token probe and record TTFT with N/M token breakdown.

        The probe prompt is X_static + question.  Because X_static is fully
        cached after _warm_up_cache, the server only prefills the question
        tokens (M tokens).  The clock runs from the HTTP request until the
        first non-empty streaming chunk arrives.  static_tokens (N) and
        dynamic_tokens (M) are counted separately so the CSR N/M claim —
        that TTFT ≈ f(M) rather than f(N+M) — is directly verifiable from
        the recorded data.

        Returns ttft=0.0 for non-streaming providers (Gemini, Anthropic).
        """
        static_tokens = self._count_tokens(x_static)
        dynamic_tokens = self._count_tokens(question)
        probe_prompt = f"{x_static}\n\n# Question\n{question}"

        if self.client is None:
            return CSRInferenceRecord(
                qa_index=qa_index,
                ttft=0.0,
                static_tokens=static_tokens,
                dynamic_tokens=dynamic_tokens,
            )
        provider = getattr(self.client, "provider", "")
        if provider not in ("vllm", "openai", "deepseek"):
            return CSRInferenceRecord(
                qa_index=qa_index,
                ttft=0.0,
                static_tokens=static_tokens,
                dynamic_tokens=dynamic_tokens,
            )

        underlying = self.client.client

        t0 = time.perf_counter()
        ttft = 0.0
        with underlying.chat.completions.create(
                model=self.client.model,
                messages=[{
                    "role": "user",
                    "content": probe_prompt
                }],
                max_tokens=1,
                temperature=0.0,
                stream=True,
        ) as stream:
            for chunk in stream:
                content = chunk.choices[
                    0].delta.content if chunk.choices else None
                if content is not None and content != "":
                    ttft = time.perf_counter() - t0
                    break
        if ttft == 0.0:
            ttft = time.perf_counter() - t0

        return CSRInferenceRecord(
            qa_index=qa_index,
            ttft=ttft,
            static_tokens=static_tokens,
            dynamic_tokens=dynamic_tokens,
        )

    # ------------------------------------------------------------------
    # BaseMethod interface
    # ------------------------------------------------------------------

    def memory_construction(self, traj_text: str, task: str = "") -> CSRMemory:
        """
        Build CSRMemory and warm the server KV-cache before any QA is issued.

        Called once per episode by MemoryQAInterface.  After this returns,
        every token of X_static is cached server-side.  The N sequential
        warm-up calls simulate the incremental agent-step queries that would
        occur in a live CSR deployment, keeping the experiment conditions
        consistent with the theoretical model.
        """
        x_pre = self._build_x_pre(task)
        x_chunks = self._parse_chunks(traj_text)
        self._warm_up_cache(x_pre, x_chunks)
        return CSRMemory(x_pre=x_pre, x_chunks=x_chunks)

    def memory_retrieve(self, memory: CSRMemory, question: str) -> str:
        """
        Measure TTFT against the warm cache and return X_static as context.

        Called once per QA pair by MemoryQAInterface (sequentially for CSR).
        Issues a streaming 1-token probe to capture TTFT, appends a
        CSRInferenceRecord to memory.inference_records, then returns X_static.
        The framework appends the question and answer-format instructions to
        this string before calling client.query for the full generation; that
        call also benefits from the warm prefix cache.
        """
        x_static = self._assemble_x_static(memory.x_pre, memory.x_chunks)

        # qa_index derived from how many records have already been appended,
        # which is safe because the CSR branch in process_episode is sequential.
        qa_index = len(memory.inference_records)
        record = self._measure_ttft(x_static, question, qa_index)
        memory.inference_records.append(record)
        self._write_record_to_file(record)

        if record.ttft > 0.0:
            print(f"[CSR] qa={record.qa_index}"
                  f"  TTFT: {record.ttft:.3f}s"
                  f"  static_tokens: {record.static_tokens}"
                  f"  dynamic_tokens: {record.dynamic_tokens}")

        return x_static
