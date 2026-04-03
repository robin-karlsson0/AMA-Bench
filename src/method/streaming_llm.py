"""
StreamingLLM State Memory — AMA-Bench adapter.

StreamingLLM enables infinite-horizon operation by maintaining a fixed-size
KV cache composed of two segments:
  - Attention sink: the first `sink_size` tokens (default 4) of the full
    sequence, permanently pinned regardless of semantic content.
  - Rolling window: the most recent `rolling_window_size` tokens, sliding
    forward as new observations and actions accumulate.

Any tokens between the sink and the rolling window are permanently evicted
once the total sequence length exceeds sink_size + rolling_window_size.

AMA-Bench integration
---------------------
This module simulates the StreamingLLM eviction algorithm as an offline
text-level truncation applied at memory_construction time.  The resulting
x_static string faithfully represents only the information a StreamingLLM
robot would retain at end-of-episode: the first 4 tokens of its prompt plus
the most recent rolling_window_size tokens of its trajectory.

memory_construction(traj_text, task)
  Builds the full sequence (task prefix + trajectory), applies token-level
  eviction to produce x_static, and records eviction statistics.

memory_retrieve(memory, question)
  Issues a streaming 1-token probe of (x_static + question) to measure TTFT
  against the TensorRT-LLM server.  Because StreamingLLM does not benefit
  from prefix KV-cache sharing across requests, the server must prefill the
  entire (sink + rolling window + question) on every call.
  Records (qa_index, ttft, static_tokens, dynamic_tokens,
           total_tokens_original, tokens_evicted, eviction_ratio) in
  memory.inference_records and appends each record to output_file.
  Returns x_static; the framework appends the question and answer format
  before the full generation call.

Limitations (by design)
-----------------------
  - All trajectory tokens evicted from the middle are permanently forgotten.
  - The task description (x_pre tokens beyond position 4) is the first
    content evicted as episodes grow.
  - No prefix KV-cache benefit: TTFT scales with sink_size + rolling_window_size
    rather than with the question length alone (contrast with CSR).
  - rolling_window_size must be set explicitly in config; there is no default
    to prevent accidental equivalence with full-context methods.

Requirements
------------
  - An OpenAI-compatible inference server (e.g. TensorRT-LLM container).
  - transformers installed for accurate token counting and ID-level slicing.
  - rolling_window_size set smaller than the typical episode token count for
    eviction to occur; otherwise the method degrades to LongContext behaviour.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from src.method.base_method import BaseMethod


@dataclass
class StreamingLLMInferenceRecord:
    qa_index: int
    ttft: float
    static_tokens: int
    dynamic_tokens: int
    total_tokens_original: int
    tokens_evicted: int
    eviction_ratio: float


@dataclass
class StreamingLLMMemory:
    x_static: str
    total_tokens_original: int
    tokens_evicted: int
    eviction_ratio: float
    inference_records: List[StreamingLLMInferenceRecord] = field(
        default_factory=list)


class StreamingLLMMethod(BaseMethod):
    """
    AMA-Bench method adapter for the StreamingLLM framework.

    Implements the two-stage BaseMethod interface:
      memory_construction — applies sink + rolling window eviction to produce
                            a truncated x_static from the full trajectory
      memory_retrieve     — measures TTFT via a streaming probe, returns x_static

    TTFT here reflects the cost of prefilling (sink + rolling window + question)
    tokens, since StreamingLLM does not share a prefix KV cache across requests.
    This is expected to be higher than CSR's M-token prefill cost, providing an
    empirical latency comparison alongside the accuracy degradation signal.
    """

    _CONSTRAINTS = (
        "(1) Do not invent facts not supported by the provided turns.\n"
        "(2) Keep rationales short and grounded.\n"
        "(3) Use consistent objective names across turns.")

    def __init__(self,
                 config_path=None,
                 client=None,
                 embedding_engine=None,
                 output_file=None):
        if config_path:
            config = self._load_config(config_path)
            rolling = config.get("rolling_window_size")
            if rolling is None:
                raise ValueError(
                    "[StreamingLLM] 'rolling_window_size' is required in config. "
                    "Set it smaller than a typical episode's token count to observe eviction. "
                    "There is no default to prevent silent equivalence with full-context methods."
                )
            self.rolling_window_size = int(rolling)
            self.sink_size = int(config.get("sink_size", 4))
            self.temperature = float(config.get("temperature", 0.0))
            self.seed = config.get("seed", None)
            output_file = config.get("output_file", output_file)
        else:
            raise ValueError("[StreamingLLM] config_path is required. "
                             "Create a config with 'rolling_window_size' set.")

        if output_file is not None:
            p = Path(output_file)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = str(p.with_stem(f"{p.stem}_{timestamp}"))

        self.client = client
        self._tokenizer = None
        self.output_file = output_file

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_x_pre(self, task: str) -> str:
        parts = []
        if task:
            parts.append(f"# Task\n{task}")
        parts.append(f"# Constraints\n{self._CONSTRAINTS}")
        return "\n\n".join(parts)

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                model_name = getattr(self.client, "model",
                                     "") if self.client else ""
                if model_name:
                    self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            except (ImportError, Exception):
                pass
        return self._tokenizer

    def _get_token_ids(self, text: str) -> list:
        """Return a list of token IDs for text; falls back to word list."""
        tokenizer = self._get_tokenizer()
        if tokenizer is not None:
            return tokenizer.encode(text, add_special_tokens=False)
        return text.split()

    def _decode_ids(self, ids: list) -> str:
        """Decode a list of token IDs back to text; falls back to joining words."""
        tokenizer = self._get_tokenizer()
        if tokenizer is not None:
            return tokenizer.decode(ids, skip_special_tokens=True)
        return " ".join(ids)

    def _count_tokens(self, text: str) -> int:
        tokenizer = self._get_tokenizer()
        if tokenizer is not None:
            return len(tokenizer.encode(text, add_special_tokens=False))
        return len(text.split())

    def _apply_streaming_eviction(self, x_pre: str, traj_text: str):
        """
        Apply the StreamingLLM eviction algorithm at token level.

        The full sequence is: x_pre_ids + traj_section_ids
        If total > sink_size + rolling_window_size:
            keep full_ids[:sink_size] (attention sink)
            keep full_ids[-rolling_window_size:] (rolling window)
            evict everything in between

        Returns:
            (x_static, total_tokens, tokens_evicted, eviction_ratio)
        """
        x_pre_ids = self._get_token_ids(x_pre)
        traj_section = f"# Agent Trajectory\n{traj_text}" if traj_text else ""
        traj_ids = self._get_token_ids(traj_section) if traj_section else []

        full_ids = list(x_pre_ids) + list(traj_ids)
        total = len(full_ids)
        max_len = self.sink_size + self.rolling_window_size

        if total <= max_len:
            x_static = self._decode_ids(full_ids)
            tokens_evicted = 0
        else:
            sink_ids = full_ids[:self.sink_size]
            rolling_ids = full_ids[total - self.rolling_window_size:]
            # Decode each segment separately so the decoder handles BOS/EOS
            # boundaries correctly, then join with a single newline.
            x_static = self._decode_ids(sink_ids) + "\n" + self._decode_ids(
                rolling_ids)
            tokens_evicted = total - max_len

        eviction_ratio = tokens_evicted / total if total > 0 else 0.0
        return x_static, total, tokens_evicted, eviction_ratio

    def _measure_ttft(
        self,
        x_static: str,
        question: str,
        qa_index: int,
        memory: "StreamingLLMMemory",
    ) -> StreamingLLMInferenceRecord:
        """
        Measure TTFT by issuing a streaming 1-token probe of (x_static + question).

        Unlike CSR — where only M question tokens require fresh prefill — here
        the server must prefill the entire (sink + rolling window + question)
        context because no shared prefix KV cache exists across requests.
        """
        static_tokens = self._count_tokens(x_static)
        dynamic_tokens = self._count_tokens(question)
        probe_prompt = f"{x_static}\n\n# Question\n{question}"

        if self.client is None:
            return StreamingLLMInferenceRecord(
                qa_index=qa_index,
                ttft=0.0,
                static_tokens=static_tokens,
                dynamic_tokens=dynamic_tokens,
                total_tokens_original=memory.total_tokens_original,
                tokens_evicted=memory.tokens_evicted,
                eviction_ratio=memory.eviction_ratio,
            )

        provider = getattr(self.client, "provider", "")
        if provider not in ("vllm", "openai", "deepseek"):
            return StreamingLLMInferenceRecord(
                qa_index=qa_index,
                ttft=0.0,
                static_tokens=static_tokens,
                dynamic_tokens=dynamic_tokens,
                total_tokens_original=memory.total_tokens_original,
                tokens_evicted=memory.tokens_evicted,
                eviction_ratio=memory.eviction_ratio,
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
                temperature=self.temperature,
                **(({"seed": self.seed}) if self.seed is not None else {}),
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

        return StreamingLLMInferenceRecord(
            qa_index=qa_index,
            ttft=ttft,
            static_tokens=static_tokens,
            dynamic_tokens=dynamic_tokens,
            total_tokens_original=memory.total_tokens_original,
            tokens_evicted=memory.tokens_evicted,
            eviction_ratio=memory.eviction_ratio,
        )

    def _write_record_to_file(self,
                              record: StreamingLLMInferenceRecord) -> None:
        if self.output_file is None:
            return
        try:
            Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_file, "a") as f:
                json.dump(asdict(record), f)
                f.write("\n")
        except Exception as e:
            print(
                f"[StreamingLLM] Warning: failed to write record to {self.output_file}: {e}"
            )

    # ------------------------------------------------------------------
    # BaseMethod interface
    # ------------------------------------------------------------------

    def memory_construction(self,
                            traj_text: str,
                            task: str = "") -> StreamingLLMMemory:
        x_pre = self._build_x_pre(task)
        x_static, total, tokens_evicted, eviction_ratio = self._apply_streaming_eviction(
            x_pre, traj_text)
        print(
            f"[StreamingLLM] total_tokens: {total}"
            f"  evicted: {tokens_evicted}"
            f"  eviction_ratio: {eviction_ratio:.1%}"
            f"  window: sink={self.sink_size} + rolling={self.rolling_window_size}"
        )
        return StreamingLLMMemory(
            x_static=x_static,
            total_tokens_original=total,
            tokens_evicted=tokens_evicted,
            eviction_ratio=eviction_ratio,
        )

    def memory_retrieve(self, memory: StreamingLLMMemory,
                        question: str) -> str:
        qa_index = len(memory.inference_records)
        record = self._measure_ttft(memory.x_static, question, qa_index,
                                    memory)
        memory.inference_records.append(record)
        self._write_record_to_file(record)
        if record.ttft > 0.0:
            print(
                f"[StreamingLLM] qa={record.qa_index}"
                f"  TTFT: {record.ttft:.3f}s"
                f"  static_tokens: {record.static_tokens}"
                f"  dynamic_tokens: {record.dynamic_tokens}"
                f"  evicted: {record.tokens_evicted} ({record.eviction_ratio:.1%})"
            )
        return memory.x_static
