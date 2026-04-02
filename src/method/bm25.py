# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
BM25 Method - Uses BM25 retrieval for memory construction and retrieval
"""

import time
from dataclasses import dataclass
from typing import Any, List

from rank_bm25 import BM25Okapi

from src.method.base_method import BaseMethod


@dataclass
class BM25InferenceRecord:
    """
    Per-query latency record captured during memory_retrieve.

    Unlike CSR, BM25 does not benefit from prefix caching: the retrieved
    context differs per query, so the server must prefill
    (retrieved_tokens + dynamic_tokens) fresh on every call.  Comparing
    BM25 TTFT (≈ f(retrieved_tokens + dynamic_tokens)) against CSR TTFT
    (≈ f(dynamic_tokens)) directly quantifies the caching advantage.

    Attributes
    ----------
    qa_index : int
        Zero-based position of this QA pair within the episode.
    ttft : float
        Wall-clock seconds from sending the streaming probe to receiving the
        first non-empty token.
    retrieved_tokens : int
        Token count of the BM25-retrieved context (top-k documents).  This is
        the portion that CSR avoids prefilling via caching; here it is always
        prefilled fresh.
    dynamic_tokens : int
        Token count of the question string — the irreducible per-query cost
        shared with CSR.
    """

    qa_index: int
    ttft: float
    retrieved_tokens: int
    dynamic_tokens: int


class BM25Memory:
    """Memory object for BM25 method"""

    def __init__(self, documents: List[str], bm25_index: BM25Okapi,
                 corpus_tokens: List[List[str]]):
        self.documents = documents
        self.bm25_index = bm25_index
        self.corpus_tokens = corpus_tokens
        self.inference_records: List[BM25InferenceRecord] = []


class BM25Method(BaseMethod):
    """
    BM25-based memory method.

    Uses BM25 (Best Matching 25) ranking function to retrieve relevant trajectory segments.
    """

    def __init__(
        self,
        top_k: int = 5,
        config_path: str = None,
        embedding_engine: Any = None,
        client: Any = None,
    ):
        """
        Initialize BM25 method.

        Args:
            top_k: Number of top documents to retrieve for each question
            config_path: Path to configuration file (optional)
            embedding_engine: Optional embedding engine (not used by BM25, for compatibility)
            client: ModelClient instance used for TTFT measurement (optional).
                    When provided, memory_retrieve issues a streaming 1-token
                    probe after retrieval to record latency comparable to CSR.
        """

        # Load config if provided
        if config_path:
            config = self._load_config(config_path)
            top_k = config.get('top_k', top_k)

        self.top_k = top_k
        self.embedding_engine = embedding_engine  # Not used, for compatibility
        self.client = client
        self._tokenizer = None  # Loaded lazily on first probe

    def _get_tokenizer(self) -> Any:
        """Return the model tokenizer, loading from HuggingFace Hub on first call."""
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

    def _measure_ttft(
        self,
        retrieved_context: str,
        question: str,
        qa_index: int,
        t0: float = 0.0,
    ) -> BM25InferenceRecord:
        """
        Issue a streaming 1-token probe and record TTFT.

        t0 must be captured by the caller before the BM25 retrieval step so
        that retrieval latency is included in the measurement.  The probe
        prompt is retrieved_context + question.  Unlike CSR, BM25 provides no
        KV-cache prefix: the server must prefill every token of
        retrieved_context fresh for each query, so
        TTFT ≈ retrieval_time + f(retrieved_tokens + dynamic_tokens).
        This is the baseline against which CSR's caching advantage is measured.

        Returns ttft=0.0 when no streaming-capable client is available.
        """
        retrieved_tokens = self._count_tokens(retrieved_context)
        dynamic_tokens = self._count_tokens(question)
        probe_prompt = f"{retrieved_context}\n\n# Question\n{question}"

        if self.client is None:
            return BM25InferenceRecord(
                qa_index=qa_index,
                ttft=0.0,
                retrieved_tokens=retrieved_tokens,
                dynamic_tokens=dynamic_tokens,
            )
        provider = getattr(self.client, "provider", "")
        if provider not in ("vllm", "openai", "deepseek"):
            return BM25InferenceRecord(
                qa_index=qa_index,
                ttft=0.0,
                retrieved_tokens=retrieved_tokens,
                dynamic_tokens=dynamic_tokens,
            )

        underlying = self.client.client
        if t0 == 0.0:
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
                content = chunk.choices[0].delta.content if chunk.choices else None
                if content is not None and content != "":
                    ttft = time.perf_counter() - t0
                    break
        if ttft == 0.0:
            ttft = time.perf_counter() - t0

        return BM25InferenceRecord(
            qa_index=qa_index,
            ttft=ttft,
            retrieved_tokens=retrieved_tokens,
            dynamic_tokens=dynamic_tokens,
        )

    def memory_construction(self,
                            traj_text: str,
                            task: str = "") -> BM25Memory:
        """
        Build BM25 index from trajectory text.

        Args:
            traj_text: String-formatted trajectory text
            task: Task description (optional, not used in BM25)

        Returns:
            BM25Memory object containing the index and documents
        """
        # Split trajectory into documents (one per turn)
        # Each turn is separated by double newline
        documents = []
        lines = traj_text.split('\n')

        current_turn = []
        for line in lines:
            if line.strip().startswith('Turn ') or line.strip().startswith(
                    'Step '):
                if current_turn:
                    documents.append('\n'.join(current_turn))
                    current_turn = []
            current_turn.append(line)

        # Add the last turn
        if current_turn:
            documents.append('\n'.join(current_turn))

        # If no turns found, treat entire text as one document
        if not documents:
            documents = [traj_text]

        # Tokenize documents for BM25 (simple whitespace tokenization)
        corpus_tokens = [doc.lower().split() for doc in documents]

        # Build BM25 index
        bm25_index = BM25Okapi(corpus_tokens)

        return BM25Memory(documents, bm25_index, corpus_tokens)

    def memory_retrieve(self, memory: BM25Memory, question: str) -> str:
        """
        Retrieve relevant documents using BM25.

        Args:
            memory: BM25Memory object
            question: Question to retrieve information for

        Returns:
            Retrieved context as string (top-k documents concatenated)
        """
        if not isinstance(memory, BM25Memory):
            raise ValueError("Memory must be a BM25Memory object")

        # Start the clock before retrieval so that BM25 scoring and document
        # fetching are included in the TTFT measurement.  In a real deployment
        # the retrieval step is part of the critical path from question received
        # to first answer token, unlike CSR where no retrieval is needed.
        t0 = time.perf_counter()

        # Tokenize question
        query_tokens = question.lower().split()

        # Retrieve top-k documents using BM25
        scores = memory.bm25_index.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)),
                             key=lambda i: scores[i],
                             reverse=True)[:self.top_k]

        # Get top documents
        retrieved_docs = [memory.documents[i] for i in top_indices]

        # Concatenate retrieved documents
        retrieved_context = "\n\n".join(retrieved_docs)

        # Measure TTFT: includes retrieval time + prefill of retrieved context
        # + question.  Because BM25 has no prefix caching, the server must
        # prefill all retrieved_tokens + dynamic_tokens fresh — the baseline
        # cost that CSR eliminates for the static portion via KV-cache reuse.
        qa_index = len(memory.inference_records)
        record = self._measure_ttft(retrieved_context, question, qa_index, t0)
        memory.inference_records.append(record)

        if record.ttft > 0.0:
            print(f"[BM25] qa={record.qa_index}"
                  f"  TTFT: {record.ttft:.3f}s"
                  f"  retrieved_tokens: {record.retrieved_tokens}"
                  f"  dynamic_tokens: {record.dynamic_tokens}")

        return retrieved_context
