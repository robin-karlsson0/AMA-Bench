"""Latency-Tracking AMA-Agent Method.

Measures the estimated online latency of the AMA-Agent memory pipeline
per QA pair::

    Online Latency = avg_construction_time_per_turn + retrieve_ttft

    avg_construction_time_per_turn -- total wall-clock time of
        memory_construction divided by the number of trajectory turns.
    retrieve_ttft -- TTFT of the first LLM call in memory_retrieve,
        measured by streaming up to the first non-empty token.

Each record is printed to stdout and appended to a JSONL file
(mirroring CSRMethod behaviour).
"""
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from src.method.ama_agent import AMAAgentMemory, AMAAgentMethod
from src.method.ama_agent_core.retrieve import memory_retrieve as _do_retrieve


@dataclass
class AMALatencyRecord:
    """Per-QA-pair latency record.

    Attributes
    ----------
    qa_index : int
        Zero-based index of the QA pair within the episode.
    online_latency : float
        avg_construction_time_per_turn + retrieve_ttft (seconds).
    total_construction_time : float
        Total wall-clock seconds for the entire memory_construction call.
    num_turns : int
        Number of trajectory turns processed during construction.
    avg_construction_time_per_turn : float
        total_construction_time / num_turns -- per-turn approximation
        of incremental construction cost.
    retrieve_ttft : float
        Seconds from sending the first retrieval prompt to the first
        non-empty token (pure prefill cost of the retrieve phase).
    retrieve_prompt_tokens : int
        Whitespace-split token count of the first retrieval prompt.
    """

    qa_index: int
    online_latency: float
    total_construction_time: float
    num_turns: int
    avg_construction_time_per_turn: float
    retrieve_ttft: float
    retrieve_prompt_tokens: int


class LatencyTrackingAMAAgent(AMAAgentMethod):
    """AMAAgentMethod subclass that records estimated online latency.

    memory_construction is timed end-to-end; the total wall time is
    divided by the number of trajectory turns to approximate the
    per-turn incremental cost.  The first retrieval call per QA pair
    uses streaming to capture TTFT.  Subsequent retrieval calls fall
    back to the standard client.query path.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        client: Optional[Any] = None,
        embedding_engine: Optional[Any] = None,
        output_file: Optional[str] = None,
    ) -> None:
        """Initialise, inheriting all AMAAgentMethod parameters."""
        super().__init__(
            config_path=config_path,
            client=client,
            embedding_engine=embedding_engine,
        )

        self._total_construction_time: float = 0.0
        self._num_turns: int = 0

        self._in_retrieve: bool = False
        self._retrieve_call_count: int = 0
        self._first_retrieve_ttft: float = 0.0
        self._first_retrieve_prompt_tokens: int = 0

        self._qa_index: int = 0
        self.latency_records: List[AMALatencyRecord] = []

        if output_file is None:
            if config_path:
                cfg = self._load_config(config_path)
                output_file = cfg.get(
                    'output_file',
                    'results/ama_agent_latency/inference_records.jsonl',
                )
            else:
                output_file = (
                    'results/ama_agent_latency/inference_records.jsonl'
                )

        p = Path(output_file)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_file: str = str(p.with_stem(f'{p.stem}_{timestamp}'))

    def _write_record(self, record: AMALatencyRecord) -> None:
        """Append one record to the JSONL output file."""
        try:
            Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_file, 'a') as f:
                json.dump(asdict(record), f)
                f.write('\n')
        except Exception as e:
            print(
                f'[AMA-Latency] Warning: failed to write record to '
                f'{self.output_file}: {e}'
            )

    def _streaming_call(self, prompt: str) -> tuple:
        """Run a streaming LLM call, returning (ttft, wall_time, response).

        ttft      -- seconds to first non-empty token.
        wall_time -- seconds to last token (full generation).
        response  -- full concatenated response text.
        """
        underlying = self.client.client
        extra_body = None
        if getattr(self.client, 'enable_thinking', None) is not None:
            extra_body = {
                'chat_template_kwargs': {
                    'enable_thinking': self.client.enable_thinking,
                }
            }

        create_kwargs: dict = {
            'model': self.client.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'stream': True,
        }
        if extra_body is not None:
            create_kwargs['extra_body'] = extra_body

        t0 = time.perf_counter()
        ttft = 0.0
        chunks: List[str] = []
        with underlying.chat.completions.create(**create_kwargs) as stream:
            for chunk in stream:
                content = (
                    chunk.choices[0].delta.content
                    if chunk.choices
                    else None
                )
                if content is not None and content != '':
                    if ttft == 0.0:
                        ttft = time.perf_counter() - t0
                    chunks.append(content)
        wall_time = time.perf_counter() - t0
        if ttft == 0.0:
            ttft = wall_time
        return ttft, wall_time, ''.join(chunks)

    def _call_llm(self, prompt: str) -> tuple:
        """Route to the correct measurement strategy based on phase.

        - Construction phase: standard client.query (timed at the
          memory_construction level, not per-call).
        - First retrieve call: streaming, record TTFT only.
        - Subsequent retrieve calls: standard client.query.
        """
        if not self._in_retrieve:
            response = self.client.query(
                prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return None, response

        self._retrieve_call_count += 1
        if self._retrieve_call_count == 1:
            ttft, _, response = self._streaming_call(prompt)
            self._first_retrieve_ttft = ttft
            self._first_retrieve_prompt_tokens = len(prompt.split())
            return None, response

        response = self.client.query(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return None, response

    def memory_construction(
        self, traj_text: str, task: str = ''
    ) -> AMAAgentMemory:
        """Build state memory, timing the full call and counting turns."""
        self._qa_index = 0
        self._total_construction_time = 0.0
        self._num_turns = 0
        self.latency_records = []

        t0 = time.perf_counter()
        memory = super().memory_construction(traj_text, task)
        self._total_construction_time = time.perf_counter() - t0
        self._num_turns = (
            len(memory.trajectory) if memory.trajectory else 1
        )

        avg = self._total_construction_time / self._num_turns
        print(
            f'[AMA-Latency] construction done'
            f'  total_time: {self._total_construction_time:.3f}s'
            f'  num_turns: {self._num_turns}'
            f'  avg_per_turn: {avg:.3f}s'
        )
        return memory

    def memory_retrieve(
        self, memory: AMAAgentMemory, question: str
    ) -> str:
        """Retrieve context, measuring TTFT of the first LLM call."""
        self._in_retrieve = True
        self._retrieve_call_count = 0
        self._first_retrieve_ttft = 0.0
        self._first_retrieve_prompt_tokens = 0

        context = _do_retrieve(
            memory=memory.to_dict(),
            question=question,
            call_llm_func=self._call_llm,
            top_k=self.top_k,
        )

        self._in_retrieve = False

        avg = self._total_construction_time / self._num_turns
        online_latency = avg + self._first_retrieve_ttft
        record = AMALatencyRecord(
            qa_index=self._qa_index,
            online_latency=online_latency,
            total_construction_time=self._total_construction_time,
            num_turns=self._num_turns,
            avg_construction_time_per_turn=avg,
            retrieve_ttft=self._first_retrieve_ttft,
            retrieve_prompt_tokens=self._first_retrieve_prompt_tokens,
        )
        self._qa_index += 1
        self.latency_records.append(record)
        self._write_record(record)

        print(
            f'[AMA-Latency] qa={record.qa_index}'
            f'  online_latency: {online_latency:.3f}s'
            f'  (avg_construction/turn: {avg:.3f}s'
            f' + retrieve_ttft: {self._first_retrieve_ttft:.3f}s)'
            f'  retrieve_tokens: {record.retrieve_prompt_tokens}'
        )
        return context

    @property
    def last_incremental_latency(self) -> Optional[float]:
        """Online latency of the most recently completed QA pair."""
        return (
            self.latency_records[-1].online_latency
            if self.latency_records
            else None
        )
