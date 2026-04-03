bash scripts/evaluate.sh \
  --answers-file results/streaming_llm/answers_$(ls results/streaming_llm/answers_*.jsonl 2>/dev/null | tail -1 | xargs -I{} basename {}) \
  --test-file dataset/test/open_end_qa_set.jsonl \
  --judge-config configs/llm_judge.yaml \
  --judge-server openai
