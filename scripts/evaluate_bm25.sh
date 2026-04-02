bash scripts/evaluate.sh \
  --answers-file results/bm25/answers_Qwen_Qwen3.5-122B-A10B-FP8_openend_bm25_embodied_ai_20260402_145442.jsonl \
  --test-file dataset/test/open_end_qa_set.jsonl \
  --judge-config configs/llm_judge.yaml \
  --judge-server vllm