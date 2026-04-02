bash scripts/evaluate.sh \
  --answers-file results/ama_agent/answers_Qwen_Qwen3.5-122B-A10B-FP8_openend_ama_agent_embodied_ai_20260402_214445.jsonl \
  --test-file dataset/test/open_end_qa_set.jsonl \
  --judge-config configs/llm_judge.yaml \
  --judge-server vllm