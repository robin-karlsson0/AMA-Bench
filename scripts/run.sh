LLM_SERVER="vllm"    # api or vllm
LLM_CONFIG="${LLM_CONFIG:-configs/qwen3-32B.yaml}"
SUBSET="openend"
TEST_DIR="${TEST_DIR:-dataset/test}"
OUTPUT_DIR="${OUTPUT_DIR:-results/openend}"

# Launch vLLM server
bash scripts/launch_vllm_32B.sh "$LLM_CONFIG"
echo ""
MAX_CONCURRENCY_EPISODES="${MAX_CONCURRENCY_EPISODES:-10}"  # Limit concurrency
MAX_CONCURRENCY_QUESTIONS_PER_EPISODE="${MAX_CONCURRENCY_QUESTIONS_PER_EPISODE:-12}"  # Limit concurrency for questions within an episode
METHOD="${METHOD:-longcontext}"  # Available methods: longcontext (default), bm25, embedding

# LLM-as-Judge configuration
JUDGE_CONFIG="${JUDGE_CONFIG:-configs/llm_judge.yaml}"
JUDGE_SERVER="${JUDGE_SERVER:-api}"
EVALUATE="${EVALUATE:-True}"  # Whether to evaluate answers

# Method-specific configuration (optional)
METHOD_CONFIG="${METHOD_CONFIG:-}"

# Domain filter (optional)
# Set to one or more space-separated values to restrict evaluation to specific domains.
# Available: embodied_ai  game  text2sql  openworld_qa  web  software_engineering
# Example: DOMAIN="game web" bash scripts/run.sh
DOMAIN="${DOMAIN:-}"

# Build arguments
ARGS=(
  --llm-server "$LLM_SERVER"
  --llm-config "$LLM_CONFIG"
  --subset "$SUBSET"
  --method "$METHOD"
  --test-dir "$TEST_DIR"
  --output-dir "$OUTPUT_DIR"
  --max-concurrency-episodes "$MAX_CONCURRENCY_EPISODES"
  --max-concurrency-questions-per-episode "$MAX_CONCURRENCY_QUESTIONS_PER_EPISODE"
  --judge-config "$JUDGE_CONFIG"
  --judge-server "$JUDGE_SERVER"
  --evaluate "$EVALUATE"
)

# Add method config if provided
if [ -n "$METHOD_CONFIG" ]; then
  ARGS+=(--method-config "$METHOD_CONFIG")
fi

# Add domain filter if provided
if [ -n "$DOMAIN" ]; then
  # shellcheck disable=SC2086
  ARGS+=(--domain $DOMAIN)
fi

# Run evaluation with LLM-as-Judge
echo "Running OpenEnd evaluation with method: $METHOD"
echo "LLM-as-Judge: $JUDGE_SERVER (config: $JUDGE_CONFIG)"
echo "Evaluate: $EVALUATE"
[ -n "$DOMAIN" ] && echo "Domain filter: $DOMAIN"
python src/run.py "${ARGS[@]}"
