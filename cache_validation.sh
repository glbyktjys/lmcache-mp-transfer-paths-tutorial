#!/usr/bin/env bash
# cache_validation.sh [shm|pickle]
#
# Local cache-hit validation for the LMCache MP server + vLLM on CPU.
#
# Credits: mirrors Phase 3 of
# .buildkite/k3_tests/multiprocess/scripts/run-cpu-e2e-validation.sh,
# adapted for local runs (no install phase; uses the active environment).
#
# Scenario:
#   - LMCache server stays running the entire time
#   - vLLM instance 1: request A -> LMCache store; request A again -> LMCache hit
#   - vLLM restart (instance 2): request A -> LMCache hit (cross-instance)
#   - All three outputs must be identical (bit-exact with temperature=0)
#   - Request B (different prompt) -> cache miss
#
# Requires an activated environment with lmcache and vLLM (CPU) installed;
# see the non-CUDA setup guide.
set -euo pipefail

MODE="${1:-shm}"
case "${MODE}" in
  shm|pickle) ;;
  *) echo "usage: $0 [shm|pickle]"; exit 1 ;;
esac

# --- Configuration (override via environment) ---
# `auto` lets vLLM pick the dtype each CPU backend supports (bf16 on x86,
# fp16 on Apple Silicon, which has no bf16).
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
# Cap memory the way the CI script does: a small reservation fraction plus an
# absolute KV-cache size. Without the absolute cap, the CPU backend sizes the
# KV cache from `total_memory * gpu_memory_utilization`, which fails on hosts
# where that much memory isn't free.
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.1}"
VLLM_CPU_KVCACHE_SPACE="${VLLM_CPU_KVCACHE_SPACE:-1}"  # GiB
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"

MODEL="${MODEL:-facebook/opt-125m}"
VLLM_PORT="${VLLM_PORT:-8000}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
LMCACHE_ZMQ_PORT="${LMCACHE_ZMQ_PORT:-5555}"
LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-2}"
LMCACHE_EVICTION_POLICY="${LMCACHE_EVICTION_POLICY:-LRU}"
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-128}"
LMCACHE_HEALTHCHECK_TIMEOUT="${LMCACHE_HEALTHCHECK_TIMEOUT:-30}"
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-180}"
MAX_TOKENS="${MAX_TOKENS:-50}"
LMCACHE_LOG="${LMCACHE_LOG:-/tmp/cache_validation_${MODE}_lmcache.log}"
VLLM_LOG="${VLLM_LOG:-/tmp/cache_validation_${MODE}_vllm.log}"
PROMPT_FILE_A="/tmp/cache_validation_${MODE}_prompt_a.txt"
PROMPT_FILE_B="/tmp/cache_validation_${MODE}_prompt_b.txt"

# Print the first free localhost port at or above $1.
find_free_port() {
  python3 - "$1" <<'PY'
import socket, sys
port = int(sys.argv[1])
while True:
    try:
        with socket.socket() as s:
            s.bind(("127.0.0.1", port))
        break
    except OSError:
        port += 1
print(port)
PY
}

# Auto-bump any port that is already taken (VS Code extensions are known to
# squat 5555/8080 on developer machines), so a plain `bash cache_validation.sh`
# works without manual port overrides.
for var in VLLM_PORT LMCACHE_HTTP_PORT LMCACHE_ZMQ_PORT; do
  requested="${!var}"
  resolved="$(find_free_port "${requested}")"
  if [ "${resolved}" != "${requested}" ]; then
    echo "Port ${requested} is taken; using ${resolved} for ${var}"
    printf -v "${var}" '%s' "${resolved}"
  fi
done

LMCACHE_URL="http://localhost:${LMCACHE_HTTP_PORT}"
VLLM_URL="http://localhost:${VLLM_PORT}"
LMCACHE_PID=""
VLLM_PID=""

# --- Cleanup and error handling ---

cleanup() {
  set +e
  for pid in "${VLLM_PID}" "${LMCACHE_PID}"; do
    if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null
      wait "${pid}" 2>/dev/null
    fi
  done
  rm -f "${PROMPT_FILE_A}" "${PROMPT_FILE_B}"
}
trap cleanup EXIT INT TERM

print_failure_logs() {
  echo "=== LMCache server log (${LMCACHE_LOG}) ==="
  tail -n 100 "${LMCACHE_LOG}" 2>/dev/null || echo "Log not found"
  echo "=== vLLM log (${VLLM_LOG}) ==="
  tail -n 100 "${VLLM_LOG}" 2>/dev/null || echo "Log not found"
}

on_error() {
  local exit_code=$?
  trap - ERR
  echo "❌ [${MODE}] cache validation FAILED (exit code: ${exit_code})"
  print_failure_logs
  exit "${exit_code}"
}
trap on_error ERR

# --- Helpers ---

# Poll url until the response contains expected (any 2xx when expected is
# empty). Fails early if the process behind it dies during startup.
wait_for_endpoint_contains() {
  local url="$1" timeout="$2" expected="$3" label="$4" pid="$5"
  local response
  for _ in $(seq 1 "${timeout}"); do
    if response="$(curl -fsS "${url}" 2>/dev/null)"; then
      if [ -z "${expected}" ] || echo "${response}" | grep -q "${expected}"; then
        return 0
      fi
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "❌ ${label} exited during startup"
      return 1
    fi
    sleep 1
  done
  echo "❌ ${label} did not become ready within ${timeout}s"
  return 1
}

# Sum a Prometheus counter across label sets from the LMCache /metrics endpoint.
scrape_metric() {
  curl -s "${LMCACHE_URL}/metrics" \
    | awk -v m="$1" '$1 !~ /^#/ && $1 ~ "^"m {s+=$2} END {print s+0}'
}

# Fail the run when a metric delta is below 1.
require_metric_delta() {
  local delta="$1" what="$2"
  if [ "${delta}" -lt 1 ]; then
    echo "❌ ${what} (delta: ${delta})"
    false
  fi
}

# Send a completion request for the prompt in the given file; print the
# generated text. Values reach python via argv (and the prompt via a file),
# so nothing is spliced into code and shell quoting never breaks.
send_completion() {
  curl -fsS "${VLLM_URL}/v1/completions" -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "prompt": open(sys.argv[2]).read(),
    "max_tokens": int(sys.argv[3]),
    "temperature": 0,
}))' "${MODEL}" "$1" "${MAX_TOKENS}")" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['text'])"
}

# Send a completion for <prompt-file>, then poll <metric> until it grows or
# <timeout>s (default 5) pass. Sets COMPLETION_OUTPUT and METRIC_DELTA.
# Returning as soon as the metric moves keeps positive checks fast; the
# negative (miss) check reuses the same mechanism and fails fast on an
# unexpected hit while still waiting out the full window on a clean miss.
send_and_measure() {
  local metric="$1" prompt_file="$2" timeout="${3:-5}"
  local before
  before=$(scrape_metric "${metric}")
  COMPLETION_OUTPUT="$(send_completion "${prompt_file}")"
  METRIC_DELTA=0
  for _ in $(seq 1 $((timeout * 5))); do
    METRIC_DELTA=$(( $(scrape_metric "${metric}") - before ))
    if [ "${METRIC_DELTA}" -ge 1 ]; then
      return 0
    fi
    sleep 0.2
  done
}

start_vllm() {
  echo "Starting vLLM server (log: ${VLLM_LOG})"
  # Exported so multiproc worker children inherit it; without it the CPU
  # backend sizes the KV cache from `total_memory * gpu_memory_utilization`,
  # which fails on hosts where that much memory isn't free (same guard as
  # the CI script).
  export VLLM_CPU_KVCACHE_SPACE
  local kv_cache_bytes=$((VLLM_CPU_KVCACHE_SPACE * 1024 * 1024 * 1024))
  VLLM_TARGET_DEVICE=cpu vllm serve "${MODEL}" \
    --port "${VLLM_PORT}" \
    --dtype "${VLLM_DTYPE}" \
    --disable-hybrid-kv-cache-manager \
    --no-enable-prefix-caching \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --kv-cache-memory-bytes "${kv_cache_bytes}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
    --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://localhost\",\"lmcache.mp.port\":${LMCACHE_ZMQ_PORT}}}" \
    >"${VLLM_LOG}" 2>&1 &
  VLLM_PID=$!
  wait_for_endpoint_contains "${VLLM_URL}/v1/models" "${VLLM_READY_TIMEOUT}" \
    "${MODEL}" "vLLM server" "${VLLM_PID}"
  echo "✅ vLLM server is ready (PID=${VLLM_PID})"
}

stop_vllm() {
  if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "Stopping vLLM (PID=${VLLM_PID})"
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
    VLLM_PID=""
  fi
}

# --- Validation ---

echo "=== Cache Hit Validation (${MODE} transport) ==="

echo "[Step 1] Writing prompt files"
# Prompt A: a non-repetitive story long enough to fill several chunks.
# Implicit string concatenation keeps it readable without injecting newlines.
python3 - >"${PROMPT_FILE_A}" <<'PY'
story = (
    "Once upon a time in a small coastal village, there lived an old "
    "lighthouse keeper named Thomas. Every evening he climbed the one "
    "hundred and thirty-seven steps to the top of the lighthouse to "
    "light the great lamp. The sea was unpredictable in those parts. "
    "Ships from distant lands carried spices, silk, and stories of "
    "places Thomas had never seen. One stormy night in November a "
    "merchant vessel called the Silver Heron appeared on the horizon, "
    "listing dangerously to starboard. Thomas watched through his brass "
    "telescope as the waves crashed against its hull. He knew that if "
    "the ship did not change course within ten minutes it would strike "
    "the jagged rocks known locally as the Devil's Teeth. He grabbed "
    "the flare gun from the cabinet and fired three red flares into the "
    "sky. The captain saw the warning and ordered hard to port. The "
    "ship groaned as it turned, barely clearing the outermost rock. The "
    "next morning the captain rowed ashore to thank Thomas personally. "
    "He brought a gift: a small wooden box holding a compass that always "
    "pointed not north, but toward home. Years later Thomas passed it to "
    "his granddaughter Elena, who became a marine biologist studying the "
    "migration of humpback whales along the Pacific coast. She traveled "
    "from Alaska to Mexico following the pods, documenting their songs "
    "and social behaviors across thousands of miles. One afternoon while "
    "diving near a coral reef off Baja California, Elena discovered "
    "something extraordinary beneath a rocky overhang:"
)
print(story, end="")
PY

# Prompt B: shares no prefix with prompt A — used for the negative (miss) test.
python3 - >"${PROMPT_FILE_B}" <<'PY'
story = (
    "In the year 2147 humanity established its first permanent colony on "
    "Mars. The settlement, named Arcadia, housed three thousand "
    "researchers working to terraform the red planet. Chief botanist Dr. "
    "Yuki Tanaka spent her days in the greenhouse domes cultivating crops "
    "that could thrive in Martian soil. Every morning she recorded the "
    "oxygen levels in her logbook. Today the readings showed"
)
print(story, end="")
PY
echo "✅ Prompt files written"

echo "[Step 2] Starting LMCache server (log: ${LMCACHE_LOG})"
LMCACHE_ARGS=(
  --l1-size-gb "${LMCACHE_L1_SIZE_GB}"
  --eviction-policy "${LMCACHE_EVICTION_POLICY}"
  --chunk-size "${LMCACHE_CHUNK_SIZE}"
  --port "${LMCACHE_ZMQ_PORT}"
  --http-port "${LMCACHE_HTTP_PORT}"
)
if [ "${MODE}" = "pickle" ]; then
  LMCACHE_ARGS+=(--shm-name "")
fi
lmcache server "${LMCACHE_ARGS[@]}" >"${LMCACHE_LOG}" 2>&1 &
LMCACHE_PID=$!
wait_for_endpoint_contains "${LMCACHE_URL}/healthcheck" \
  "${LMCACHE_HEALTHCHECK_TIMEOUT}" "" "LMCache server" "${LMCACHE_PID}"
echo "✅ LMCache server is healthy (PID=${LMCACHE_PID})"
curl -fsS -X POST "${LMCACHE_URL}/metrics/reset" >/dev/null
echo "✅ Metrics reset"

echo "[Step 3] Starting vLLM (instance 1)"
start_vllm

echo "[Step 4] Request A (first) — expecting LMCache store"
send_and_measure lmcache_mp_l1_write_chunks_total "${PROMPT_FILE_A}"
OUTPUT_1="${COMPLETION_OUTPUT}"
require_metric_delta "${METRIC_DELTA}" "No L1 write activity after first request"
echo "✅ Store verified (${METRIC_DELTA} chunks written)"

# Transport mode is logged once the vLLM worker registers with the server.
echo "[Step 5] Verifying transport mode: expecting '${MODE}'"
if ! grep -q "Using ${MODE}" "${LMCACHE_LOG}"; then
  echo "❌ Expected '${MODE}' transport but 'Using ${MODE}' not found in log"
  false
fi
echo "✅ Transport mode confirmed: ${MODE}"

echo "[Step 6] Request A (second) — expecting LMCache hit"
READ_BEFORE=$(scrape_metric lmcache_mp_l1_read_chunks_total)
OUTPUT_2="$(send_completion "${PROMPT_FILE_A}")"
sleep 2
READ_DELTA=$(( $(scrape_metric lmcache_mp_l1_read_chunks_total) - READ_BEFORE ))
require_metric_delta "${READ_DELTA}" "No L1 read activity on second request"
echo "✅ Hit verified on same instance (${READ_DELTA} chunks read)"

echo "[Step 7] Restarting vLLM (instance 2) — expecting cross-instance hit"
stop_vllm
sleep 2
start_vllm
READ_BEFORE=$(scrape_metric lmcache_mp_l1_read_chunks_total)
OUTPUT_3="$(send_completion "${PROMPT_FILE_A}")"
sleep 2
READ_DELTA=$(( $(scrape_metric lmcache_mp_l1_read_chunks_total) - READ_BEFORE ))
require_metric_delta "${READ_DELTA}" "No L1 read activity after vLLM restart"
echo "✅ Cross-instance hit verified (${READ_DELTA} chunks read)"

echo "[Step 8] Verifying output consistency"
if [ "${OUTPUT_1}" != "${OUTPUT_2}" ] || [ "${OUTPUT_1}" != "${OUTPUT_3}" ]; then
  echo "❌ Outputs differ across requests"
  echo "  Output 1: ${OUTPUT_1}"
  echo "  Output 2: ${OUTPUT_2}"
  echo "  Output 3: ${OUTPUT_3}"
  false
fi
echo "✅ All three outputs are identical — cache does not alter inference results"

echo "[Step 9] Request B (different prompt) — expecting cache MISS"
READ_BEFORE=$(scrape_metric lmcache_mp_l1_read_chunks_total)
send_completion "${PROMPT_FILE_B}" >/dev/null
sleep 5
READ_DELTA=$(( $(scrape_metric lmcache_mp_l1_read_chunks_total) - READ_BEFORE ))
if [ "${READ_DELTA}" -gt 0 ]; then
  echo "❌ Unexpected cache hit on a different prompt (delta: ${READ_DELTA})"
  false
fi
echo "✅ Cache miss confirmed for different prompt"

echo "=========================================="
echo "✅ [${MODE}] cache validation PASSED"
echo "=========================================="