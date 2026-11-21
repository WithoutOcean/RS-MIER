

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/model_path}"
TRAIN_FILE="${TRAIN_FILE:-/train_data_path}"
VAL_FILE="${VAL_FILE:-/val_data_path}"
IMAGE_DIR="${IMAGE_DIR:-/image_dir}"

TRAIN_GPUS="${TRAIN_GPUS:-0,1}"
JUDGE_GPUS="${JUDGE_GPUS:-2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-multi_image_reason}"

JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/judeg_model}"
JUDGE_SERVED_MODEL_NAME="${JUDGE_SERVED_MODEL_NAME:-qwen3-32b-judge}"
JUDGE_BASE_PORT="${JUDGE_BASE_PORT:-8001}"
JUDGE_HOST="${JUDGE_HOST:-127.0.0.1}"

export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${JUDGE_HOST},localhost"
export no_proxy="${no_proxy:+${no_proxy},}${JUDGE_HOST},localhost"
JUDGE_GPU_MEMORY_UTILIZATION="${JUDGE_GPU_MEMORY_UTILIZATION:-0.95}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-2048}"
JUDGE_MAX_NUM_SEQS="${JUDGE_MAX_NUM_SEQS:-128}"
JUDGE_MAX_NUM_BATCHED_TOKENS="${JUDGE_MAX_NUM_BATCHED_TOKENS:-16384}"
JUDGE_STARTUP_TIMEOUT="${JUDGE_STARTUP_TIMEOUT:-600}"
JUDGE_LOG_DIR="${JUDGE_LOG_DIR:-${PROJECT_ROOT}/examples/reward_function/qwen3_32b_judge_logs}"

JUDGE_MODEL="${JUDGE_MODEL:-${JUDGE_SERVED_MODEL_NAME}}"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-120}"
JUDGE_RETRIES="${JUDGE_RETRIES:-2}"
JUDGE_MAX_WORKERS="${JUDGE_MAX_WORKERS:-128}"

ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
MINI_ROLLOUT_BATCH_SIZE="${MINI_ROLLOUT_BATCH_SIZE:-4}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ACTOR_GLOBAL_BATCH_SIZE="${ACTOR_GLOBAL_BATCH_SIZE:-8}"
ACTOR_MICRO_BATCH_SIZE="${ACTOR_MICRO_BATCH_SIZE:-1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
VAL_FREQ="${VAL_FREQ:-300}"
WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/wandb}"

FORMAT_WEIGHT="${FORMAT_WEIGHT:-0.1}"
PERCEPTION_WEIGHT="${PERCEPTION_WEIGHT:-0.4}"
REASONING_WEIGHT="${REASONING_WEIGHT:-0.2}"
ANSWER_WEIGHT="${ANSWER_WEIGHT:-0.3}"

for required_file in "${MODEL_PATH}/config.json" "${JUDGE_MODEL_PATH}/config.json" "${TRAIN_FILE}" "${VAL_FILE}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 1
    fi
done

if [[ ! -d "${IMAGE_DIR}" ]]; then
    echo "image directory not found: ${IMAGE_DIR}" >&2
    exit 1
fi

IFS=',' read -r -a train_devices <<< "${TRAIN_GPUS}"
IFS=',' read -r -a judge_devices <<< "${JUDGE_GPUS}"
N_GPUS="${#train_devices[@]}"

if (( N_GPUS == 0 )); then
    echo "TRAIN_GPUS must contain at least one GPU." >&2
    exit 1
fi

if (( ${#judge_devices[@]} == 0 )); then
    echo "JUDGE_GPUS must contain at least one GPU." >&2
    exit 1
fi

declare -A assigned_devices=()
for device in "${train_devices[@]}"; do
    if [[ ! "${device}" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU id in TRAIN_GPUS: ${device}" >&2
        exit 1
    fi
    if [[ -n "${assigned_devices[${device}]:-}" ]]; then
        echo "GPU ${device} is listed more than once." >&2
        exit 1
    fi
    assigned_devices["${device}"]="training"
done

for device in "${judge_devices[@]}"; do
    if [[ ! "${device}" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU id in JUDGE_GPUS: ${device}" >&2
        exit 1
    fi
    if [[ -n "${assigned_devices[${device}]:-}" ]]; then
        echo "GPU ${device} is already assigned to ${assigned_devices[${device}]} and cannot also run a judge." >&2
        exit 1
    fi
    assigned_devices["${device}"]="judge"
done

judge_urls=()
for index in "${!judge_devices[@]}"; do
    port=$(( JUDGE_BASE_PORT + index ))
    judge_urls+=("http://${JUDGE_HOST}:${port}/v1/chat/completions")
done
JUDGE_URLS="$(IFS=,; echo "${judge_urls[*]}")"

if (( N_GPUS % TENSOR_PARALLEL_SIZE != 0 )); then
    echo "N_GPUS (${N_GPUS}) must be divisible by TENSOR_PARALLEL_SIZE (${TENSOR_PARALLEL_SIZE})." >&2
    exit 1
fi

if (( ROLLOUT_BATCH_SIZE % MINI_ROLLOUT_BATCH_SIZE != 0 )); then
    echo "ROLLOUT_BATCH_SIZE must be divisible by MINI_ROLLOUT_BATCH_SIZE." >&2
    exit 1
fi

if (( MINI_ROLLOUT_BATCH_SIZE % N_GPUS != 0 )); then
    echo "MINI_ROLLOUT_BATCH_SIZE must be divisible by N_GPUS to avoid rollout padding." >&2
    exit 1
fi

if (( ROLLOUT_BATCH_SIZE % ACTOR_GLOBAL_BATCH_SIZE != 0 )); then
    echo "ROLLOUT_BATCH_SIZE must be divisible by ACTOR_GLOBAL_BATCH_SIZE." >&2
    exit 1
fi

actor_batch_per_device=$(( ACTOR_GLOBAL_BATCH_SIZE * ROLLOUT_N / N_GPUS ))
if (( ACTOR_GLOBAL_BATCH_SIZE * ROLLOUT_N % N_GPUS != 0 || actor_batch_per_device % ACTOR_MICRO_BATCH_SIZE != 0 )); then
    echo "Actor batch after rollout expansion must divide evenly across GPUs and micro batches." >&2
    exit 1
fi

# Validate model metadata before the expensive judge initialization.
PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" MODEL_PATH="${MODEL_PATH}" python - <<'MODEL_PREFLIGHT'
import os
from verl.utils.tokenizer import get_tokenizer, get_processor
path = os.environ["MODEL_PATH"]
tokenizer = get_tokenizer(path, local_files_only=True, trust_remote_code=False)
processor = get_processor(path, local_files_only=True, trust_remote_code=False, use_fast=True)
if processor is None:
    raise RuntimeError("Expected a multimodal processor for the Qwen3-VL actor")
print(f"Actor metadata ready: {type(tokenizer).__name__}, {type(processor).__name__}", flush=True)
MODEL_PREFLIGHT

# Start both persistent Qwen3-32B judges before allocating the actor GPUs.
mkdir -p "${JUDGE_LOG_DIR}"
judge_pids=()
judge_logs=()

cleanup_judges() {
    local pid
    for pid in "${judge_pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
    wait "${judge_pids[@]}" 2>/dev/null || true
}
trap cleanup_judges EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_judge() {
    local gpu="$1"
    local port="$2"
    local log_file="$3"

    CUDA_VISIBLE_DEVICES="${gpu}" python -m vllm.entrypoints.openai.api_server \
        --model "${JUDGE_MODEL_PATH}" \
        --served-model-name "${JUDGE_SERVED_MODEL_NAME}" \
        --host "${JUDGE_HOST}" \
        --port "${port}" \
        --tensor-parallel-size 1 \
        --dtype bfloat16 \
        --gpu-memory-utilization "${JUDGE_GPU_MEMORY_UTILIZATION}" \
        --max-model-len "${JUDGE_MAX_MODEL_LEN}" \
        --max-num-seqs "${JUDGE_MAX_NUM_SEQS}" \
        --max-num-batched-tokens "${JUDGE_MAX_NUM_BATCHED_TOKENS}" \
        --enable-prefix-caching >"${log_file}" 2>&1 &
    judge_pids+=("$!")
}

for index in "${!judge_devices[@]}"; do
    gpu="${judge_devices[${index}]}"
    port=$(( JUDGE_BASE_PORT + index ))
    log_file="${JUDGE_LOG_DIR}/judge_gpu_${gpu}_port_${port}.log"
    judge_logs+=("${log_file}")
    start_judge "${gpu}" "${port}" "${log_file}"
done

echo "Training GPUs: ${TRAIN_GPUS}"
echo "Starting Qwen3-32B judge(s) on GPU(s): ${JUDGE_GPUS}"
printf 'Judge log: %s\n' "${judge_logs[@]}"

JUDGE_URLS="${JUDGE_URLS}" \
JUDGE_STARTUP_TIMEOUT="${JUDGE_STARTUP_TIMEOUT}" \
JUDGE_PIDS="$(IFS=,; echo "${judge_pids[*]}")" \
python - <<'PY'
import os
import time

import requests

urls = [url.strip() for url in os.environ["JUDGE_URLS"].split(",") if url.strip()]
if not urls:
    raise SystemExit("JUDGE_URLS does not contain any endpoint.")

session = requests.Session()
session.trust_env = False

health_urls = [url.split("/v1/", maxsplit=1)[0].rstrip("/") + "/health" for url in urls]
pids = [int(pid) for pid in os.environ["JUDGE_PIDS"].split(",")]
deadline = time.monotonic() + float(os.environ["JUDGE_STARTUP_TIMEOUT"])

while True:
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            raise SystemExit(f"Qwen3-32B judge process {pid} exited during startup; check the judge logs.")

    ready = []
    for health_url in health_urls:
        try:
            response = session.get(health_url, timeout=2)
            ready.append(response.ok)
        except requests.RequestException:
            ready.append(False)

    if all(ready):
        print("Both Qwen3-32B judge services are ready.")
        break
    if time.monotonic() >= deadline:
        missing = [url for url, is_ready in zip(health_urls, ready) if not is_ready]
        raise SystemExit(f"Timed out waiting for Qwen3-32B judge service(s): {missing}")
    time.sleep(2)
PY

REWARD_KWARGS="{\"format_weight\":${FORMAT_WEIGHT},\"perception_weight\":${PERCEPTION_WEIGHT},\"reasoning_weight\":${REASONING_WEIGHT},\"answer_weight\":${ANSWER_WEIGHT},\"judge_urls\":\"${JUDGE_URLS}\",\"judge_model\":\"${JUDGE_MODEL}\",\"judge_timeout\":${JUDGE_TIMEOUT},\"judge_retries\":${JUDGE_RETRIES},\"judge_max_workers\":${JUDGE_MAX_WORKERS}}"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-}"
export WANDB_MODE=offline
export WANDB_DIR
mkdir -p "${WANDB_DIR}"

python -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.prompt_key=problem \
    data.answer_key=answer \
    data.image_key=images \
    data.image_dir="${IMAGE_DIR}" \
    data.rollout_batch_size="${ROLLOUT_BATCH_SIZE}" \
    data.mini_rollout_batch_size="${MINI_ROLLOUT_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.format_prompt="${PROJECT_ROOT}/examples/format_prompt/multi_image.jinja" \
    worker.actor.global_batch_size="${ACTOR_GLOBAL_BATCH_SIZE}" \
    worker.actor.micro_batch_size_per_device_for_update="${ACTOR_MICRO_BATCH_SIZE}" \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.model.enable_gradient_checkpointing=true \
    worker.actor.offload.offload_params=true \
    worker.actor.offload.offload_optimizer=true \
    worker.ref.fsdp.enable_cpu_offload=true \
    worker.reward.reward_function="${PROJECT_ROOT}/examples/reward_function/reward_four_part.py:compute_score" \
    worker.reward.reward_function_kwargs="${REWARD_KWARGS}" \
    algorithm.adv_estimator=grpo \
    worker.rollout.n="${ROLLOUT_N}" \
    worker.rollout.limit_images=5 \
    worker.rollout.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    trainer.validation_accuracy_only=true \
    trainer.best_val_metric=accuracy \
    trainer.val_freq="${VAL_FREQ}" \
    trainer.save_freq=600 \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS}"
