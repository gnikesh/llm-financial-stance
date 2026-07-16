#!/bin/bash
# ==============================================================================
# Run All zero, few, and CoT Experiments from the Paper
# ==============================================================================
#
# This script runs all zero, few, and CoT experiments described in the paper:
#   "Evaluating Large Language Models for Stance Detection on Financial Targets
#    from SEC Filing Reports and Earnings Call Transcripts"
#
# Experiments:
#   1. Zero-shot (with and without CoT) x 3 context modes x N models
#   2. Few-shot x 3 context modes x 2 sampling methods x N models
#   3. Chain-of-Thought x 3 context modes x 2 sampling methods x N models
#   4. Few-shot with CoT x 3 context modes x 2 sampling methods x N models
#
# Resume support:
#   Each Python script checks for existing output files and skips completed
#   experiments. You can safely re-run this script after a crash.
#
# Usage:
#   bash run_all_experiments.sh           # Run everything
#   bash run_all_experiments.sh zero-shot # Run only zero-shot experiments
#   bash run_all_experiments.sh few-shot  # Run only few-shot experiments
#   bash run_all_experiments.sh cot       # Run only chain-of-thought experiments
#
# Prerequisites:
#   - NAUTILUS_API_KEY set for Nautilus-hosted models
#   - OPENAI_API_KEY set for OpenAI-hosted models
#   - Local/remote OpenAI-compatible servers running for mapped local models
#
# ==============================================================================

# Go to the experiments directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- Configuration ----------
# Models used in the paper
# gemma, gpt-oss, qwen models → Nautilus API (NAUTILUS_API_KEY)
# gpt-4.1-mini-2025-04-14    → OpenAI API (OPENAI_API_KEY)
# llama3                     → LLAMA3_URL
# mistral-small-3.2          → MISTRAL_URL
MODELS=("llama3" "gemma" "gpt-4.1-mini-2025-04-14" "gpt-oss" "mistral-small-3.2")

# OpenAI-compatible LLM endpoints. Override these with environment variables.
NAUTILUS_URL="${NAUTILUS_URL:-https://ellm.nrp-nautilus.io/v1}"
LLAMA3_URL="${LLAMA3_URL:-http://localhost:8005/v1}"
MISTRAL_URL="${MISTRAL_URL:-http://localhost:8006/v1}"

# Context modes
CONTEXT_MODES=("no-transcript" "summarized" "full")

# Few-shot sampling strategies
SAMPLING_METHODS=("random" "most_similar")

# Number of runs per experiment (paper uses 3)
NUM_RUNS=3

# Few-shot k values (paper uses k=0,1,5,10)
K_VALUES="0 1 5 10"

# Number of parallel workers per model (adjust based on API rate limits)
NUM_WORKERS=16

# Embedding backend ('nautilus' uses NAUTILUS_API_KEY, 'local' uses EMBEDDING_URL)
# EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-nautilus}"
# EMBEDDING_URL="${EMBEDDING_URL:-$NAUTILUS_URL}"

EMBEDDING_BACKEND="local"
EMBEDDING_URL="http://localhost:8099"


# Ensure Python output is not buffered (helps tqdm render promptly)
export PYTHONUNBUFFERED=1

# ---------- Logging ----------
RESULTS_ROOT="../results"
LOG_DIR="$RESULTS_ROOT/logs"
mkdir -p "$LOG_DIR"

# ---------- Helper ----------
log() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "  $(date)"
    echo "============================================================"
    echo ""
}

# Populate CLIENT_ARGS for the selected model. OpenAI models use the default
# client, while other models receive their OpenAI-compatible endpoint details.
configure_client() {
    local model="$1"
    CLIENT_ARGS=()

    case "$model" in
        gpt-4.1*)
            # Default OpenAI client: OPENAI_API_KEY and standard endpoint.
            ;;
        llama3*)
            CLIENT_ARGS=(--client-base-url "$LLAMA3_URL" --client-api-key "${LLAMA3_API_KEY:-not-needed}")
            ;;
        mistral-small-3.2*)
            CLIENT_ARGS=(--client-base-url "$MISTRAL_URL" --client-api-key "${MISTRAL_API_KEY:-not-needed}")
            ;;
        gemma*|gpt-oss|qwen*)
            if [ -z "${NAUTILUS_API_KEY:-}" ]; then
                echo "  [FAILED] NAUTILUS_API_KEY is required for model: $model"
                return 1
            fi
            CLIENT_ARGS=(--client-base-url "$NAUTILUS_URL" --client-api-key "$NAUTILUS_API_KEY")
            ;;
        *)
            echo "  [FAILED] No client endpoint mapping for model: $model"
            return 1
            ;;
    esac
}

# The embedding client uses different URL flags for Nautilus and local modes.
if [ "$EMBEDDING_BACKEND" = "nautilus" ]; then
    EMBEDDING_ARGS=(--embedding_backend nautilus --nautilus_url "$EMBEDDING_URL")
else
    EMBEDDING_ARGS=(--embedding_backend local --embedding_url "$EMBEDDING_URL")
fi

# Run a python command with stdout going to a log file and stderr (tqdm) on terminal.
# Usage: run_model <log_file> <command> [args...]
run_model() {
    local log_file="$1"
    shift
    echo "  Running: $1 ... (log: $log_file)"
    # stdout → log file, stderr (tqdm progress bars) → terminal
    "$@" > "$log_file"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "  [FAILED] exit code $rc — see $log_file"
    fi
    return $rc
}

# Merge per-model summary CSVs into one all_results_summary.csv
# Usage: merge_summaries <results_dir>
# Finds all all_results_summary_*.csv in the dir, concatenates them (keeping
# the header from the first file only), and writes all_results_summary.csv.
merge_summaries() {
    local dir="$1"
    local merged="$dir/all_results_summary.csv"
    local parts=("$dir"/all_results_summary_*.csv)

    # Check if any per-model files exist
    if [ ! -e "${parts[0]}" ]; then
        return
    fi

    # Take header from the first file, then all data rows from every file
    head -n 1 "${parts[0]}" > "$merged"
    for f in "${parts[@]}"; do
        tail -n +2 "$f" >> "$merged"
    done

    echo "  Merged ${#parts[@]} per-model summaries → $merged"
}

# ---------- Zero-Shot Experiments ----------
run_zero_shot() {
    log "ZERO-SHOT EXPERIMENTS (without CoT)"

    local failed=0
    for mode in "${CONTEXT_MODES[@]}"; do
        for model in "${MODELS[@]}"; do
            if ! configure_client "$model"; then
                failed=$((failed + 1))
                continue
            fi
            LOG_FILE="$LOG_DIR/zero-shot_${mode}_${model}_no-cot.log"
            run_model "$LOG_FILE" \
                python 1_zero-shot-experiment.py \
                    --mode "$mode" \
                    --model "$model" \
                    "${CLIENT_ARGS[@]}" \
                    --num-runs "$NUM_RUNS" \
                    --num-workers "$NUM_WORKERS" \
                    --yes \
                || failed=$((failed + 1))
        done
    done

    [ "$failed" -gt 0 ] && echo "[WARNING] Zero-shot (no CoT): $failed run(s) failed. Check $LOG_DIR/" \
                         || echo "[OK] Zero-shot (no CoT): All models completed successfully."

    # Merge per-model summaries for each results dir
    for dir in "$RESULTS_ROOT"/zero-shot/*/; do
        merge_summaries "$dir"
    done

    log "ZERO-SHOT EXPERIMENTS (with CoT)"

    failed=0
    for mode in "${CONTEXT_MODES[@]}"; do
        for model in "${MODELS[@]}"; do
            if ! configure_client "$model"; then
                failed=$((failed + 1))
                continue
            fi
            LOG_FILE="$LOG_DIR/zero-shot_${mode}_${model}_cot.log"
            run_model "$LOG_FILE" \
                python 1_zero-shot-experiment.py \
                    --mode "$mode" \
                    --model "$model" \
                    "${CLIENT_ARGS[@]}" \
                    --num-runs "$NUM_RUNS" \
                    --num-workers "$NUM_WORKERS" \
                    --use-cot \
                    --yes \
                || failed=$((failed + 1))
        done
    done

    [ "$failed" -gt 0 ] && echo "[WARNING] Zero-shot (CoT): $failed run(s) failed. Check $LOG_DIR/" \
                         || echo "[OK] Zero-shot (CoT): All models completed successfully."

    # Merge again (CoT results go to different dirs)
    for dir in "$RESULTS_ROOT"/zero-shot/*/; do
        merge_summaries "$dir"
    done
}

# ---------- Few-Shot Experiments ----------
run_few_shot() {
    log "FEW-SHOT EXPERIMENTS (without CoT)"

    local failed=0
    for mode in No Full Summarized; do
        for sampling in "${SAMPLING_METHODS[@]}"; do
            for model in "${MODELS[@]}"; do
                if ! configure_client "$model"; then
                    failed=$((failed + 1))
                    continue
                fi
                LOG_FILE="$LOG_DIR/few-shot_${mode}_${sampling}_${model}.log"
                run_model "$LOG_FILE" \
                    python 2_few-shot-experiment.py \
                        --model "$model" \
                        "${CLIENT_ARGS[@]}" \
                        --transcripts "$mode" \
                        --few_shot_sampling "$sampling" \
                        --k_values $K_VALUES \
                        --num_runs "$NUM_RUNS" \
                        --num_workers "$NUM_WORKERS" \
                        "${EMBEDDING_ARGS[@]}" \
                        --yes \
                    || failed=$((failed + 1))
            done
        done
    done

    [ "$failed" -gt 0 ] && echo "[WARNING] Few-shot: $failed run(s) failed. Check $LOG_DIR/" \
                         || echo "[OK] Few-shot: All models completed successfully."

    # Merge per-model summaries
    for dir in "$RESULTS_ROOT"/few-shot/*/; do
        merge_summaries "$dir"
    done
}

# ---------- Chain-of-Thought Experiments ----------
run_cot() {
    log "CHAIN-OF-THOUGHT (CoT) EXPERIMENTS"

    local failed=0
    for mode in No Full Summarized; do
        for sampling in "${SAMPLING_METHODS[@]}"; do
            for model in "${MODELS[@]}"; do
                if ! configure_client "$model"; then
                    failed=$((failed + 1))
                    continue
                fi
                LOG_FILE="$LOG_DIR/cot_${mode}_${sampling}_${model}.log"
                run_model "$LOG_FILE" \
                    python 3_chain-of-thoughts-experiment.py \
                        --model "$model" \
                        "${CLIENT_ARGS[@]}" \
                        --transcripts "$mode" \
                        --few_shot_sampling "$sampling" \
                        --k_values $K_VALUES \
                        --num_runs "$NUM_RUNS" \
                        --num_workers "$NUM_WORKERS" \
                        "${EMBEDDING_ARGS[@]}" \
                        --yes \
                    || failed=$((failed + 1))
            done
        done
    done

    [ "$failed" -gt 0 ] && echo "[WARNING] CoT: $failed run(s) failed. Check $LOG_DIR/" \
                         || echo "[OK] CoT: All models completed successfully."

    # Merge per-model summaries
    for dir in "$RESULTS_ROOT"/chain-of-thought/*/; do
        merge_summaries "$dir"
    done
}

# ---------- Few-Shot with CoT via script 2 ----------
run_few_shot_with_cot() {
    log "FEW-SHOT + CoT EXPERIMENTS (via few-shot script)"

    local failed=0
    for mode in No Full Summarized; do
        for sampling in "${SAMPLING_METHODS[@]}"; do
            for model in "${MODELS[@]}"; do
                if ! configure_client "$model"; then
                    failed=$((failed + 1))
                    continue
                fi
                LOG_FILE="$LOG_DIR/few-shot-cot_${mode}_${sampling}_${model}.log"
                run_model "$LOG_FILE" \
                    python 2_few-shot-experiment.py \
                        --model "$model" \
                        "${CLIENT_ARGS[@]}" \
                        --transcripts "$mode" \
                        --few_shot_sampling "$sampling" \
                        --use_cot \
                        --k_values $K_VALUES \
                        --num_runs "$NUM_RUNS" \
                        --num_workers "$NUM_WORKERS" \
                        "${EMBEDDING_ARGS[@]}" \
                        --yes \
                    || failed=$((failed + 1))
            done
        done
    done

    [ "$failed" -gt 0 ] && echo "[WARNING] Few-shot+CoT: $failed run(s) failed. Check $LOG_DIR/" \
                         || echo "[OK] Few-shot+CoT: All models completed successfully."

    # Merge per-model summaries
    for dir in "$RESULTS_ROOT"/few-shot/*/; do
        merge_summaries "$dir"
    done
}


# ==============================================================================
# Main: decide which experiments to run based on argument
# ==============================================================================
EXPERIMENT="${1:-all}"

echo ""
echo "============================================================"
echo "  Models will run SEQUENTIALLY (tqdm visible on terminal)"
echo "  Logs: $LOG_DIR/"
echo "  Models: ${MODELS[*]}"
echo "============================================================"
echo ""

case "$EXPERIMENT" in
    zero-shot)
        run_zero_shot
        ;;
    few-shot)
        run_few_shot
        ;;
    cot)
        run_cot
        ;;
    few-shot-cot)
        run_few_shot_with_cot
        ;;
    all)
        run_zero_shot
        run_few_shot
        run_cot
        run_few_shot_with_cot
        ;;
    *)
        echo "Usage: $0 [zero-shot|few-shot|cot|few-shot-cot|all]"
        echo ""
        echo "  zero-shot     - Run zero-shot experiments (with and without CoT)"
        echo "  few-shot      - Run few-shot experiments (without CoT)"
        echo "  cot           - Run chain-of-thought experiments (always CoT)"
        echo "  few-shot-cot  - Run few-shot experiments with CoT flag"
        echo "  all           - Run all experiments (default)"
        exit 1
        ;;
esac

# ---------- Final Report ----------
echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "============================================================"
echo "  Logs:      $LOG_DIR/"
echo "  Summaries: $RESULTS_ROOT/*/*/all_results_summary.csv"
echo "  Metrics:   $RESULTS_ROOT/metrics/"
echo ""
echo "  Check individual log files for details on any failures."
echo "  Re-run this script to retry failed experiments (completed ones are skipped)."
echo "============================================================"
