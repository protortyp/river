#!/usr/bin/env bash
#
# Profiling script for pokergpu training pipeline
#
# Usage:
#   ./bench/bench.sh              # Run all benchmarks
#   ./bench/bench.sh quick        # Quick micro-benchmarks only
#   ./bench/bench.sh full         # Full training profiler
#   ./bench/bench.sh nsys         # Deep GPU analysis with Nsight Systems
#   ./bench/bench.sh baseline     # Baseline throughput tests
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# Default settings
NUM_ENVS="${NUM_ENVS:-65536}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-profile_output}"

mkdir -p "$OUTPUT_DIR"

header() {
    echo ""
    echo "========================================================================"
    echo " $1"
    echo "========================================================================"
}

run_baseline() {
    header "Baseline Throughput Tests"

    echo ">> Environment-only throughput (kernel ceiling)"
    uv run python bench/bench_sps.py --device "$DEVICE" --num-envs "$NUM_ENVS" --steps 200

    echo ""
    echo ">> League rollout - no opponents (self-play only)"
    uv run python bench/bench_league_rollout.py --device "$DEVICE" --num-envs "$NUM_ENVS" \
        --snapshot-filled 0 --iters 10

    echo ""
    echo ">> League rollout - with opponents (k=4)"
    uv run python bench/bench_league_rollout.py --device "$DEVICE" --num-envs "$NUM_ENVS" \
        --snapshot-filled 8 --rollout-snapshot-k 4 --iters 10

    echo ""
    echo ">> League rollout - worst case (k=0, all 32 snapshots)"
    uv run python bench/bench_league_rollout.py --device "$DEVICE" --num-envs "$NUM_ENVS" \
        --snapshot-filled 32 --rollout-snapshot-k 0 --iters 5
}

run_quick() {
    header "Quick Micro-benchmarks"
    uv run python bench/profile_kernels.py --device "$DEVICE" --num-envs "$NUM_ENVS" --iterations 50
}

run_full() {
    header "Full Training Profiler"
    uv run python bench/profile_training.py --device "$DEVICE" --num-envs "$NUM_ENVS" \
        --steps 50 --output-dir "$OUTPUT_DIR"

    echo ""
    echo "Chrome trace saved to: $OUTPUT_DIR/torch_trace.json"
    echo "Open in: chrome://tracing or https://ui.perfetto.dev/"
}

run_nsys() {
    header "Nsight Systems Deep GPU Analysis"

    if ! command -v nsys &> /dev/null; then
        echo "ERROR: nsys (Nsight Systems) not found in PATH"
        echo "Install NVIDIA Nsight Systems from: https://developer.nvidia.com/nsight-systems"
        exit 1
    fi

    NSYS_OUTPUT="$OUTPUT_DIR/nsys_profile"
    echo "Running Nsight Systems profiler..."
    echo "Output: ${NSYS_OUTPUT}.nsys-rep"

    nsys profile \
        -t cuda,nvtx,osrt \
        -o "$NSYS_OUTPUT" \
        --force-overwrite true \
        uv run python bench/profile_training.py \
            --device "$DEVICE" \
            --num-envs 32768 \
            --steps 20 \
            --skip-torch-profiler \
            --output-dir "$OUTPUT_DIR"

    echo ""
    echo "Nsight Systems report saved to: ${NSYS_OUTPUT}.nsys-rep"
    echo "Open with: nsys-ui ${NSYS_OUTPUT}.nsys-rep"
}

run_training() {
    header "Training SPS Benchmark (Full System)"
    uv run python bench/bench_training_sps.py --device "$DEVICE" --num-envs "$NUM_ENVS" \
        --warmup 2 --iters 3
}

run_all() {
    run_baseline
    run_quick
    run_full

    header "Summary"
    echo "All benchmarks complete. Results in: $OUTPUT_DIR/"
    echo ""
    echo "Next steps:"
    echo "  1. Review Chrome trace: $OUTPUT_DIR/torch_trace.json"
    echo "  2. For deeper GPU analysis: ./bench/bench.sh nsys"
    echo "  3. See docs/profiling_guide.md for optimization recommendations"
}

# Parse command
case "${1:-all}" in
    quick)
        run_quick
        ;;
    full)
        run_full
        ;;
    nsys)
        run_nsys
        ;;
    baseline)
        run_baseline
        ;;
    training)
        run_training
        ;;
    all)
        run_all
        ;;
    help|--help|-h)
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  quick     - Quick micro-benchmarks (~1 min)"
        echo "  full      - Full training profiler with torch.profiler (~2 min)"
        echo "  training  - End-to-end training SPS with optimization comparison (~3 min)"
        echo "  nsys      - Deep GPU analysis with Nsight Systems (~2 min)"
        echo "  baseline  - Baseline throughput tests (~3 min)"
        echo "  all       - Run baseline + quick + full (default)"
        echo ""
        echo "Environment variables:"
        echo "  NUM_ENVS    - Number of environments (default: 65536)"
        echo "  DEVICE      - CUDA device (default: cuda)"
        echo "  OUTPUT_DIR  - Output directory (default: profile_output)"
        echo ""
        echo "Examples:"
        echo "  $0                           # Run all benchmarks"
        echo "  NUM_ENVS=131072 $0 quick     # Quick benchmarks with 131k envs"
        echo "  DEVICE=cuda:1 $0 full        # Full profiler on GPU 1"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run '$0 help' for usage"
        exit 1
        ;;
esac
