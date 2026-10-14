#!/usr/bin/env bash
# Reproducible experiment driver for the write-up.
#
# Every number that appears in the report comes from one of these phases, and
# every phase writes its own JSON into results/.  Run phases in order; each is
# idempotent and safe to re-run.
#
#   ./scripts/experiments.sh sweep     # pick the normalisation/augmentation config
#   ./scripts/experiments.sh main      # model comparison, 3-class, full channels
#   ./scripts/experiments.sh hw        # sensor ablation, hardware-only channels
#   ./scripts/experiments.sh binary    # literature-standard stress-vs-baseline
#   ./scripts/experiments.sh kfold     # the base paper's window-level k-fold
#   ./scripts/experiments.sh all
#
# Long phases are launched with `setsid` so they survive the shell that started
# them; progress is written to logs/.

set -u
cd "$(dirname "$0")/.."
PY="${PY:-/home/pratik/.venvs/threatwatch/bin/python}"
mkdir -p logs results

FULL="data/processed/w15_s5_fs32.npz"
HW="data/processed/w15_s5_fs32_hw.npz"

BASELINES="svm naive_bayes random_forest logistic"
DEEP="cnn cnn_transformer patch_vit bilstm transformer"

# The pure Transformer attends over all 480 timesteps (O(T^2)) and is ~10x slower
# per step than the CNN+Transformer (see results/benchmark.json), so it gets a
# shorter budget.  This is a resource limit, stated as such in the report, not a
# fair-weather omission.
TRANSFORMER_ARGS="--epochs 15 --patience 6"

run() {  # run <logname> <args...>
  local log="$1"; shift
  echo "[experiments] $log: $PY scripts/train.py $*" | tee -a "logs/experiments.log"
  "$PY" -u scripts/train.py "$@" >> "logs/$log" 2>&1
}

phase_sweep() {
  # Does per-window normalisation / augmentation fix the subject-shift problem?
  # CNN and Patch-ViT are the two cheapest models, so they carry the sweep.
  for model in cnn patch_vit; do
    for norm in global window; do
      for aug in aug noaug; do
        extra=""; [ "$aug" = "noaug" ] && extra="--no-augment"
        run "sweep_${model}_${norm}_${aug}.log" \
          --dataset "$FULL" --task 3class --channels full \
          --models "$model" --normalize "$norm" $extra \
          --epochs 40 --patience 10 --no-tensorboard
      done
    done
  done
  "$PY" scripts/summarize_sweep.py
}

phase_main() {
  run "train_main.log" --dataset "$FULL" --task 3class --channels full \
    --models $BASELINES cnn cnn_transformer patch_vit bilstm --normalize window --no-tensorboard
  run "train_main_transformer.log" --dataset "$FULL" --task 3class --channels full \
    --models transformer --normalize window $TRANSFORMER_ARGS --no-tensorboard
}

# The ablation exists to answer Task 2 with a measured number: what does the
# missing GSR/EDA sensor cost?  A baseline plus one deep model is enough to
# answer it, and keeps the phase to a few minutes.
phase_hw() {
  run "train_hw.log" --dataset "$HW" --task 3class --channels hw \
    --models $BASELINES cnn --normalize window --no-tensorboard
}

phase_binary() {
  run "train_binary.log" --dataset "$FULL" --task binary --channels full \
    --models $BASELINES cnn_transformer --normalize window --no-tensorboard
}

phase_kfold() {
  # The base paper's protocol: window-level stratified k-fold that mixes subjects
  # across folds.  Included only so the comparison against it is like-for-like.
  run "kfold.log" --dataset "$FULL" --task 3class --channels full \
    --protocol window_kfold --kfold 5 --normalize window \
    --models $BASELINES cnn patch_vit --no-tensorboard
}

cmd="${1:-all}"
case "$cmd" in
  sweep)    phase_sweep ;;
  main)     phase_main ;;
  hw)       phase_hw ;;
  binary)   phase_binary ;;
  kfold)    phase_kfold ;;
  all)      phase_sweep; phase_main; phase_hw; phase_binary; phase_kfold ;;
  *) echo "usage: $0 {sweep|main|hw|binary|kfold|all}"; exit 2 ;;
esac
echo "[experiments] $cmd done"
