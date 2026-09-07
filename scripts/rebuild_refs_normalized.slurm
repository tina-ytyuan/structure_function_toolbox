#!/bin/bash
# Rebuild normative references from existing per-subject measure maps, with
# global normalisation (DPABI 'm' convention).
#
# Submit from a login node -- job submission is permitted there; the work runs
# on a compute node. No SSH key or interactive session required.
#
#   sbatch scripts/rebuild_refs_normalized.slurm rsfa alff reho falff
#
# Optionally pin the cohort so the reference matches the group's 936 subjects
# rather than every map on disk:
#
#   sbatch --export=ALL,SUBJECTS=/path/to/cohort_ids.txt \
#          scripts/rebuild_refs_normalized.slurm rsfa alff reho falff

#SBATCH --job-name=sft_renorm
#SBATCH --account=dataplus
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
mkdir -p slurm_logs outputs

# ---- CONFIG (override with --export=ALL,VAR=value) -------------------
PROJECT="${PROJECT:-/path/to/project}"

# RUN=all pools every run (matching the group's mean/SD maps, which used
# Rest1LR, Rest1RL, Rest2LR and Rest2RL). A subject's runs are averaged
# together before aggregating, so n stays equal to the number of people.
RUN="${RUN:-all}"
if [ "$RUN" = "all" ]; then RUN_GLOB="*"; else RUN_GLOB="$RUN"; fi
# Two source trees: the core four measures and the extended set.
CORE_ROOT="${CORE_ROOT:-$PROJECT/core_measures/$RUN_GLOB}"
EXTRA_ROOT="${EXTRA_ROOT:-$PROJECT/extra_measures/$RUN_GLOB}"
MASK="${MASK:-$PROJECT/data_FA/masks/final_mask_no_ventricles.nii}"
OUT_DIR="${OUT_DIR:-outputs}"
SUBJECTS="${SUBJECTS:-}"          # optional cohort ID list
# ---------------------------------------------------------------------

# The two pipelines name their outputs differently, and a loose glob would
# cross-match (e.g. *alff_slow4* also hits falff_slow4). Map each toolbox
# measure to its exact source glob instead.
#   core:  <root>/<subject>/<subject>_..._<measure>.nii.gz
#   extra: <root>/<subject>/<key>.nii.gz
glob_for() {
    case "$1" in
        rsfa|alff|reho|falff)  echo "$CORE_ROOT/*/*_$1.nii.gz" ;;
        alff_slow4)            echo "$EXTRA_ROOT/*/alff_slow4.nii.gz" ;;
        alff_slow5)            echo "$EXTRA_ROOT/*/alff_slow5.nii.gz" ;;
        falff_slow4)           echo "$EXTRA_ROOT/*/falff_slow4.nii.gz" ;;
        falff_slow5)           echo "$EXTRA_ROOT/*/falff_slow5.nii.gz" ;;
        int)                   echo "$EXTRA_ROOT/*/int_map.nii.gz" ;;
        coherence_reho)        echo "$EXTRA_ROOT/*/coherence_reho.nii.gz" ;;
        # Toolbox MSE uses scales 1-5, matching mse_complexity_index_1_5.
        mse)                   echo "$EXTRA_ROOT/*/mse_complexity_index_1_5.nii.gz" ;;
        *)                     echo "" ;;
    esac
}

MEASURES=("$@")
if [ "${#MEASURES[@]}" -eq 0 ]; then
    MEASURES=(rsfa alff reho falff alff_slow4 alff_slow5 falff_slow4 \
              falff_slow5 int coherence_reho mse)
fi

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) | start $(date)"
echo "core maps : $CORE_ROOT"
echo "extra maps: $EXTRA_ROOT"
echo "mask      : $MASK"
echo "measures  : ${MEASURES[*]}"
[ -n "$SUBJECTS" ] && echo "cohort list: $SUBJECTS" || \
    echo "cohort list: (none - using every map found)"

if [ ! -f "$MASK" ]; then
    echo "ERROR: mask not found: $MASK" >&2
    exit 2
fi

source .venv/bin/activate

subj_args=()
[ -n "$SUBJECTS" ] && subj_args=(--subjects "$SUBJECTS")

for M in "${MEASURES[@]}"; do
    G="$(glob_for "$M")"
    echo ""
    echo "=== $M ==="
    if [ -z "$G" ]; then
        echo "  SKIP: no source glob defined for '$M'" >&2
        continue
    fi
    echo "  source: $G"
    # Count without letting bash expand into the command line.
    N=$(ls -1 $G 2>/dev/null | wc -l)
    if [ "$N" -eq 0 ]; then
        echo "  SKIP: no maps matched" >&2
        continue
    fi
    echo "  found : $N maps"

    # -u keeps progress visible in the log while the job runs.
    python -u scripts/reference_from_subject_maps.py \
        --measure "$M" \
        --maps "$G" \
        --mask "$MASK" \
        "${subj_args[@]}" \
        --out "$OUT_DIR/measure_ref_${M}.npz" || { echo "  BUILD FAILED: $M" >&2; continue; }

    echo "--- calibration check: $M ---"
    python -u scripts/check_reference_calibration.py \
        --ref "$OUT_DIR/measure_ref_${M}.npz" \
        --maps "$G" \
        --mask "$MASK" --n 20 || echo "  (calibration check failed for $M)"
done

echo ""
echo "done $(date)"
