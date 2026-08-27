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
PROJECT="${PROJECT:-/hpc/group/396-brainfun26}"
RUN="${RUN:-Rest1LR}"
MAPS_ROOT="${MAPS_ROOT:-$PROJECT/ajay/results_gpu_raw/$RUN}"
MASK="${MASK:-$PROJECT/data_FA/masks/final_mask_no_ventricles.nii}"
OUT_DIR="${OUT_DIR:-outputs}"
SUBJECTS="${SUBJECTS:-}"          # optional cohort ID list
# ---------------------------------------------------------------------

MEASURES=("$@")
if [ "${#MEASURES[@]}" -eq 0 ]; then
    MEASURES=(rsfa alff reho falff)
fi

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) | start $(date)"
echo "maps: $MAPS_ROOT"
echo "mask: $MASK"
echo "measures: ${MEASURES[*]}"
[ -n "$SUBJECTS" ] && echo "cohort list: $SUBJECTS" || \
    echo "cohort list: (none - using every map found)"

if [ ! -d "$MAPS_ROOT" ]; then
    echo "ERROR: MAPS_ROOT does not exist: $MAPS_ROOT" >&2
    exit 2
fi
if [ ! -f "$MASK" ]; then
    echo "ERROR: mask not found: $MASK" >&2
    exit 2
fi

source .venv/bin/activate

subj_args=()
[ -n "$SUBJECTS" ] && subj_args=(--subjects "$SUBJECTS")

for M in "${MEASURES[@]}"; do
    echo ""
    echo "=== $M ==="
    # -u keeps progress visible in the log while the job runs.
    python -u scripts/reference_from_subject_maps.py \
        --measure "$M" \
        --maps "$MAPS_ROOT/*/*_${M}.nii.gz" \
        --mask "$MASK" \
        "${subj_args[@]}" \
        --out "$OUT_DIR/measure_ref_${M}.npz"

    echo "--- calibration check: $M ---"
    python -u scripts/check_reference_calibration.py \
        --ref "$OUT_DIR/measure_ref_${M}.npz" \
        --maps "$MAPS_ROOT/*/*_${M}.nii.gz" \
        --mask "$MASK" --n 20 || echo "  (calibration check failed for $M)"
done

echo ""
echo "done $(date)"
