#!/bin/bash
# Build normative measure references from HCP data on the DCC.
#
# Submit from your toolbox checkout on the cluster, e.g.:
#     sbatch scripts/build_refs.slurm reho alff falff rsfa
# (measures default to the core four if none are given). Edit the CONFIG block
# below to match your paths, and the SBATCH account/partition to your cluster.

#SBATCH --job-name=sft_refs
#SBATCH --account=your_account
#SBATCH --partition=your_partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
mkdir -p slurm_logs

# ---- CONFIG (edit these) --------------------------------------------
PROJECT_DIR=/path/to/project                  # your project root on the cluster
RUN=Rest1LR                                   # which HCP run to use
SUBJECTS_ROOT="$PROJECT_DIR/data/$RUN"        # folder of *_finproc.nii.gz
OUT_DIR="$PROJECT_DIR/$USER/refs"             # where references are written
MASK="$PROJECT_DIR/data/final_mask.nii"       # group mask applied to every subject
PATTERN='*_finproc.nii.gz'
TR=0.72
# LIMIT: set to a small number for a quick test, or leave empty for all subjects
LIMIT=20
# ---------------------------------------------------------------------

MEASURES=("$@")
if [ "${#MEASURES[@]}" -eq 0 ]; then
    MEASURES=(reho alff falff rsfa)
fi

echo "Job $SLURM_JOB_ID on $(hostname) | start $(date)"
echo "Subjects: $SUBJECTS_ROOT | out: $OUT_DIR | measures: ${MEASURES[*]}"

# Activate the project's virtualenv (created with: python3 -m venv .venv).
source .venv/bin/activate

mkdir -p "$OUT_DIR"
LIMIT_ARG=""
[ -n "${LIMIT:-}" ] && LIMIT_ARG="--limit $LIMIT"

for M in "${MEASURES[@]}"; do
    echo "=== building reference: $M ==="
    python scripts/build_measure_reference.py \
        --subjects-root "$SUBJECTS_ROOT" \
        --pattern "$PATTERN" \
        --measure "$M" \
        --tr "$TR" \
        --mask "$MASK" \
        $LIMIT_ARG \
        --out "$OUT_DIR/measure_ref_${M}.npz"
done

echo "done $(date)"
