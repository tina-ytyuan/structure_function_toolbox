#!/bin/bash
# Build normative measure references from HCP data on the DCC.
#
# Build normative measure references (mean + SD + n) from a folder of subjects.
# The resulting references support the single-subject t-test.
#
# Everything is configurable without editing this file: pass the SLURM account
# and partition on the command line, and the paths as environment variables.
#
#   sbatch --account=ACCOUNT --partition=PARTITION \
#          --export=ALL,PROJECT_DIR=/path/to/project,LIMIT=20 \
#          scripts/build_refs.slurm reho alff falff rsfa
#
# Measures default to the core four if none are given. Override any of
# PROJECT_DIR, RUN, SUBJECTS_ROOT, OUT_DIR, MASK, PATTERN, TR, LIMIT.

#SBATCH --job-name=sft_refs
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
mkdir -p slurm_logs

# ---- CONFIG (env-overridable; defaults are placeholders) -------------
PROJECT_DIR="${PROJECT_DIR:-/path/to/project}"          # project root on the cluster
RUN="${RUN:-Rest1LR}"                                   # which run to use
SUBJECTS_ROOT="${SUBJECTS_ROOT:-$PROJECT_DIR/data/$RUN}"  # folder of subject NIfTIs
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/$USER/refs}"           # where references go
MASK="${MASK:-$PROJECT_DIR/data/final_mask.nii}"        # mask applied to every subject
PATTERN="${PATTERN:-*_finproc.nii.gz}"
TR="${TR:-0.72}"
# LIMIT: a small number for a quick test; set LIMIT= (empty) for all subjects.
LIMIT="${LIMIT-20}"
# ---------------------------------------------------------------------

if [ ! -d "$SUBJECTS_ROOT" ]; then
    echo "ERROR: SUBJECTS_ROOT does not exist: $SUBJECTS_ROOT" >&2
    echo "Set PROJECT_DIR (or SUBJECTS_ROOT) via --export, e.g." >&2
    echo "  sbatch --export=ALL,PROJECT_DIR=/hpc/group/yourproject ..." >&2
    exit 2
fi

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
    # -u: unbuffered stdout so progress appears in the log as it happens,
    # rather than only when the process exits.
    python -u scripts/build_measure_reference.py \
        --subjects-root "$SUBJECTS_ROOT" \
        --pattern "$PATTERN" \
        --measure "$M" \
        --tr "$TR" \
        --mask "$MASK" \
        $LIMIT_ARG \
        --out "$OUT_DIR/measure_ref_${M}.npz"
done

echo "done $(date)"
