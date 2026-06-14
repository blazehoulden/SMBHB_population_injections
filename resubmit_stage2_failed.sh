#!/usr/bin/env bash
# resubmit_stage2_failed.sh
# Resubmits stage2 only for sims identified as failed, skipping already-complete ones.

REPO_DIR="/fred/oz005/users/bhoulden/SMBHB_population_injections"
ENV_SETUP="module purge; unset PYTHONPATH; module load mamba; mamba activate smbhb312;"

declare -A N_CHUNKS=(
    ["pessimistic"]=8
    ["optimistic"]=1
    ["realistic"]=1
)

declare -A N_SIMS=(
    ["pessimistic"]=450 
    ["optimistic"]=450
    ["realistic"]=450
)

N_TEST=1000
CGW_FLAG="--cgw"
SYNTHETIC_PTAS_FLAG="--synthetic-ptas"
NOISE_SEED_BASE=26072001

# Resources matching original submission
S2_MEM="38G"
S2_TIME="04:30:00"
S2_CPUS=1

declare -A RUN_CONFIGS=(
    ["pessimistic"]="pessimistic"
    ["optimistic"]="optimistic"
    ["realistic"]="realistic"
)

declare -A RUN_DIRS=(
    ["pessimistic"]="${REPO_DIR}/runs/2026-06-05_pessimistic"
    ["optimistic"]="${REPO_DIR}/runs/2026-06-05_optimistic"
    ["realistic"]="${REPO_DIR}/runs/2026-06-05_realistic"
)

# Resubmit lists from analyse_failed_sims.sh (2026-06-09)
declare -A RESUBMIT_LISTS=(
    ["pessimistic"]="1 3 4 5 6 7 10 12 14 19 35 39 54 59 66 69 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 418 423"
    ["optimistic"]="1 3 4 37 38 39 40 44 45 46 47 48 49 50 51 52 53 54 55 56 57"
    ["realistic"]="10 11 12 13 17 21 37 43 45 54 59 66 69 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116"
)

for CONFIG in pessimistic optimistic realistic; do
    OUTPUT_DIR="${RUN_DIRS[$CONFIG]}"
    SIM_LIST="${RESUBMIT_LISTS[$CONFIG]}"

    echo ""
    echo "========================================================"
    echo "Submitting ${CONFIG}"
    echo "========================================================"

    for sim_id in $SIM_LIST; do
        sim_id_padded=$(printf '%03d' "$sim_id")
        SENTINEL="${OUTPUT_DIR}/sim${sim_id_padded}/metadata/stage2_complete.json"

        # Double-check not already complete (in case it finished since analysis)
        if [[ -f "$SENTINEL" ]]; then
            echo "  sim${sim_id_padded}: SKIP (already complete)"
            continue
        fi

        # Check stage1 outputs exist — no point submitting if populations missing
        POP_DIR="${OUTPUT_DIR}/sim${sim_id_padded}/populations"
        if [[ ! -d "$POP_DIR" ]] || [[ -z "$(ls ${POP_DIR}/subpop_*.pkl.gz 2>/dev/null)" ]]; then
            echo "  sim${sim_id_padded}: SKIP (no stage1 populations found — needs stage1 rerun)"
            continue
        fi

        S2_JOB=$(sbatch --parsable \
            --job-name="s2_${sim_id_padded}_fix" \
            --nodes=1 --ntasks=1 \
            --cpus-per-task="${S2_CPUS}" \
            --mem="${S2_MEM}" \
            --time="${S2_TIME}" \
            --output="${OUTPUT_DIR}/logs/stage2_sim${sim_id_padded}_fix_%j.out" \
            --error="${OUTPUT_DIR}/logs/stage2_sim${sim_id_padded}_fix_%j.err" \
            --wrap="${ENV_SETUP} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
                python -u ${REPO_DIR}/stage2_inject.py \
                --output-dir ${OUTPUT_DIR} \
                --config ${CONFIG} \
                --target-snr 3.75 \
                --snr-range 3.5 4.0 \
                --sim-id ${sim_id} \
                --n-chunks ${N_CHUNKS[$CONFIG]} \
                --n-test ${N_TEST} \
                --noise-seed-base ${NOISE_SEED_BASE} \
                --n-sims ${N_SIMS[$CONFIG]} \
                ${CGW_FLAG} \
                ${SYNTHETIC_PTAS_FLAG}"
        )
        echo "  sim${sim_id_padded}: submitted job ${S2_JOB}"
    done
done

echo ""
echo "Done. Monitor with: squeue -u \$USER"