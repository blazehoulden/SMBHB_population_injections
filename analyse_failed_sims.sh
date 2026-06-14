#!/usr/bin/env bash
# analyse_failed_sims.sh

RUNS=(
    "/fred/oz005/users/bhoulden/SMBHB_population_injections/runs/2026-06-05_pessimistic"
    "/fred/oz005/users/bhoulden/SMBHB_population_injections/runs/2026-06-05_optimistic"
    "/fred/oz005/users/bhoulden/SMBHB_population_injections/runs/2026-06-05_realistic"
)

for RUN in "${RUNS[@]}"; do
    LOGDIR="${RUN}/logs"
    CONFIG=$(basename "$RUN" | sed 's/[0-9]*-[0-9]*-[0-9]*_//')

    echo ""
    echo "========================================================"
    echo "Run: ${CONFIG}"
    echo "========================================================"

    # --- Category 1: KeyError noise_seed ----------------------------------------
    KEYERROR_SIMS=$(grep -l "KeyError: 'noise_seed'" "${LOGDIR}"/stage2_*try1*.err \
        2>/dev/null | grep -oP 'sim\K[0-9]+' | sort -nu)

    echo ""
    echo "Category 1 — KeyError: noise_seed (scenario failed, baseline OK):"
    if [[ -z "$KEYERROR_SIMS" ]]; then
        echo "  none"
    else
        for sim in $KEYERROR_SIMS; do
            echo "  sim$(printf '%03d' $sim)"
        done
        echo "  Count: $(echo "$KEYERROR_SIMS" | wc -w)"
    fi

    # --- Category 2: handoff file not found ------------------------------------
    HANDOFF_SIMS=$(grep -l "handoff file not found" "${LOGDIR}"/stage2_*try1*.err \
        2>/dev/null | grep -oP 'sim\K[0-9]+' | sort -nu)

    echo ""
    echo "Category 2 — handoff file not found (baseline subprocess never ran):"
    if [[ -z "$HANDOFF_SIMS" ]]; then
        echo "  none"
    else
        for sim in $HANDOFF_SIMS; do
            echo "  sim$(printf '%03d' $sim)"
        done
        echo "  Count: $(echo "$HANDOFF_SIMS" | wc -w)"
    fi

    # --- Category 3: noise-only SNR above threshold (ValueError raised) --------
    # Exclude sims where iter-1 SNR was also negative — those are unfixable.
    echo ""
    echo "Category 3 — noise-only SNR above threshold (ValueError raised):"
    SNR_SIMS_LIST=()

    for errfile in "${LOGDIR}"/stage2_*try1*.err; do
        [[ -f "$errfile" ]] || continue

        # Must contain the actual ValueError text
        grep -q "at or below the noise-only SNR\|SNR band ceiling.*at or below" \
            "$errfile" 2>/dev/null || continue

        sim=$(echo "$errfile" | grep -oP 'sim\K[0-9]+' | head -1)
        [[ -z "$sim" ]] && continue
        SIMPAD=$(printf '%03d' $sim)

        # Check iter-1 SNR in the corresponding .out file
        outfile=$(ls "${LOGDIR}"/stage2_sim${SIMPAD}_try1*.out 2>/dev/null | head -1)
        iter1_snr=""
        if [[ -n "$outfile" ]]; then
            iter1_snr=$(grep "Iter  1: OS SNR=" "$outfile" 2>/dev/null | head -1 \
                | grep -oP 'OS SNR=\K[-0-9.]+')
        fi

        if [[ -n "$iter1_snr" ]] && (( $(echo "$iter1_snr < 0" | bc -l) )); then
            echo "  sim${SIMPAD}  SKIP — iter1 SNR=${iter1_snr} (unfixable, not resubmitting)"
            continue
        fi

        echo "  sim${SIMPAD}  ← FAILED (ValueError raised)"
        SNR_SIMS_LIST+=("$sim")
    done

    SNR_SIMS=$(printf '%s\n' "${SNR_SIMS_LIST[@]}" | sort -nu | tr '\n' ' ')
    echo "  Count: ${#SNR_SIMS_LIST[@]}"

    # --- Combined: all sims needing resubmission --------------------------------
    ALL_FAILED=$(echo "$KEYERROR_SIMS $HANDOFF_SIMS $SNR_SIMS" \
        | tr ' ' '\n' | grep -v '^$' | sort -nu)

    echo ""
    echo "All sims needing resubmission (excluding already-complete):"
    RESUBMIT=()
    for sim in $ALL_FAILED; do
        SIMPAD=$(printf '%03d' $sim)
        SENTINEL="${RUN}/sim${SIMPAD}/metadata/stage2_complete.json"
        if [[ -f "$SENTINEL" ]]; then
            echo "  sim${SIMPAD}  SKIP (already complete)"
        else
            echo "  sim${SIMPAD}  ← RESUBMIT"
            RESUBMIT+=("$sim")
        fi
    done

    echo ""
    echo "Resubmit list for ${CONFIG}: ${RESUBMIT[*]}"
    echo "Total to resubmit: ${#RESUBMIT[@]}"

done