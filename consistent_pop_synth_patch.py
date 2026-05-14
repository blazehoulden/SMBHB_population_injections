"""
consistent_pop_synth_patch.py
─────────────────────────────
Monkey-patch consistent_pop_synth to capture the exact pre-injection stoas
(post-noise, pre-GW) that were used during SNR computation and return them
inside each population result dict under the key "pre_injection_stoas".

Import this module BEFORE importing consistent_pop_synth everywhere in the
Slurm pipeline (stage1_setup.py already does this).  It patches the module
in-place so no source file changes are required.

What changes
────────────
1. _build_result_distance_scaling gains a `pre_injection_stoas` parameter
   (dict {psr_name: np.ndarray of stoas in days, float64}).  It stores it in
   the result dict only when explicitly provided (backward-compatible).

2. generate_consistent_population_distance_scaling — the two call-sites that
   call _build_result_distance_scaling now pass `pre_injection_stoas=current_stoas`.
   current_stoas is already in scope at both call-sites (it was set just before
   compute_population_snr was called).

The patch is applied once at import time via _apply_patch().
"""

import types
import numpy as np
import consistent_pop_synth as _mod


# ─────────────────────────────────────────────────────────────────────────────
# Replacement for _build_result_distance_scaling
# ─────────────────────────────────────────────────────────────────────────────

def _patched_build_result(
    population,
    SNR_final,
    pta,
    psrs,
    timing_profile=None,
    memory_profile=None,
    pre_injection_stoas=None,   # NEW: dict {psr_name: stoas_days float64}
):
    """
    Drop-in replacement for _build_result_distance_scaling that optionally
    stores the pre-injection stoa snapshot.
    """
    result = {
        "population"  : population,
        "SNR_final"   : float(SNR_final),
        "n_bininaries": len(population.f) if population is not None else 0,
        "pta"         : pta,
        "psrs"        : psrs,
    }
    if timing_profile is not None:
        result["timing_profile"] = timing_profile
    if memory_profile is not None:
        result["memory_profile"] = memory_profile
    if pre_injection_stoas is not None:
        # Deep-copy so the caller cannot mutate them later
        result["pre_injection_stoas"] = {
            name: np.array(stoas, dtype=np.float64)
            for name, stoas in pre_injection_stoas.items()
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Replacement for generate_consistent_population_distance_scaling
# ─────────────────────────────────────────────────────────────────────────────
# We rewrite only the parts that call _build_result_distance_scaling, adding
# pre_injection_stoas=current_stoas at each call-site.  Everything else is
# delegated to the original implementation via a thin wrapper that intercepts
# the _build_result calls.

def _patched_generate_consistent(
    config_template,
    smbhb_module,
    psrs_clean,
    raw_noise_params,
    Tspan,
    target_SNR,
    original_stoas,
    SNR_range=None,
    snr_noise_baseline=0.0,
    timer=True,
    verbose=True,
    n_iterations=0,
    toggle_memory_profiling=False,
    keep_amplitudes_in_result=False,
    inject_eps=1e-6,
    precompute_parallel=False,
    require_snr_in_range=True,
):
    """
    Wrapper around the original generate_consistent_population_distance_scaling
    that injects pre_injection_stoas into every _build_result call.

    Strategy: temporarily replace _build_result_distance_scaling in the module
    with a closure that captures current_stoas from the outer scope, then
    restore it after the call.
    """
    # We need to capture current_stoas at the moment _build_result is called.
    # The original function sets current_stoas immediately before calling
    # compute_population_snr, which is immediately before _build_result.
    # We intercept by wrapping compute_population_snr to record the stoas
    # that were passed as current_stoas, then use them inside _build_result.

    _captured_current_stoas = {}   # mutated by the wrappers below

    original_compute_snr   = _mod.compute_population_snr
    original_build_result  = _mod._build_result_distance_scaling

    def _snr_interceptor(*args, **kwargs):
        """Record current_stoas whenever compute_population_snr is called."""
        # current_stoas is the 4th positional arg or a keyword arg
        cs = kwargs.get("current_stoas", args[3] if len(args) > 3 else None)
        if cs is not None:
            _captured_current_stoas.clear()
            _captured_current_stoas.update({
                name: np.array(stoas, dtype=np.float64)
                for name, stoas in cs.items()
            })
        return original_compute_snr(*args, **kwargs)

    def _build_interceptor(*args, **kwargs):
        """Pass pre_injection_stoas into every _build_result call."""
        if _captured_current_stoas:
            kwargs.setdefault("pre_injection_stoas", dict(_captured_current_stoas))
        return _patched_build_result(*args, **kwargs)

    # Patch module-level references
    _mod.compute_population_snr          = _snr_interceptor
    _mod._build_result_distance_scaling  = _build_interceptor

    try:
        result = _mod._original_generate_consistent(
            config_template        = config_template,
            smbhb_module           = smbhb_module,
            psrs_clean             = psrs_clean,
            raw_noise_params       = raw_noise_params,
            Tspan                  = Tspan,
            target_SNR             = target_SNR,
            original_stoas         = original_stoas,
            SNR_range              = SNR_range,
            snr_noise_baseline     = snr_noise_baseline,
            timer                  = timer,
            verbose                = verbose,
            n_iterations           = n_iterations,
            toggle_memory_profiling= toggle_memory_profiling,
            keep_amplitudes_in_result = keep_amplitudes_in_result,
            inject_eps             = inject_eps,
            precompute_parallel    = precompute_parallel,
            require_snr_in_range   = require_snr_in_range,
        )
    finally:
        # Always restore originals even if an exception is raised
        _mod.compute_population_snr         = original_compute_snr
        _mod._build_result_distance_scaling = original_build_result

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Apply the patch
# ─────────────────────────────────────────────────────────────────────────────

def _apply_patch():
    """
    Called once at import time.  Saves the originals then installs the patches.
    Idempotent: if already patched, does nothing.
    """
    if getattr(_mod, "_slurm_patch_applied", False):
        return

    # Save originals under private names so the wrappers can call them
    _mod._original_generate_consistent  = _mod.generate_consistent_population_distance_scaling
    _mod._original_build_result         = _mod._build_result_distance_scaling

    # Install patched versions
    _mod._build_result_distance_scaling               = _patched_build_result
    _mod.generate_consistent_population_distance_scaling = _patched_generate_consistent

    _mod._slurm_patch_applied = True
    print("[consistent_pop_synth_patch] Patch applied: "
          "pre_injection_stoas will be captured in each population result.")


_apply_patch()
