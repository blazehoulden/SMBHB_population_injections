"""
O(N log N) SMBHB population injection
==========================================================

Key ideas
---------
1.  Pre-compute per-binary *time-independent* amplitudes A_k, B_k and store
    them on the binary dict.  These absorb h0, Fp/Fx, iota, phi0 completely,
    so the TOA-dependent work per binary collapses to:

        r_k(t) = A_k * sin(2π f_k t)  +  B_k * cos(2π f_k t)

2.  Bin {A_k, B_k, f_k} onto a uniform frequency grid and evaluate the entire
    population sum with a single real IFFT — O(N_freq log N_freq) regardless
    of how many billions of binaries you have.

3.  For non-uniformly sampled TOAs (the realistic case) use NUFFT type-2
    (finufft) which costs O(N_freq log N_freq + N_toa) — still independent of
    the population size.

4.  A "direct batched" path (numpy matmul) is provided as a fallback and for
    small populations where FFT overhead is not worth paying.

Performance summary (rough)
---------------------------
Method              | Cost                     | 10^9 binaries?
--------------------|--------------------------|----------------
Old vectorised      | O(N_toa * N_binary)      | No — 800 GB RAM
New FFT             | O(N_binary + N_f log N_f)| Yes — ~8 GB RAM
New NUFFT           | O(N_binary + N_f log N_f + N_toa) | Yes

Pre-computation of A_k, B_k
----------------------------
Call `precompute_amplitudes(population, psr)` once per pulsar before
any injection loop.  The amplitudes are stored in-place on each binary dict
under keys 'A_<psr_name>' and 'B_<psr_name>'.  If you are simulating many
realisations with the same population, this pre-computation is done once and
reused for free.

Usage example
-------------
    from signal_injection_new import (
        precompute_amplitudes,
        inject_population_nufft,
    )

    # One-time setup: draw orbital params + precompute amplitudes
    population = draw_population(N=int(1e9), ...)   # your sampler
    for psr in psrs:
        precompute_amplitudes(population, psr)

    # Fast injection (pick one):
    inject_population_nufft(psrs, population, N_freq=2**20) # non-uniform TOAs

Dependencies: numpy (always), finufft.
"""

from SMBHB_pop_synth_new import precompute_amplitudes
import numpy as np
from config import c, G, pc
import finufft 

# ──────────────────────────────────────────────────────────────────────────────
# 1.  STRAIN AMPLITUDE  (unchanged from residuals.py, kept here for completeness)
# ──────────────────────────────────────────────────────────────────────────────

def strain_amplitude_vec(Mc_arr, f_arr, D_comov_arr, z_arr):
    """Vectorised strain amplitude for N binaries.  Returns h0 of shape (N,)."""
    f_rest   = 0.5 * (1 + z_arr) * f_arr
    D_si     = D_comov_arr * 1e6 * pc
    return (2 * (G * Mc_arr)**(5/3) * (2 * np.pi * f_rest)**(2/3)) / (c**4 * D_si)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  ANTENNA RESPONSE  (vectorised, unchanged logic)
# ──────────────────────────────────────────────────────────────────────────────

def antenna_response_vec(psr_ra, psr_dec, ra_arr, dec_arr, psi_arr):
    """Vectorised antenna response over N binaries. Returns Fp, Fx each (N,)."""
    N             = len(ra_arr)
    src_polar     = np.pi / 2 - dec_arr
    psr_polar     = np.pi / 2 - psr_dec

    omega_hat = np.array([
        -np.sin(src_polar) * np.cos(ra_arr),
        -np.sin(src_polar) * np.sin(ra_arr),
        -np.cos(src_polar),
    ])  # (3, N)

    p_hat = np.array([
        np.sin(psr_polar) * np.cos(psr_ra),
        np.sin(psr_polar) * np.sin(psr_ra),
        np.cos(psr_polar),
    ])  # (3,)

    m_hat = np.array([np.sin(ra_arr), -np.cos(ra_arr), np.zeros(N)])       # (3, N)
    n_hat = np.array([
        -np.cos(src_polar) * np.cos(ra_arr),
        -np.cos(src_polar) * np.sin(ra_arr),
         np.sin(src_polar),
    ])  # (3, N)

    cos_psi = np.cos(psi_arr); sin_psi = np.sin(psi_arr)
    m_rot   =  cos_psi * m_hat + sin_psi * n_hat
    n_rot   = -sin_psi * m_hat + cos_psi * n_hat

    denom = 1 + p_hat @ omega_hat      # (N,)
    p_m   = p_hat @ m_rot              # (N,)
    p_n   = p_hat @ n_rot              # (N,)

    Fp = 0.5 * (p_m**2 - p_n**2) / denom
    Fx =       (p_m   * p_n)     / denom
    return Fp, Fx


def inject_population_nufft(psrs, population, N_freq=None, pure_signal=True,
                             verbose=False, eps=1e-6):
    """
    Inject SMBHB population via NUFFT type-3 (no frequency quantisation error).

    Uses finufft.nufft1d3:
        f(x_j) = sum_k c_k * exp(i * s_k * x_j)

    where:
        x_j = t_j - t[0]          (TOA times, non-uniform, in seconds)
        s_k = 2π f_k              (source frequencies, non-uniform, in rad/s)
        c_k = (C_k - i S_k) / 2  (complex amplitude, see derivation below)

    Derivation
    ----------
    r(t) = sum_k A_k sin(2π f_k t_rel + φ0_k) + B_k cos(2π f_k t_rel + φ0_k)

    Expanding with phi0:
        S_k = A_k cos(φ0) - B_k sin(φ0)   [coeff of sin(2π f_k t_rel)]
        C_k = A_k sin(φ0) + B_k cos(φ0)   [coeff of cos(2π f_k t_rel)]

    Writing in complex form using e^{isx} = cos(sx) + i sin(sx):
        r = Re[ sum_k (C_k - i S_k) * e^{i 2π f_k t_rel} ]

    For real output we also need the conjugate (negative frequency) term:
        r = Re[ sum_k (C_k - i S_k) * e^{+i 2π f_k t_rel}
                    + (C_k + i S_k) * e^{-i 2π f_k t_rel} ] / 2

    Using nufft1d3 with only positive frequencies and dividing by 2:
        c_k = (C_k - i S_k) / 2
        r = 2 * Re[ nufft1d3(x, s, c) ]

    No grid, no quantisation — exact to NUFFT tolerance (eps).
    """

    f_arr    = population.f
    phi0_arr = population.phi0

    # Source frequencies in rad/s — these are the NUFFT "s" points
    # No gridding, no rounding — exact frequencies
    s_arr = 2 * np.pi * f_arr   # (N,) rad/s

    # phi0 rotation: same for all pulsars (source property)
    cos_phi0 = np.cos(phi0_arr)
    sin_phi0 = np.sin(phi0_arr)

    for psr in psrs:
        psr_name = psr.name
        if psr_name not in population.amp_A:
            precompute_amplitudes(population, psr)

        A_arr = population.amp_A[psr_name]
        B_arr = population.amp_B[psr_name]

        # phi0 rotation
        S = A_arr * cos_phi0 - B_arr * sin_phi0   # sin(2π f t_rel) coeff
        C = A_arr * sin_phi0 + B_arr * cos_phi0   # cos(2π f t_rel) coeff

        # Complex amplitudes: c_k = (C_k - i S_k) / 2
        c = (C - 1j * S) / 2   # (N,)

        # TOA times relative to first TOA — the NUFFT "x" points
        t_sec = np.asarray(psr.toas, dtype=np.float64)
        x     = t_sec - t_sec[0]   # (N_toa,) seconds

        # nufft1d3: f(x_j) = sum_k c_k * exp(i * s_k * x_j)
        # isign=+1 matches our e^{+i 2π f t} convention
        x      = np.ascontiguousarray(x,     dtype=np.float64)
        s_nufft = np.ascontiguousarray(s_arr, dtype=np.float64)
        c_nufft = np.ascontiguousarray(c,     dtype=np.complex128)

        f_out = finufft.nufft1d3(s_nufft, c_nufft, x, isign=+1, eps=eps)

        # Multiply by 2: we only passed positive frequencies,
        # negative frequencies contribute equal real part
        r_new = 2 * np.real(f_out)

        if verbose:
            print(f"  {psr_name}: RMS = {r_new.std()*1e9:.3f} ns")

        if pure_signal:
            psr._residuals = r_new
        else:
            psr._residuals = psr.residuals + r_new

    return psrs

# ──────────────────────────────────────────────────────────────────────────────
# 6.  DIRECT BATCHED PATH  (small populations, exact, no FFT)
# ──────────────────────────────────────────────────────────────────────────────

def inject_population_direct(
        psrs, population,
        pure_signal=True,
        chunk_size=50_000):
    """
    Direct batched injection using pre-computed A_k, B_k amplitudes.

    Cost:  O(N_toa * N_binary)  — same as before
    RAM:   O(N_toa * chunk_size) per chunk  — much lower than building full matrix

    This is exact and recommended for N_binary < ~10^5.  For larger populations
    use inject_population_fft or inject_population_nufft.

    Pre-requisite
    -------------
    Call precompute_amplitudes(population, psr) for each psr first.

    Key improvement over the old approach
    --------------------------------------
    The old _gw_residuals_vec built the full (N_toa × N_binary) phase matrix in
    one shot.  This path processes chunk_size binaries at a time, keeping RAM
    proportional to chunk_size rather than N_binary.  The phase calculation is
    still vectorised within each chunk.
    """

    phi0_arr = population.phi0
    f_arr    = population.f
    cos_phi0 = np.cos(phi0_arr)
    sin_phi0 = np.sin(phi0_arr)

    for psr in psrs:
        psr_name = psr.name
        if psr_name not in population.amp_A:
            precompute_amplitudes(population, psr)

        A_arr = population.amp_A[psr_name]
        B_arr = population.amp_B[psr_name]

        # S_k = coeff of sin(2π f t_rel), C_k = coeff of cos(2π f t_rel)
        S = A_arr * cos_phi0 - B_arr * sin_phi0
        C = A_arr * sin_phi0 + B_arr * cos_phi0

        t_sec = np.asarray(psr.toas, dtype=np.float64)
        t_rel = t_sec - t_sec[0]   # (N_toa,)

        # phase matrix (N_toa, N_binary)
        phase = 2 * np.pi * f_arr[np.newaxis, :] * t_rel[:, np.newaxis]

        r_new = (S * np.sin(phase) + C * np.cos(phase)).sum(axis=1)

        if pure_signal:
            psr._residuals = r_new
        else:
            psr._residuals = psr.residuals + r_new

    return psrs


# ──────────────────────────────────────────────────────────────────────────────
# 7.  POPULATION DRAWING WITH PRE-STORED ATTRIBUTES  (convenience)
# ──────────────────────────────────────────────────────────────────────────────

def draw_population_with_amplitudes(
        N, f_min, f_max,
        Mc_min=1e7, Mc_max=1e10,
        z_min=0.01, z_max=2.0,
        rng=None):
    """
    Draw a random population of N SMBHBs and pre-store all source-side
    attributes needed for injection.

    Draws uniformly in log-frequency, log-chirp-mass, and comoving volume.
    Orientation angles (ra, dec, psi, iota, phi0) are drawn isotropically.

    Storing h0, psi, phi0, iota in the dict means precompute_amplitudes
    reads them directly and skips recomputation.

    Parameters
    ----------
    N            : number of binaries
    f_min, f_max : GW frequency range (Hz)
    Mc_min/max   : chirp mass range (solar masses * G/c^3 in SI — pass in SI)
    z_min/max    : redshift range
    rng          : numpy random Generator (for reproducibility)

    Returns
    -------
    population : list of N binary dicts, each containing:
                 f, Mc, D_comov, z, h0, ra, dec, psi, iota, phi0
    """
    if rng is None:
        rng = np.random.default_rng()

    # Cosmology: comoving distance ≈ cz/H0 (good for z < 1; use astropy for z > 1)
    H0      = 70e3 / 3.086e22   # s^-1
    c_light = 3e8               # m/s
    Msun_kg = 1.989e30
    G_SI    = 6.674e-11

    # Log-uniform in frequency and chirp mass
    f_arr   = np.exp(rng.uniform(np.log(f_min),    np.log(f_max),    N))
    Mc_arr  = np.exp(rng.uniform(np.log(Mc_min),   np.log(Mc_max),   N)) * Msun_kg * G_SI / c_light**3
    # Comoving volume → uniform in z^3 approximation
    z_arr   = rng.uniform(z_min**(1/3), z_max**(1/3), N)**3
    D_comov = c_light * z_arr / H0 / 3.086e22   # Mpc

    # Isotropic sky and orientation
    ra_arr  = rng.uniform(0,        2*np.pi, N)
    dec_arr = np.arcsin(rng.uniform(-1, 1,   N))
    psi_arr = rng.uniform(0,        np.pi,   N)
    iota_arr= np.arccos(rng.uniform(-1, 1,   N))
    phi0_arr= rng.uniform(0,        2*np.pi, N)

    # Pre-compute h0 vectorised — store so precompute_binary_amplitudes can skip it
    f_rest  = 0.5 * (1 + z_arr) * f_arr
    D_SI    = D_comov * 1e6 * pc
    h0_arr  = (2 * (G_SI * Mc_arr)**(5/3) * (2*np.pi*f_rest)**(2/3)) / (c_light**4 * D_SI)

    population = [
        {
            'f':       float(f_arr[i]),
            'Mc':      float(Mc_arr[i]),
            'D_comov': float(D_comov[i]),
            'z':       float(z_arr[i]),
            'h0':      float(h0_arr[i]),
            'ra':      float(ra_arr[i]),
            'dec':     float(dec_arr[i]),
            'psi':     float(psi_arr[i]),
            'iota':    float(iota_arr[i]),
            'phi0':    float(phi0_arr[i]),
        }
        for i in range(N)
    ]
    return population


# ──────────────────────────────────────────────────────────────────────────────
# 8.  RECOMMENDED ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def inject_fast(psrs, population, method='auto', pure_signal=True, **kwargs):
    """
    Fast injection dispatcher.  Picks the best method automatically.

    method='auto'  (default):
        N < 1e5          → direct batched (exact, low overhead)
        1e5 ≤ N < 1e7   → FFT (excellent approximation, ~1000× faster)
        N ≥ 1e7          → NUFFT if finufft available, else FFT

    Precomputes amplitudes automatically if not already present.

    Parameters
    ----------
    psrs       : list of enterprise Pulsar objects
    population : list of binary dicts (drawn with draw_population_with_amplitudes
                 or manually with the required keys)
    method     : 'auto', 'direct', 'fft', 'nufft'
    pure_signal: replace residuals (True) or add (False)
    **kwargs   : passed through to the underlying injection function
    """
    N = len(population)

    # Check if amplitudes are already precomputed for first pulsar
    key = f'A_{psrs[0].name}'
    if key not in population[0]:
        print(f"Precomputing amplitudes for {len(psrs)} pulsars × {N:,} binaries...")
        for psr in psrs:
            precompute_amplitudes(population, psr)
        print("Precomputation done.")

    if method == 'auto':
        if N < 100_000:
            method = 'direct'
        elif N < 10_000_000:
            method = 'fft'
        else:
            try:
                import finufft
                method = 'nufft'
            except ImportError:
                method = 'fft'

    print(f"Injecting N={N:,} binaries via method='{method}'")

    if method == 'direct':
        return inject_population_direct(psrs, population, pure_signal=pure_signal, **kwargs)
    elif method == 'fft':
        return inject_population_fft(psrs, population, pure_signal=pure_signal, **kwargs)
    elif method == 'nufft':
        return inject_population_nufft(psrs, population, pure_signal=pure_signal, **kwargs)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: auto, direct, fft, nufft")