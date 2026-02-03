import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator, AutoMinorLocator


def plot_initial_injection_analysis(psrs_injected, population, snr, xi, rho):
    """
    Generate comprehensive diagnostic plots for initial injection analysis.
    
    Parameters:
    -----------
    psrs_injected : list
        Pulsars with injected signals
    population : list
        SMBHB population
    snr : float
        Signal-to-noise ratio from OS
    xi : array
        ORF values
    rho : array
        Correlation values
    """
    print("\n📊 Generating diagnostic plots...")
    
    # ========== FIGURE 1: Multi-panel residual analysis ==========
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    colors = plt.cm.tab10(np.linspace(0, 1, len(psrs_injected)))
    
    # 1. Time series of residuals
    ax1 = fig.add_subplot(gs[0, :])
    for i, psr in enumerate(psrs_injected[:4]):
        t_mjd = psr.toas / 86400.0
        t_years = (t_mjd - t_mjd.min()) / 365.25
        ax1.plot(t_years, psr.residuals * 1e6, '.', alpha=0.6, 
                label=psr.name, markersize=3, color=colors[i])
    ax1.set_xlabel('Time (years from start)', fontsize=12)
    ax1.set_ylabel('Timing Residual (μs)', fontsize=12)
    ax1.set_title('Injected SMBHB Signal: Time Series', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Residual distribution
    ax2 = fig.add_subplot(gs[1, 0])
    all_residuals = np.concatenate([psr.residuals for psr in psrs_injected])
    ax2.hist(all_residuals * 1e6, bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Timing Residual (μs)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution Across All Pulsars', fontsize=12, fontweight='bold')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. RMS residuals per pulsar
    ax3 = fig.add_subplot(gs[1, 1])
    rms_values = [np.sqrt(np.var(psr.residuals)) * 1e6 for psr in psrs_injected]
    psr_names = [psr.name for psr in psrs_injected]
    x_pos = np.arange(len(psr_names))
    ax3.bar(x_pos, rms_values, color=colors[:len(psrs_injected)], 
            alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Pulsar', fontsize=12)
    ax3.set_ylabel('RMS Residual (μs)', fontsize=12)
    ax3.set_title('RMS by Pulsar', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(psr_names, rotation=45, ha='right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Phase-folded plot
    ax4 = fig.add_subplot(gs[2, 0])
    psr_example = psrs_injected[0]
    t_mjd = psr_example.toas / 86400.0
    t_phase = (t_mjd - t_mjd.min()) / (t_mjd.max() - t_mjd.min())
    scatter = ax4.scatter(t_phase, psr_example.residuals * 1e6, 
                        c=t_mjd - t_mjd.min(), cmap='viridis', 
                        alpha=0.6, s=20)
    ax4.set_xlabel('Observation Phase (0-1)', fontsize=12)
    ax4.set_ylabel('Timing Residual (μs)', fontsize=12)
    ax4.set_title(f'Pattern: {psr_example.name}', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Days from Start')
    
    # 5. Power spectrum
    ax5 = fig.add_subplot(gs[2, 1])
    dt = np.median(np.diff(psr_example.toas))
    fft_vals = np.fft.rfft(psr_example.residuals)
    fft_freq = np.fft.rfftfreq(len(psr_example.residuals), d=dt)
    fft_freq_nHz = fft_freq * 1e9
    power = np.abs(fft_vals)**2
    ax5.loglog(fft_freq_nHz[1:], power[1:], linewidth=1, alpha=0.7)
    ax5.set_xlabel('Frequency (nHz)', fontsize=12)
    ax5.set_ylabel('Power', fontsize=12)
    ax5.set_title(f'Power Spectrum: {psr_example.name}', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, which='both')
    
    # Mark injected frequencies (first 5 binaries)
    for binary in population[:5]:
        f_nHz = binary['f'] * 1e9
        ax5.axvline(f_nHz, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.suptitle(f'SMBHB Injection Analysis | {len(psrs_injected)} Pulsars | SNR={snr:.1f}', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('figures/pulsar_residuals_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: figures/pulsar_residuals_analysis.png")
    plt.close()
    
    # ========== FIGURE 2: Correlation matrix ==========
    fig2, ax = plt.subplots(figsize=(10, 8))
    n_psr = len(psrs_injected)
    corr_matrix = np.zeros((n_psr, n_psr))
    
    for i in range(n_psr):
        for j in range(n_psr):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                res_i = psrs_injected[i].residuals
                res_j = psrs_injected[j].residuals
                if len(res_i) == len(res_j):
                    corr_matrix[i, j] = np.corrcoef(res_i, res_j)[0, 1]
    
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(np.arange(n_psr))
    ax.set_yticks(np.arange(n_psr))
    ax.set_xticklabels(psr_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(psr_names, fontsize=8)
    ax.set_title('Inter-Pulsar Correlation Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Correlation Coefficient')
    
    # Add correlation values
    for i in range(n_psr):
        for j in range(n_psr):
            if not np.isnan(corr_matrix[i, j]):
                ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                    ha="center", va="center", color="black", fontsize=6)
    
    plt.tight_layout()
    plt.savefig('figures/pulsar_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: figures/pulsar_correlation_matrix.png")
    plt.close()

def plot_scaling_results(results, N_needed, target_SNR, config_name):
    """Generate scaling analysis plots."""
    if len(results['N_binaries']) == 0:
        return
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # SNR vs N
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(results['N_binaries'], results['SNR'], 'o-', linewidth=2.5, 
            markersize=8, color='steelblue', label='Computed SNR')
    ax1.axhline(target_SNR, color='red', linestyle='--', linewidth=2.5, 
                label=f'Target ({target_SNR}σ)', alpha=0.8)
    if N_needed:
        ax1.axvline(N_needed, color='orange', linestyle='--', linewidth=2, 
                    label=f'N ≈ {N_needed}', alpha=0.8)
    ax1.set_xlabel('Number of Binaries', fontsize=13, fontweight='bold')
    ax1.set_ylabel('SNR', fontsize=13, fontweight='bold')
    ax1.set_title('SNR Scaling', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    plt.suptitle(f'Scaling Analysis: {config_name}', fontsize=16, fontweight='bold')
    plt.savefig(f'figures/scaling_analysis_{config_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def print_binary_statistics(df, top_n=10):
    """Print comprehensive statistics about top binaries."""
    print(f"\n{'='*80}")
    print(f"TOP {top_n} BINARIES BY SNR")
    print(f"{'='*80}")
    
    top = df.head(top_n)
    
    for i, row in top.iterrows():
        print(f"\nRank {row['final_rank']}:")
        print(f"  SNR: {row['SNR']:.3f} (|SNR| = {row['abs_SNR']:.3f})")
        print(f"  Strain: h₀ = {row['h_0']:.2e}")
        print(f"  Residual: {row['residual_amplitude_us']:.2f} μs")
        print(f"  Chirp mass: {row['chirp_mass_Msun']:.2e} M☉")
        print(f"  Comoving distance: {row['comoving_distance_Mpc']:.1f} Mpc")
        print(f"  Frequency: {row['frequency_nHz']:.3f} nHz")
        print(f"  Sky: RA={row['ra_deg']:.1f}°, Dec={row['dec_deg']:.1f}°")
        print(f"  Nearest pulsar: {row['min_psr_separation_deg']:.1f}° away")
        print(f"  Correlations: {row['pos_corr']} positive, {row['neg_corr']} negative")
    
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS (All {len(df)} binaries)")
    print(f"{'='*80}")
    print(f"SNR:")
    print(f"  Mean: {df['SNR'].mean():.3f}")
    print(f"  Median: {df['SNR'].median():.3f}")
    print(f"  Std: {df['SNR'].std():.3f}")
    print(f"  Range: [{df['SNR'].min():.3f}, {df['SNR'].max():.3f}]")
    
    print(f"\nStrain (h₀):")
    print(f"  Mean: {df['h_0'].mean():.2e}")
    print(f"  Median: {df['h_0'].median():.2e}")
    print(f"  Range: [{df['h_0'].min():.2e}, {df['h_0'].max():.2e}]")
    
    print(f"\nSky Location Impact:")
    print(f"  Mean nearest pulsar: {df['min_psr_separation_deg'].mean():.1f}°")
    print(f"  Median nearest pulsar: {df['min_psr_separation_deg'].median():.1f}°")
    print(f"  Range: [{df['min_psr_separation_deg'].min():.1f}°, {df['min_psr_separation_deg'].max():.1f}°]")
    
    print(f"\nCorrelations with |SNR|:")
    print(f"  vs h₀: {df['abs_SNR'].corr(df['h_0']):.3f}")
    print(f"  vs Comoving Distance: {df['abs_SNR'].corr(df['comoving_distance_Mpc']):.3f}")
    print(f"  vs Chirp Mass: {df['abs_SNR'].corr(df['chirp_mass_Msun']):.3f}")
    print(f"  vs Nearest Pulsar Sep: {df['abs_SNR'].corr(df['min_psr_separation_deg']):.3f}")
    print(f"{'='*80}\n")


def setup_ticks(ax, logx=False, logy=False):
    """Setup publication-quality ticks."""
    if logx:
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    
    if logy:
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, bottom=True, left=True,
                   length=5, width=1)
    ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True, bottom=True, left=True,
                   length=2.5, width=0.7)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1)


def plot_individual_binaries(df, psrs_injected, top_N=50):
    """Generate comprehensive individual binary analysis plots."""
    if df is None or len(df) == 0:
        return
    
    fig = plt.figure(figsize=(30, 24))
    gs = GridSpec(5, 3, figure=fig, hspace=0.35, wspace=0.35)
    df_top = df.head(top_N)
    
    # ==========================================================================
    # 1. SNR ranking (showing sign - positive=green, negative=red)
    # ==========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    # Color by sign: positive = green, negative = red
    bar_colors = ['green' if snr > 0 else 'red' for snr in df_top['SNR']]
    ax1.barh(range(len(df_top)), df_top['SNR'], color=bar_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axvline(0, color='black', linestyle='-', linewidth=1)
    ax1.set_yticks(range(len(df_top)))
    ax1.set_yticklabels([f"#{r}" for r in df_top['final_rank']], fontsize=9)
    ax1.set_xlabel('SNR (with sign)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Top {top_N} Binaries by |SNR|', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    setup_ticks(ax1)
    
    # ==========================================================================
    # 2. Sky map with pulsars (colored by SIGNED SNR)
    # ==========================================================================
    ax2 = fig.add_subplot(gs[0, 1:], projection='mollweide')
    
    # Use signed SNR with diverging colormap
    snr_max = df['abs_SNR'].quantile(0.95)
    scatter = ax2.scatter(
        np.radians(df['ra_deg']) - np.pi,  # Mollweide convention,,
        np.radians(df['dec_deg']),
        c=df['SNR'],  # SIGNED SNR
        s=100 * df['abs_SNR'] / df['abs_SNR'].max(),
        cmap='RdBu_r',  # Red=negative, Blue=positive
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5,
        vmin=-snr_max,
        vmax=snr_max
    )
    
    if psrs_injected is not None and len(psrs_injected) > 0:
        pulsar_ra = np.array([psr._raj for psr in psrs_injected])
        pulsar_ra_moll = pulsar_ra - np.pi  # Mollweide convention
        pulsar_dec = np.array([psr._decj for psr in psrs_injected])
        
        ax2.scatter(
            pulsar_ra_moll,
            pulsar_dec,
            marker='*',
            s=90,
            color='white',
            edgecolor='black',
            linewidth=0.4,
            alpha=1.0,
            label='Pulsars',
            zorder=10
        )
    
    ax2.legend(loc="lower left", fontsize=10)
    ax2.set_xlabel('Right Ascension', fontsize=11)
    ax2.set_ylabel('Declination', fontsize=11)
    ax2.set_title('Sky Distribution (size ∝ |SNR|, color = sign)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter, ax=ax2, label='SNR (red=neg, blue=pos)', pad=0.1)
    
    # ==========================================================================
    # 3. Frequency vs SNR
    # ==========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    scatter3 = ax3.scatter(
        df['frequency_nHz'],
        df['abs_SNR'],
        c=df['chirp_mass_Msun'],
        s=50,
        alpha=0.6,
        cmap='viridis',
        edgecolors='none',
        norm=plt.cm.colors.LogNorm()
    )
    ax3.set_xlabel('GW Frequency (nHz)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('|SNR|', fontsize=11, fontweight='bold')
    ax3.set_title('SNR vs Frequency', fontsize=12, fontweight='bold')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    setup_ticks(ax3, logx=True, logy=True)
    plt.colorbar(scatter3, ax=ax3, label='Chirp Mass (M☉)')
    
    # ==========================================================================
    # 4. Mass vs Distance (colored by SNR)
    # ==========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    scatter4 = ax4.scatter(
        df['chirp_mass_Msun'],
        df['comoving_distance_Mpc'],
        c=df['SNR'],
        s=100,
        alpha=0.6,
        cmap='coolwarm',
        edgecolors='none'
    )
    ax4.set_xlabel('Chirp Mass (M☉)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Comoving Distance (Mpc)', fontsize=11, fontweight='bold')
    ax4.set_title('Mass-Distance Distribution', fontsize=12, fontweight='bold')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    setup_ticks(ax4, logx=True, logy=True)
    plt.colorbar(scatter4, ax=ax4, label='SNR')
    
    # ==========================================================================
    # 5. Strain amplitude vs SNR
    # ==========================================================================
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(df['h_0'], df['abs_SNR'], alpha=0.6, s=50, edgecolors='none', color='steelblue')
    ax5.set_xlabel('Strain Amplitude h₀', fontsize=11, fontweight='bold')
    ax5.set_ylabel('|SNR|', fontsize=11, fontweight='bold')
    ax5.set_title('Strain vs SNR', fontsize=12, fontweight='bold')
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    setup_ticks(ax5, logx=True, logy=True)
    
    # Fit power law
    log_h0 = np.log10(df['h_0'])
    log_snr = np.log10(df['abs_SNR'])
    valid = np.isfinite(log_h0) & np.isfinite(log_snr)
    if np.sum(valid) > 2:
        coeffs = np.polyfit(log_h0[valid], log_snr[valid], 1)
        h0_fit = np.logspace(log_h0[valid].min(), log_h0[valid].max(), 100)
        snr_fit = 10**(coeffs[0] * np.log10(h0_fit) + coeffs[1])
        ax5.plot(h0_fit, snr_fit, 'r--', linewidth=2, alpha=0.7, 
                label=f'SNR ∝ h₀$^{{{coeffs[0]:.2f}}}$')
        ax5.legend(fontsize=9)
    
    # ==========================================================================
    # 6. NEW: Nearest Pulsar Separation vs SNR (SIGNED, colored by h₀)
    # ==========================================================================
    ax6 = fig.add_subplot(gs[2, 0])
    scatter6 = ax6.scatter(
        df['min_psr_separation_deg'],
        df['SNR'],  # SIGNED SNR
        c=df['h_0'],
        s=60,
        alpha=0.7,
        cmap='plasma',
        edgecolors='none',
        norm=plt.cm.colors.LogNorm()
    )
    ax6.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax6.set_xlabel('Nearest Pulsar Separation (deg)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('SNR (with sign)', fontsize=11, fontweight='bold')
    ax6.set_title('Sky Location Impact on SNR', fontsize=12, fontweight='bold')
    setup_ticks(ax6)
    cbar6 = plt.colorbar(scatter6, ax=ax6, label='h₀')
    
    # ==========================================================================
    # 7. NEW: Distance vs h₀ (colored by nearest pulsar - shows sky importance)
    # ==========================================================================
    ax7 = fig.add_subplot(gs[2, 1])
    scatter7 = ax7.scatter(
        df['comoving_distance_Mpc'],
        df['h_0'],
        c=df['min_psr_separation_deg'],
        s=60,
        alpha=0.7,
        cmap='coolwarm',
        edgecolors='none'
    )
    ax7.set_xlabel('Comoving Distance (Mpc)', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Strain h₀', fontsize=11, fontweight='bold')
    ax7.set_title('Distance vs Strain (color = pulsar sep)', fontsize=12, fontweight='bold')
    ax7.set_xscale('log')
    ax7.set_yscale('log')
    setup_ticks(ax7, logx=True, logy=True)
    
    # Add 1/D reference
    D_range = np.logspace(np.log10(df['comoving_distance_Mpc'].min()), 
                          np.log10(df['comoving_distance_Mpc'].max()), 100)
    h0_ref = df['h_0'].median() * df['comoving_distance_Mpc'].median() / D_range
    ax7.plot(D_range, h0_ref, 'k--', linewidth=1.5, alpha=0.6, label=r'$\propto 1/D_{\rm{comov}}$')
    ax7.legend(fontsize=9)
    cbar7 = plt.colorbar(scatter7, ax=ax7, label='Nearest Psr (deg)')
    
    # ==========================================================================
    # 8. Residual amplitude distribution (separated by SNR sign)
    # ==========================================================================
    ax8 = fig.add_subplot(gs[2, 2])
    positive_snr = df[df['SNR'] > 0]['residual_amplitude_us']
    negative_snr = df[df['SNR'] < 0]['residual_amplitude_us']
    
    ax8.hist(positive_snr, bins=30, alpha=0.6, edgecolor='none', color='blue', label='SNR > 0')
    ax8.hist(negative_snr, bins=30, alpha=0.6, edgecolor='none', color='red', label='SNR < 0')
    ax8.set_xlabel('Timing Residual Amplitude (μs)', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax8.set_title('Residual Amplitude Distribution', fontsize=12, fontweight='bold')
    ax8.set_yscale('log')
    ax8.legend(fontsize=9)
    setup_ticks(ax8, logy=True)
    
    # ==========================================================================
    # 9. Correlation analysis for top binaries
    # ==========================================================================
    ax9 = fig.add_subplot(gs[3, 0])
    pos_corr = df_top['pos_corr'].values
    neg_corr = df_top['neg_corr'].values
    x = np.arange(len(df_top))
    width = 0.35
    ax9.bar(x - width/2, pos_corr, width, label='Positive ρ', color='green', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax9.bar(x + width/2, neg_corr, width, label='Negative ρ', color='red', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax9.set_xlabel('Binary Rank', fontsize=11, fontweight='bold')
    ax9.set_ylabel('Number of Pulsar Pairs', fontsize=11, fontweight='bold')
    ax9.set_title('Correlation Sign Distribution', fontsize=12, fontweight='bold')
    ax9.set_xticks(x[::2])  # Every other label
    ax9.set_xticklabels([f"#{r}" for r in df_top['final_rank'].iloc[::2]], rotation=45, fontsize=8)
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3, axis='y')
    setup_ticks(ax9)
    
    # ==========================================================================
    # 10. SNR vs OS (consistency check)
    # ==========================================================================
    ax10 = fig.add_subplot(gs[3, 1])
    ax10.scatter(df['OS'], df['SNR'], alpha=0.6, s=50, c=df['abs_SNR'], 
                cmap='plasma', edgecolors='none')
    ax10.set_xlabel('Optimal Statistic', fontsize=11, fontweight='bold')
    ax10.set_ylabel('SNR', fontsize=11, fontweight='bold')
    ax10.set_title('OS vs SNR', fontsize=12, fontweight='bold')
    ax10.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax10.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    setup_ticks(ax10)
    
    # ==========================================================================
    # 11. Cumulative SNR contribution
    # ==========================================================================
    ax11 = fig.add_subplot(gs[3, 2])
    
    df_sorted = df.sort_values(by='abs_SNR', ascending=False)
    snr_vals = df_sorted['SNR'].values
    snr_sq = snr_vals**2
    cumulative_snr = np.sqrt(np.cumsum(snr_sq))
    fractional_snr = cumulative_snr / cumulative_snr[-1]
    ranks = np.arange(1, len(cumulative_snr) + 1)
    
    line_frac, = ax11.plot(ranks, fractional_snr, marker='o', linewidth=2, 
                           markersize=3, label='Fractional')
    ax11.set_xlabel('Number of loudest binaries', fontsize=11, fontweight='bold')
    ax11.set_ylabel('Fractional cumulative SNR', fontsize=11, fontweight='bold')
    ax11.set_ylim(0, 1.05)
    
    # Markers
    idx_50 = np.argmax(fractional_snr >= 0.5) + 1
    idx_90 = np.argmax(fractional_snr >= 0.9) + 1
    ax11.axvline(idx_50, linestyle='--', alpha=0.6, color='gray')
    ax11.axvline(idx_90, linestyle='--', alpha=0.6, color='gray')
    ax11.axhline(0.5, linestyle=':', alpha=0.5, color='gray')
    ax11.axhline(0.9, linestyle=':', alpha=0.5, color='gray')
    ax11.text(idx_50 * 1.05, 0.52, f'50% @ {idx_50}', fontsize=9)
    ax11.text(idx_90 * 1.05, 0.92, f'90% @ {idx_90}', fontsize=9)
    
    # Secondary axis
    ax11b = ax11.twinx()
    line_abs, = ax11b.plot(ranks, cumulative_snr, linestyle='-', linewidth=2, 
                           alpha=0.7, color='orange', label='Absolute')
    ax11b.set_ylabel('Cumulative SNR', fontsize=11, fontweight='bold')
    
    lines = [line_frac, line_abs]
    labels = [l.get_label() for l in lines]
    ax11.legend(lines, labels, fontsize=9, loc='lower right')
    ax11.set_title('Cumulative SNR Contribution', fontsize=12, fontweight='bold')
    setup_ticks(ax11)
    
    # ==========================================================================
    # 12. NEW: Sky location heatmap (RA vs Dec, SIGNED SNR average)
    # ==========================================================================
    ax12 = fig.add_subplot(gs[4, :])
    
    # Create 2D histogram with SIGNED SNR
    ra_bins = np.linspace(0, 360, 36)
    dec_bins = np.linspace(-90, 90, 18)
    H, xedges, yedges = np.histogram2d(df['ra_deg'], df['dec_deg'], 
                                        bins=[ra_bins, dec_bins],
                                        weights=df['SNR'])  # SIGNED SNR
    counts, _, _ = np.histogram2d(df['ra_deg'], df['dec_deg'], 
                                   bins=[ra_bins, dec_bins])
    
    # Average SNR per bin
    with np.errstate(divide='ignore', invalid='ignore'):
        H_avg = H / counts
        H_avg[~np.isfinite(H_avg)] = 0
    
    # Use diverging colormap centered at 0
    vmax = np.abs(H_avg).max()
    im = ax12.imshow(H_avg.T, origin='lower', aspect='auto', cmap='RdBu_r',
                     extent=[0, 360, -90, 90], interpolation='nearest',
                     vmin=-vmax, vmax=vmax)
    
    # Add pulsar locations
    if psrs_injected is not None:
        pulsar_ra_deg = np.array([np.degrees(psr._raj) for psr in psrs_injected])
        pulsar_dec_deg = np.array([np.degrees(psr._decj) for psr in psrs_injected])
        ax12.scatter(pulsar_ra_deg, pulsar_dec_deg, marker='*', s=200, 
                    color='yellow', edgecolor='black', linewidth=1.5, 
                    alpha=0.9, label='Pulsars', zorder=10)
    
    ax12.set_xlabel('Right Ascension (deg)', fontsize=11, fontweight='bold')
    ax12.set_ylabel('Declination (deg)', fontsize=11, fontweight='bold')
    ax12.set_title('Sky Distribution Heatmap: Average SNR per Region (red=negative, blue=positive)', 
                   fontsize=12, fontweight='bold')
    ax12.legend(fontsize=10, loc='upper right')
    ax12.grid(True, alpha=0.3, color='white', linewidth=0.5)
    cbar12 = plt.colorbar(im, ax=ax12, label='Average SNR', pad=0.01)
    setup_ticks(ax12)
    
    plt.suptitle(f'Individual Binary Analysis | {len(df)} Binaries | Top {top_N} Shown', 
                 fontsize=16, fontweight='bold')
    
    plt.savefig('figures/individual_binary_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: figures/individual_binary_analysis.png")
    
    plt.close()


def plot_ensemble_results(ensemble_results, save_plots=True):
    """Plot ensemble results."""
    if not isinstance(ensemble_results, list):
        ensemble_results = [ensemble_results]

    n_configs = len(ensemble_results)
    
    fig = plt.figure(figsize=(18, 6 * n_configs))
    gs = GridSpec(n_configs, 3, figure=fig, hspace=0.3, wspace=0.3)

    colors = plt.cm.Set2(np.linspace(0, 1, n_configs))
    
    for idx, results in enumerate(ensemble_results):
        N_array = np.array(results['N_needed_list'])
        pop_size = results['config']['N_binaries']
        fractions = 100 * N_array / pop_size
        
        # Histogram
        ax1 = fig.add_subplot(gs[idx, 0])
        n_bins = min(20, len(N_array) // 2)
        ax1.hist(N_array, bins=20, alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(N_array), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(N_array):.0f}')
        ax1.set_xlabel('N_binaries Required', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax1.set_title(f'{results["config_name"].upper()}', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)


        # 2. Histogram of fractions
        ax2 = fig.add_subplot(gs[idx, 1])
        ax2.hist(fractions, bins=n_bins, alpha=0.7, 
                edgecolor='black', color=colors[idx])
        ax2.axvline(np.mean(fractions), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(fractions):.1f}%')
        ax2.axvline(np.median(fractions), color='orange', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(fractions):.1f}%')
        
        ax2.set_xlabel('% of Population Needed', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax2.set_title(f'Fraction Distribution', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Cumulative distribution
        ax3 = fig.add_subplot(gs[idx, 2])
        sorted_N = np.sort(N_array)
        cumulative = np.arange(1, len(sorted_N) + 1) / len(sorted_N)
        ax3.plot(sorted_N, cumulative, linewidth=2.5, color=colors[idx])
        
        # Mark key percentiles
        for percentile, label, color in [(16, '16%', 'blue'), 
                                          (50, '50%', 'orange'), 
                                          (84, '84%', 'red')]:
            val = np.percentile(N_array, percentile)
            ax3.axvline(val, color=color, linestyle='--', alpha=0.7, 
                       linewidth=1.5, label=f'{label}: {val:.0f}')
            ax3.axhline(percentile/100, color=color, linestyle=':', 
                       alpha=0.5, linewidth=1)
        
        ax3.set_xlabel('N_binaries Required', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
        ax3.set_title(f'CDF', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10, loc='lower right')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])
    
    
    plt.suptitle('Ensemble Analysis Results', fontsize=16, fontweight='bold')
    if save_plots:
        plt.savefig('figures/ensemble_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_binaries_vs_frequency(
    population, 
    mass_bins=None, freq_bins=None, 
    candidate_frequencies=None,
    candidate_labels=None,
    candidate_masses=None,   # NEW
    subset_name='Subset',
    n_freq_bins=30
):
    """
    Plot number of binaries as a function of GW frequency, split by total mass bins.

    Parameters
    ----------
    population : list of dicts
        Population of binaries. Each dict must have keys: 'f', 'Mc', 'q' or 'M1', 'M2'
    mass_bins : array-like, optional
        Log10(total_mass/Msun) bin edges. Default: np.arange(7.5, 10.1, 0.5)
    freq_bins : array-like, optional
        Frequency bins. Default: log-spaced from min(fGW) to max(fGW), n_freq_bins bins.
    candidate_frequencies : array-like, optional
        Frequencies to mark as candidate events (vertical lines).
    candidate_labels : array-like, optional
        Labels for candidate frequencies.
    subset_name : str
        Name for the plot title and legend.
    n_freq_bins : int
        Number of frequency bins if freq_bins is None.
    """
    from config import Msun
    
    # Extract frequencies
    fGW = np.array([b['f'] for b in population])
    
    # Extract or compute total masses
    total_mass = []
    for b in population:
        if 'Mtot' in b:
            M_tot = b['Mtot'] * Msun
        elif 'M1' in b and 'M2' in b:
            M_tot = b['M1'] + b['M2']
        elif 'Mc' in b and 'q' in b:
            # Convert chirp mass + mass ratio to total mass
            # Mc = (M1 * M2)^(3/5) / (M1 + M2)^(1/5)
            # q = M2/M1, M_tot = M1 + M2 = M1(1+q)
            # Mc = M1^(3/5) * (M1*q)^(3/5) / (M1(1+q))^(1/5)
            # Mc = M1 * q^(3/5) / (1+q)^(1/5)
            # M_tot = Mc * (1+q)^(6/5) / q^(3/5)
            q = b['q']
            Mc = b['Mc']
            M_tot = Mc * (1 + q)**(6/5) / q**(3/5)
        else:
            raise ValueError("Population must have 'M_total', or ('M1','M2'), or ('Mc','q')")
        total_mass.append(M_tot / Msun)  # Convert to solar masses
    
    total_mass = np.array(total_mass)
    
    # Default mass bins
    if mass_bins is None:
        mass_bins = np.arange(7.5, 10.6, 0.5)
    
    # Default frequency bins (log-spaced)
    if freq_bins is None:
        freq_bins = np.logspace(np.log10(fGW.min()), np.log10(fGW.max()), n_freq_bins)
    
    plt.figure(figsize=(8, 6))
    
    # Total histogram - solid black line
    counts_total, _ = np.histogram(fGW, bins=freq_bins)
    bin_centers = (freq_bins[:-1] + freq_bins[1:]) / 2
    plt.plot(bin_centers, counts_total, color='k', linewidth=2, linestyle='-', 
             label='Total', drawstyle='steps-mid')
    
    # Histograms for each mass range - different colors and line styles
    colors = plt.cm.viridis(np.linspace(0, 1, len(mass_bins) - 1))
    colors = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # purple
    "#56B4E9",  # sky blue
    "#000000",  # black
]
    linestyles = ['--']  # Cycle through if more than 6 bins
    
    for i in range(len(mass_bins) - 1):
        mask = (np.log10(total_mass) >= mass_bins[i]) & (np.log10(total_mass) < mass_bins[i+1])
        if np.any(mask):
            counts, _ = np.histogram(fGW[mask], bins=freq_bins)
            linestyle = linestyles[i % len(linestyles)]
            plt.plot(bin_centers, counts, color=colors[i], linewidth=2, 
                     linestyle=linestyle, drawstyle='steps-mid',
                     label=r"$%.1f < \log_{10}\!\left(M_{\rm tot}/\mathrm{M}_\odot\right) < %.1f$" % (mass_bins[i], mass_bins[i+1]))
    
    # Candidate frequencies (optional)
    colours = ['r', 'm', 'c', 'y']

    if candidate_masses is None:
        candidate_masses = [None] * len(candidate_frequencies)

    for i, (f, label, m) in enumerate(zip(candidate_frequencies, candidate_labels, candidate_masses)):

        if m is not None:
            legend_label = (
                rf"{label} "
                rf"$\left[\log_{{10}}\!\left(M_{{\rm tot}}/M_\odot\right)={m:.2f}\right]$"
            )
        else:
            legend_label = label

        plt.axvline(
            f,
            color=colours[i],
            linestyle='-',
            lw=3,
            alpha=1,
            label=legend_label
        )
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Gravitational Wave Frequency [Hz]', fontsize=12, fontweight='bold')
    plt.ylabel('Number of binaries', fontsize=12, fontweight='bold')
    plt.title(f'Binaries by GW frequency ({subset_name})', fontsize=13, fontweight='bold')
    
    # Setup ticks
    ax = plt.gca()
    from matplotlib.ticker import LogLocator, AutoMinorLocator
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, bottom=True, left=True,
                   length=5, width=1)
    ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True, bottom=True, left=True,
                   length=2.5, width=0.7)
    
    # axis limits
    plt.xlim(freq_bins[0], freq_bins[-1] * 1.2)
    plt.ylim(0.2, counts_total.max() * 1.2)
    
    # Add box around plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1)
    
    # Clean legend (avoid duplicate entries)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=9, frameon=True, 
               framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig('figures/nanograv_comparison_plot.png', dpi=300)
    print("✓ Saved: figures/nanograv_comparison_plot.png")
    plt.show()

def plot_binaries_vs_frequency_mc(
    populations,
    mass_bins=None,
    freq_bins=None,
    candidate_frequencies=None,
    candidate_labels=None,
    candidate_masses=None,
    subset_name='Subset',
    n_freq_bins=30,
    ci=(5, 95),
):
    """
    Plot number of binaries vs GW frequency using many population realisations.
    Shows median and percentile envelopes to reduce Monte Carlo noise.

    Parameters
    ----------
    populations : list of populations
        Each element is a population realisation (list of dicts).
    mass_bins : array-like, optional
        log10(Mtot/Msun) bin edges.
    freq_bins : array-like, optional
        Frequency bin edges.
    candidate_frequencies, candidate_labels, candidate_masses : optional
        Candidate vertical lines and legend info.
    ci : tuple
        Percentile range to plot (e.g. (5, 95)).
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from config import Msun
    from matplotlib.ticker import LogLocator

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------
    if mass_bins is None:
        mass_bins = np.arange(7.5, 10.6, 0.5)

    if freq_bins is None:
        all_f = np.concatenate([[b['f'] for b in pop] for pop in populations])
        freq_bins = np.logspace(
            np.log10(all_f.min()),
            np.log10(all_f.max()),
            n_freq_bins
        )

    bin_centers = 0.5 * (freq_bins[:-1] + freq_bins[1:])
    n_mass = len(mass_bins) - 1

    # ------------------------------------------------------------------
    # Helper: histogram for one population
    # ------------------------------------------------------------------
    def compute_histograms(pop):
        fGW = np.array([b['f'] for b in pop])

        total_mass = []
        for b in pop:
            if 'Mtot' in b:
                M_tot = b['Mtot'] * Msun
            elif 'M1' in b and 'M2' in b:
                M_tot = b['M1'] + b['M2']
            elif 'Mc' in b and 'q' in b:
                q = b['q']
                Mc = b['Mc']
                M_tot = Mc * (1 + q)**(6/5) / q**(3/5)
            else:
                raise ValueError("Binary mass information missing")

            total_mass.append(M_tot / Msun)

        total_mass = np.array(total_mass)

        h_total, _ = np.histogram(fGW, bins=freq_bins)

        h_mass = np.zeros((n_mass, len(freq_bins) - 1))
        for i in range(n_mass):
            mask = (
                (np.log10(total_mass) >= mass_bins[i]) &
                (np.log10(total_mass) <  mass_bins[i+1])
            )
            h_mass[i], _ = np.histogram(fGW[mask], bins=freq_bins)

        return h_total, h_mass

    # ------------------------------------------------------------------
    # Stack all realisations
    # ------------------------------------------------------------------
    all_total = []
    all_mass = []

    for pop in populations:
        ht, hm = compute_histograms(pop)
        all_total.append(ht)
        all_mass.append(hm)

    all_total = np.array(all_total)  # (Nreal, Nfreq)
    all_mass  = np.array(all_mass)   # (Nreal, Nmass, Nfreq)

    lo, hi = ci

    total_med = np.median(all_total, axis=0)
    total_lo  = np.percentile(all_total, lo, axis=0)
    total_hi  = np.percentile(all_total, hi, axis=0)

    mass_med = np.median(all_mass, axis=0)
    mass_lo  = np.percentile(all_mass, lo, axis=0)
    mass_hi  = np.percentile(all_mass, hi, axis=0)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # Total population
    ax.plot(bin_centers, total_med, color='k', lw=2, label='Total (median)')
    ax.fill_between(
        bin_centers, total_lo, total_hi,
        color='k', alpha=0.2, label=f'{lo}–{hi}% range'
    )

    # Colour-blind safe palette (Okabe–Ito)
    colors = [
        "#0072B2", "#E69F00", "#009E73",
        "#D55E00", "#CC79A7", "#56B4E9"
    ]

    for i in range(n_mass):
        ax.plot(
            bin_centers,
            mass_med[i],
            color=colors[i % len(colors)],
            lw=2,
            linestyle='--',
            label=r"$%.1f < \log_{10}\!\left(M_{\rm tot}/M_\odot\right) < %.1f$"
                  % (mass_bins[i], mass_bins[i+1])
        )
        ax.fill_between(
            bin_centers,
            mass_lo[i],
            mass_hi[i],
            color=colors[i % len(colors)],
            alpha=0.25
        )

    # ------------------------------------------------------------------
    # Candidate frequencies
    # ------------------------------------------------------------------
    if candidate_frequencies is not None and candidate_labels is not None:

        if candidate_masses is None:
            candidate_masses = [None] * len(candidate_frequencies)

        cand_colors = ['r', 'm', 'c', 'y']

        for i, (f, label, m) in enumerate(
            zip(candidate_frequencies, candidate_labels, candidate_masses)
        ):
            if m is not None:
                label = (
                    rf"{label} "
                    rf"$[\log_{{10}}(M_{{\rm tot}}/M_\odot)={m:.2f}]$"
                )

            ax.axvline(
                f,
                color=cand_colors[i % len(cand_colors)],
                lw=3,
                linestyle='-',
                label=label
            )

    # ------------------------------------------------------------------
    # Axes, ticks, legend
    # ------------------------------------------------------------------
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('Gravitational Wave Frequency [Hz]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of binaries', fontsize=12, fontweight='bold')
    ax.set_title(f'Binaries by GW frequency ({subset_name})', fontsize=13, fontweight='bold')

    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))

    ax.tick_params(which='both', direction='in', top=True, right=True)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(), by_label.keys(),
        fontsize=9, frameon=True, edgecolor='black'
    )

    plt.tight_layout()
    plt.savefig('figures/nanograv_comparison_plot_mc.png', dpi=300)
    print("✓ Saved: figures/nanograv_comparison_plot_mc.png")
    plt.show()