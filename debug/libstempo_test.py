import libstempo as lt
import numpy as np

# Load with original par file
psr_orig = lt.tempopulsar(parfile='psars_narrowband/par/B1855+09_PINT_20220301.nb.par', 
                          timfile='psars_narrowband/tim/B1855+09_PINT_20220301.nb.tim')
psr_orig.fit()
resid_orig = psr_orig.residuals().copy()

# Load with modified par file  
psr_mod = lt.tempopulsar(parfile='B1855+09_PINT_20220301.nb.par', 
                         timfile='psars_narrowband/tim/B1855+09_PINT_20220301.nb.tim')
psr_mod.fit()
resid_mod = psr_mod.residuals().copy()

# Compare
diff = resid_orig - resid_mod
print(f"Max difference: {np.max(np.abs(diff)):.3e} s")
print(f"RMS difference: {np.std(diff):.3e} s")
print(f"Residual RMS (orig): {np.std(resid_orig):.3e} s")
print(f"Relative difference: {np.std(diff)/np.std(resid_orig):.3e}")