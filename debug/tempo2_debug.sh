#!/bin/bash
# ============================================
# FIX: TEMPO2 Clock File Error
# ============================================

# 1. VERIFY TEMPO2 INSTALLATION
echo "=== Checking TEMPO2 Installation ==="
export TEMPO2=$CONDA_PREFIX/share/tempo2

# Check if directory exists
if [ ! -d "$TEMPO2" ]; then
    echo "ERROR: TEMPO2 directory not found at $TEMPO2"
    echo "Searching for tempo2..."
    find $CONDA_PREFIX -name "tempo2" -type d 2>/dev/null
    exit 1
fi

echo "✓ TEMPO2 directory: $TEMPO2"

# 2. SET ALL REQUIRED TEMPO2 PATHS
export TEMPO2_CLOCK_DIR=$TEMPO2/clock
export TEMPO2_EPHEM_DIR=$TEMPO2/ephemeris

# Critical: Also set these alternative variable names
export CLOCK_DIR=$TEMPO2_CLOCK_DIR
export EPHEM_DIR=$TEMPO2_EPHEM_DIR

echo "✓ TEMPO2_CLOCK_DIR: $TEMPO2_CLOCK_DIR"
echo "✓ TEMPO2_EPHEM_DIR: $TEMPO2_EPHEM_DIR"

# 3. VERIFY CLOCK FILES EXIST
echo ""
echo "=== Checking Clock Files ==="
if [ ! -d "$TEMPO2_CLOCK_DIR" ]; then
    echo "ERROR: Clock directory not found!"
    echo "Searching for .clk files..."
    find $CONDA_PREFIX -name "*.clk" 2>/dev/null | head -5
    exit 1
fi

CLK_COUNT=$(ls $TEMPO2_CLOCK_DIR/*.clk 2>/dev/null | wc -l)
echo "Found $CLK_COUNT clock files:"
ls $TEMPO2_CLOCK_DIR/*.clk 2>/dev/null | head -5

if [ $CLK_COUNT -eq 0 ]; then
    echo "ERROR: No clock files found! TEMPO2 may not be properly installed."
    exit 1
fi

# 4. VERIFY EPHEMERIS FILES
echo ""
echo "=== Checking Ephemeris Files ==="
if [ ! -d "$TEMPO2_EPHEM_DIR" ]; then
    echo "WARNING: Ephemeris directory not found!"
fi

EPHEM_COUNT=$(ls $TEMPO2_EPHEM_DIR/*.dat 2>/dev/null | wc -l)
echo "Found $EPHEM_COUNT ephemeris files"

# 5. TEST TEMPO2 BINARY
echo ""
echo "=== Testing TEMPO2 Binary ==="
if command -v tempo2 &> /dev/null; then
    tempo2 -v
else
    echo "WARNING: tempo2 command not found in PATH"
fi

# 6. TEST PYTHON IMPORT
echo ""
echo "=== Testing Enterprise/libstempo ==="
python << 'PYEOF'
import os
print(f"Python sees TEMPO2={os.environ.get('TEMPO2', 'NOT SET')}")
print(f"Python sees TEMPO2_CLOCK_DIR={os.environ.get('TEMPO2_CLOCK_DIR', 'NOT SET')}")

try:
    import libstempo
    print("✓ libstempo imported successfully")
except ImportError as e:
    print(f"✗ libstempo import failed: {e}")

try:
    from enterprise.pulsar import Pulsar
    print("✓ enterprise.pulsar imported successfully")
except ImportError as e:
    print(f"✗ enterprise import failed: {e}")
PYEOF

# 7. TEST SINGLE PULSAR LOAD
echo ""
echo "=== Testing Single Pulsar Load ==="
python << 'PULSAREOF'
from enterprise.pulsar import Pulsar
import os

# Ensure environment is set in Python too
os.environ['TEMPO2'] = os.environ.get('TEMPO2', '')
os.environ['TEMPO2_CLOCK_DIR'] = os.environ.get('TEMPO2_CLOCK_DIR', '')
os.environ['TEMPO2_EPHEM_DIR'] = os.environ.get('TEMPO2_EPHEM_DIR', '')

par = "./psars_narrowband/par/J0030+0451_PINT_20220302.nb.par"
tim = "./psars_narrowband/tim/J0030+0451_PINT_20220302.nb.tim"

print(f"Loading: {par}")
try:
    psr = Pulsar(par, tim, timing_package='tempo2', drop_t2pulsar=False)
    print(f"✓✓✓ SUCCESS! Loaded {len(psr.toas)} TOAs")
    print(f"    Pulsar: {psr.name}")
    print(f"    TOA range: {psr.toas.min():.2f} - {psr.toas.max():.2f} MJD")
except Exception as e:
    print(f"✗✗✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
PULSAREOF

echo ""
echo "=== Diagnostic Complete ==="
echo ""
echo "If still failing, try:"
echo "  1. Reinstall tempo2: mamba install -c conda-forge tempo2"
echo "  2. Check .par/.tim files are valid"
echo "  3. Add to your ~/.bashrc:"
echo "      export TEMPO2=\$CONDA_PREFIX/share/tempo2"
echo "      export TEMPO2_CLOCK_DIR=\$TEMPO2/clock"
echo "      export TEMPO2_EPHEM_DIR=\$TEMPO2/ephemeris"
