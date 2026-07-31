# apj_style.py
import matplotlib as mpl

def apply_apj_style():
    mpl.rcParams.update({
        'font.family':       'serif',
        'font.serif':        ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset':  'stix',
        'font.size':         12,
        'axes.titlesize':    12,
        'axes.labelsize':    12,
        'xtick.labelsize':   10,
        'ytick.labelsize':   10,
        'legend.fontsize':   10,
        'figure.figsize':    (3.5, 2.8),
        'axes.linewidth':    0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.direction':   'in',
        'ytick.direction':   'in',
        'xtick.top':         True,
        'ytick.right':       True,
        'axes.spines.top':    True,
        'axes.spines.right':  True,
        'axes.spines.left':   True,
        'axes.spines.bottom': True,
        'savefig.dpi':       300,
        'savefig.bbox':      'tight',
    })

APJ_COL_WIDTH = 3.5
APJ_COL_WIDTH_DOUBLE = 7.0
APJ_FIGSIZE = (APJ_COL_WIDTH, 2.8)