import matplotlib.pyplot as plt
import matplotlib as mpl

def apply_btc_style():
    """Apply a clean whitegrid style similar to the BTC example."""
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update({
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'axes.edgecolor': '#cccccc',
        'grid.color': '#eaeaea',
        'grid.linestyle': '-',
        'grid.linewidth': 0.8,
        'figure.dpi': 110,
        'savefig.bbox': 'tight'
    })
