# Shared publication figure style for the geometry-of-forgetting paper.
# Import AFTER `matplotlib.use("Agg")`:  `import mpl_style`
# Keeps DejaVu Sans (robust unicode/mathtext); improves spines, grid, fonts,
# line widths and the default color cycle. Hard-coded colors (green/red/etc.)
# in the generators are left untouched so caption references stay valid.
import matplotlib as mpl

mpl.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "font.size": 11.5,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "lines.linewidth": 2.1,
    "lines.markersize": 5,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.prop_cycle": mpl.cycler(color=[
        "#2b6cb0", "#c0392b", "#27875a", "#8854d0",
        "#d68910", "#16a085", "#7f8c8d", "#2c3e50"]),
})
