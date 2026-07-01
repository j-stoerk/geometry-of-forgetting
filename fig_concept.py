"""
Geometry of Forgetting -- conceptual (non-data) figures.

  fig_geometry_schematic.pdf   the interference functional Delta^T Sigma Delta and
                               the removable (disjoint) vs unavoidable (overlap) regimes
  fig_algorithm_hierarchy.pdf  the algorithm family: naive -> orthogonal projection
                               -> interference-gated allocation; offline merge / online
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

BLUE, RED, GREEN, ORANGE, GREY = "#4C72B0", "#C44E52", "#55A868", "#DD8452", "#666666"


def arrow(ax, p, q, color="k", lw=1.8, style="-|>", ms=10, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls, zorder=5))


# ---------------------------------------------------------------------------
# Figure 1: geometry schematic
# ---------------------------------------------------------------------------
def geometry_schematic():
    fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.9))
    for a in ax:
        a.set_aspect("equal"); a.set_xlim(-3.2, 3.2); a.set_ylim(-3.0, 3.0)
        a.set_xticks([]); a.set_yticks([])

    # --- (a) interference functional: Delta split by Sigma_A geometry ---
    a = ax[0]
    # Sigma_A iso-loss contours: active = horizontal (steep), null = vertical (flat)
    for r in (1.0, 2.0, 3.0):
        a.add_patch(Ellipse((0, 0), width=1.1 * r, height=5.6, fill=False,
                            edgecolor=BLUE, lw=1.0, alpha=0.55))
    a.text(0.0, 2.55, r"iso-loss of $\Sigma_A$", color=BLUE, ha="center", fontsize=8.5)
    a.plot(0, 0, "o", color=BLUE, ms=6)
    a.text(0.12, -0.36, r"$w_A^\ast$", color=BLUE, fontsize=11)
    # Delta and its split
    tip = np.array([2.05, 1.7])
    arrow(a, (0, 0), tip, color="k", lw=2.0)
    a.text(tip[0] + 0.05, tip[1] + 0.06, r"$\Delta=w_B-w_A^\ast$", fontsize=10)
    arrow(a, (0, 0), (tip[0], 0), color=RED, lw=2.0)
    arrow(a, (tip[0], 0), tip, color=GREEN, lw=2.0)
    a.text(1.0, -0.42, r"$\Delta_\parallel$", color=RED, fontsize=11)
    a.text(tip[0] + 0.06, 0.8, r"$\Delta_\perp$", color=GREEN, fontsize=11)
    a.text(0, -2.78, r"interference $=\Delta^\top\Sigma_A\Delta=\Delta_\parallel^\top\Sigma_A\Delta_\parallel$",
           ha="center", fontsize=9.5)
    a.set_title("(a) Forgetting is an energy in the input metric", fontsize=9.5)

    # --- (b) disjoint supports: removable ---
    a = ax[1]
    a.add_patch(Ellipse((-0.7, 0), 1.0, 4.6, fill=False, edgecolor=BLUE, lw=1.3))
    a.add_patch(Ellipse((0, 0.7), 4.6, 1.0, fill=False, edgecolor=RED, lw=1.3))
    a.text(-1.95, 1.9, r"$\Sigma_A$", color=BLUE, fontsize=11)
    a.text(1.9, 1.25, r"$\Sigma_B$", color=RED, fontsize=11)
    arrow(a, (0, 0), (0, 2.0), color=GREEN, lw=2.2)
    a.text(0.12, 1.0, r"$\Delta_B$", color=GREEN, fontsize=11)
    a.text(0, -2.45, r"$\Delta_B\in\ker\Sigma_A\ \Rightarrow\ \Delta_B^\top\Sigma_A\Delta_B=0$",
           ha="center", fontsize=9.5)
    a.set_title("(b) Disjoint supports: interference removable", fontsize=9.5)

    # --- (c) overlapping supports: unavoidable ---
    a = ax[2]
    a.add_patch(Ellipse((0, 0), 1.1, 4.6, angle=20, fill=False, edgecolor=BLUE, lw=1.3))
    a.add_patch(Ellipse((0, 0), 1.1, 4.6, angle=70, fill=False, edgecolor=RED, lw=1.3))
    a.text(-2.0, 1.7, r"$\Sigma_A$", color=BLUE, fontsize=11)
    a.text(1.5, 1.95, r"$\Sigma_B$", color=RED, fontsize=11)
    tip = np.array([2.0, 0.95])
    arrow(a, (0, 0), tip, color="k", lw=2.0)
    a.text(tip[0] - 0.2, tip[1] + 0.2, r"$\Delta_B$", fontsize=10)
    # shared active direction
    a.plot([-2.4, 2.4], [-2.4 * 0.36, 2.4 * 0.36], color=GREY, lw=1.0, ls="--")
    a.text(-2.3, -1.35, "shared active\ndirection", color=GREY, fontsize=8)
    a.text(0, -2.55, r"$\Delta_B\notin\ker\Sigma_A\ \Rightarrow\ $ floor $>0$",
           ha="center", fontsize=9.5)
    a.set_title("(c) Overlapping supports: floor unavoidable", fontsize=9.5)

    fig.tight_layout()
    fig.savefig("figures/fig_geometry_schematic.pdf"); plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: algorithm hierarchy
# ---------------------------------------------------------------------------
def box(ax, xy, w, h, text, fc, ec="k", fs=8.6, tc="k"):
    ax.add_patch(FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc=fc, ec=ec, lw=1.1, zorder=2))
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fs, zorder=3, color=tc)


def algorithm_hierarchy():
    fig, ax = plt.subplots(figsize=(11.0, 4.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.0); ax.axis("off")

    ax.text(6, 5.7, r"Principle: forgetting $=\Delta^\top\Sigma\Delta$ interference "
            r"$\Rightarrow$ control $\mathit{where}$ updates live",
            ha="center", fontsize=10.5, weight="bold")

    # incoming gradient
    box(ax, (1.5, 3.0), 2.0, 0.9, "new-task\ngradient $g$", "#EAEAEA")
    box(ax, (1.5, 1.2), 2.0, 0.9, r"occupied subspace $U$" + "\n(streaming PCA)", "#EAEAEA", fs=7.8)
    arrow(ax, (2.5, 1.4), (3.4, 2.7), color=GREY, lw=1.2)

    # three tiers
    box(ax, (5.0, 4.6), 3.0, 1.0,
        "NAIVE\nupdate along $g$", "#F4D7D7", ec=RED)
    box(ax, (9.6, 4.6), 4.0, 1.0,
        "interference: A forgotten\n(distortion = floor + shortfall)", "#FBEAEA", ec=RED, fs=8.0)
    arrow(ax, (6.5, 4.6), (7.6, 4.6), color=RED)

    box(ax, (5.0, 3.0), 3.0, 1.0,
        "ORTHOGONAL PROJECTION\n$g_\\perp$ vs all past (OGD/GPM)", "#FBEBD9", ec=ORANGE, fs=7.9)
    box(ax, (9.6, 3.0), 4.0, 1.0,
        "no forgetting, but no transfer;\ncapacity exhausts (floor early)", "#FCF1E4", ec=ORANGE, fs=8.0)
    arrow(ax, (6.5, 3.0), (7.6, 3.0), color=ORANGE)

    box(ax, (5.0, 1.3), 3.0, 1.15,
        "IGFA: signed gate\n" r"$g_\parallel$ if $s>s^\ast$ (share)" "\n"
        r"$g_\perp$ if $s<s^\ast$ (orthogonalize)", "#DCEFE0", ec=GREEN, fs=7.8)
    box(ax, (9.6, 1.3), 4.0, 1.15,
        "retention $+$ transfer;\nstate = subspace only\n(no replay, no Fisher)",
        "#E7F4EA", ec=GREEN, fs=8.0)
    arrow(ax, (6.5, 1.3), (7.6, 1.3), color=GREEN)

    arrow(ax, (3.4, 2.9), (3.5, 4.4), color=GREY, lw=1.0, style="-")
    arrow(ax, (3.5, 2.9), (3.5, 1.5), color=GREY, lw=1.0, style="-")
    arrow(ax, (2.5, 3.0), (3.5, 3.0), color=GREY, lw=1.2)

    ax.text(6.0, 0.2, "offline (batch of task vectors) $\\Rightarrow$ optimal merge "
            r"$=\Sigma$-orthogonalization  $\cdot$  online (stream) $\Rightarrow$ greedy allocation",
            ha="center", fontsize=8.6, style="italic", color="#333333")
    fig.tight_layout()
    fig.savefig("figures/fig_algorithm_hierarchy.pdf"); plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    geometry_schematic()
    algorithm_hierarchy()
    print("wrote fig_geometry_schematic.pdf, fig_algorithm_hierarchy.pdf")
