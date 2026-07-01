# =====================================================================
# Clean vector redraw of the theorem schematics (Figures 2 and 3) with a
# single uniform font size throughout, replacing the raster illustrations.
#   thm_geometry.pdf : (a) forgetting=interference energy, (b) path-averaged
#                      curvature, (c) optimal merge = Sigma-orthogonalization
#   thm_floor.pdf    : (a) distortion floor geometry, (b) task-confusability band
# Run:  python fig_theorems.py
# =====================================================================
import numpy as np
import matplotlib
matplotlib.use("Agg")
try:
    import mpl_style  # shared publication style (DejaVu Sans) -> same font as every other figure
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch

FS = 15                      # one uniform font size for ALL text
BLUE, RED, GREEN, GREY, BLACK = "#2b6cb0", "#c0392b", "#27875a", "#7f8c8d", "#1a1a1a"
plt.rcParams.update({"font.size": FS})


def arrow(ax, p, q, color=BLACK, lw=2.4, style="-|>", ms=14, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms,
                                 lw=lw, color=color, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=4))


def txt(ax, x, y, s, color=BLACK, ha="center", va="center"):
    ax.text(x, y, s, color=color, ha=ha, va=va, fontsize=FS, zorder=6)


def clean(ax):
    ax.set_aspect("equal"); ax.axis("off")


# ============================ FIGURE 2 ============================
def figure2():
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.0))

    # (a) interference energy: contours + Delta split -------------------
    a = ax[0]
    for r in (0.8, 1.4, 2.0, 2.6):
        a.add_patch(Ellipse((0, 0), 1.8 * r, 2.4 * r, fill=False, ec=BLUE, lw=1.3))
    a.plot(0, 0, "o", color=BLACK, ms=7, zorder=5)
    txt(a, 0.0, -0.62, r"$\mathbf{w}_A^\ast$", ha="center")   # inside innermost contour, below the dot
    tip = (1.7, 2.1)
    arrow(a, (0, 0), tip, BLACK)                       # Delta
    arrow(a, (0, 0), (1.7, 0), BLUE)                   # Delta_par (range, forgetting)
    arrow(a, (1.7, 0), tip, GREEN)                     # Delta_perp (ker, free)
    txt(a, 1.9, 2.5, r"$\Delta=\mathbf{w}_B-\mathbf{w}_A^\ast$", ha="left")
    txt(a, 2.15, 1.6, r"$\Delta_\perp\in\ker\Sigma_A$", color=GREEN, ha="left")
    txt(a, 2.45, 0.82, "(interference-free)", color=GREEN, ha="left")
    txt(a, 0.0, -3.7, r"$\Delta_\parallel\in\mathrm{range}\,\Sigma_A$ (causes forgetting)", color=BLUE)
    a.set_xlim(-4.9, 5.8); a.set_ylim(-4.3, 3.4); clean(a)

    # (b) path-averaged curvature ---------------------------------------
    b = ax[1]
    wA, wB = np.array([0.0, 0.0]), np.array([6.0, 1.8])
    arrow(b, wA, wB, BLACK, lw=2.2)
    d = wB - wA; ang = np.degrees(np.arctan2(d[1], d[0]))
    for t, c, lw in [(0.25, GREY, 1.4), (0.5, BLACK, 2.2), (0.75, GREY, 1.4)]:
        p = wA + t * d
        b.add_patch(Ellipse(p, 0.5, 2.0, angle=ang, fill=False, ec=c, lw=lw))
    # drifted geodesic
    ts = np.linspace(0, 1, 60); curve = wA + np.outer(ts, d) + np.outer(np.sin(ts * np.pi), [-0.2, 0.9])
    b.plot(curve[:, 0], curve[:, 1], "--", color=GREY, lw=1.6)
    txt(b, -0.2, -0.35, r"$w_A$", ha="right"); txt(b, 6.2, 1.8, r"$w_B$", ha="left")
    txt(b, 3.0, -0.9, r"path-averaged curvature $\bar H_A$")
    b.set_xlim(-1.2, 7.4); b.set_ylim(-2.0, 3.2); clean(b)

    # (c) optimal merge = Sigma-orthogonalization -----------------------
    c = ax[2]
    c.plot(0, 0, "o", color=BLACK, ms=7, zorder=5)
    txt(c, -0.25, 0.18, r"$w_0$", ha="right")
    vecs = [((1.7, 2.0), BLUE), ((2.2, -0.7), RED), ((-1.9, -1.1), GREY)]
    for (v, col) in vecs:
        arrow(c, (0, 0), v, col, lw=2.4)
        ang = np.degrees(np.arctan2(v[1], v[0]))
        c.add_patch(Ellipse(v, 0.45, 1.5, angle=ang, fill=False, ec=col, lw=1.5))
    txt(c, 0.0, -3.0, r"optimal merge: $\Delta_{t'}\in\ker\Sigma_t$")
    c.set_xlim(-3.6, 3.8); c.set_ylim(-3.6, 3.2); clean(c)

    fig.text(0.025, 0.92, "(a)", fontsize=FS, fontweight="bold")
    fig.text(0.355, 0.92, "(b)", fontsize=FS, fontweight="bold")
    fig.text(0.685, 0.92, "(c)", fontsize=FS, fontweight="bold")
    fig.tight_layout(); fig.savefig("figures/thm_geometry.pdf", bbox_inches="tight"); plt.close(fig)


# ============================ FIGURE 3 ============================
def figure3():
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # (a) distortion-floor geometry -------------------------------------
    a = ax[0]
    cA, cB = np.array([-0.9, -0.4]), np.array([0.9, 0.5])
    a.add_patch(Ellipse(cA, 1.7, 3.4, angle=35, fill=False, ec=BLUE, lw=2.0))
    a.add_patch(Ellipse(cB, 1.7, 3.4, angle=35, fill=False, ec=RED, lw=2.0))
    a.plot(*cA, "o", color=BLUE, ms=7, zorder=5); a.plot(*cB, "o", color=RED, ms=7, zorder=5)
    txt(a, cA[0] - 0.15, cA[1] + 0.05, r"$\mathbf{w}_A^\ast$", color=BLUE, ha="right")
    txt(a, cB[0] + 0.18, cB[1], r"$\mathbf{w}_B^\ast$", color=RED, ha="left")
    arrow(a, cA, cB, BLACK, lw=2.2); txt(a, 0.15, -0.35, r"$\delta$")
    # shared active direction (dashed line through both)
    dvec = np.array([1.0, 0.55]); dvec /= np.linalg.norm(dvec)
    p0, p1 = -2.6 * dvec, 3.2 * dvec
    a.plot([p0[0], p1[0]], [p0[1], p1[1]], "--", color=BLACK, lw=2.2)
    txt(a, 0.2, 2.75, r"shared active direction $(\mathbf{u})$", color=GREY)
    txt(a, -2.25, 1.0, r"$\Sigma_A$", color=BLUE); txt(a, 2.55, 1.75, r"$\Sigma_B$", color=RED)
    # floor brace: brackets the delta gap, kept below the dashed line (y<1.3 at x=2.5)
    a.annotate("", xy=(2.5, 0.6), xytext=(2.5, -0.5),
               arrowprops=dict(arrowstyle="-", color=BLACK, lw=1.4,
                               connectionstyle="bar,fraction=0.15"))
    txt(a, 3.05, 0.05, r"$D\geq\frac{1}{4}\sigma_u\delta^2$", ha="left")
    txt(a, 1.3, -2.4, "unavoidable distortion floor", color=GREY)
    a.set_xlim(-3.0, 5.0); a.set_ylim(-3.0, 3.0); clean(a)

    # (b) task-confusability band ---------------------------------------
    b = ax[1]
    x = np.linspace(0, np.log(2), 200)
    hb = x.copy()                                   # use H as proxy axis
    upper = 0.25 * hb / np.log(2)                   # ~ (1/4) h_b
    lower = 0.125 * (hb / np.log(2)) ** 2 * np.log(2) / np.log(2)
    b.fill_between(hb, lower, upper, color=GREEN, alpha=0.35, lw=0)
    b.plot(hb, upper, color=GREEN, lw=2.0); b.plot(hb, lower, color=GREEN, lw=2.0)
    b.annotate("", xy=(np.log(2) * 1.02, 0), xytext=(0, 0),
               arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.6))
    b.annotate("", xy=(0, 0.28), xytext=(0, 0),
               arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.6))
    txt(b, np.log(2) * 0.5, -0.04, r"$H(T\mid\phi)$")
    txt(b, -0.03, 0.275, r"$D^\star$", ha="right")
    txt(b, 0.1, 0.26, "task confusability", color=GREY, ha="left")
    txt(b, 0.1, 0.227, r"$\mathrm{Var}(T\mid\phi)$", color=GREY, ha="left")
    b.set_xlim(-0.06, 0.80); b.set_ylim(-0.07, 0.30); clean(b)

    fig.text(0.04, 0.93, "(a)", fontsize=FS, fontweight="bold")
    fig.text(0.53, 0.93, "(b)", fontsize=FS, fontweight="bold")
    fig.tight_layout(); fig.savefig("figures/thm_floor.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    figure2(); figure3()
    print("wrote figures/thm_geometry.pdf and figures/thm_floor.pdf")
