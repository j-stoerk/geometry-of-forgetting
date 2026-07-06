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


# ============================ FIGURE (thm_geometry) ============================
# Two panels (former (a) removed): the path-averaged-curvature identity and
# the Sigma-weighted optimal merge, both drawn from REAL 2x2 geometry so the
# ellipses, optima, and contours are exact, not schematic.
def rot(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def cov_ellipse(ax, S, center, scale, **kw):
    """Draw the ellipse x' S^{-1} x = scale^2 (a true level set of 1/2 x'Sx dual)."""
    w, V = np.linalg.eigh(S)
    ang = np.degrees(np.arctan2(V[1, -1], V[0, -1]))
    ax.add_patch(Ellipse(center, 2 * scale * np.sqrt(w[-1]), 2 * scale * np.sqrt(w[0]),
                         angle=ang, fill=False, **kw))


def figure2():
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.7))

    # ---- (a) path-averaged curvature: the identity survives drift ----------
    a = ax[0]
    wA, wB = np.array([0.0, 0.0]), np.array([6.4, 1.9])
    d = wB - wA
    # loss landscape of task A: light contours anchored at w_A
    for r, al in [(0.9, 0.5), (1.7, 0.35), (2.5, 0.22)]:
        cov_ellipse(a, rot(18) @ np.diag([2.4, 0.55]) @ rot(18).T, wA, r,
                    ec=BLUE, lw=1.3, alpha=al)
    # the straight chord w_A -> w_B and the drifted geodesic
    arrow(a, wA, wB, BLACK, lw=2.6)
    ts = np.linspace(0, 1, 80)
    curve = wA + np.outer(ts, d) + np.outer(np.sin(ts * np.pi), [-0.25, 1.05])
    a.plot(curve[:, 0], curve[:, 1], "--", color=GREY, lw=1.6)
    txt(a, 4.15, 2.75, "drifting loss geometry", color=GREY)
    # local curvature along the chord: rotating/scaling TRUE ellipses whose
    # line width encodes the (1-t) weight of  Hbar = 2 int (1-t) H(t) dt
    for t in (0.12, 0.36, 0.60, 0.84):
        p = wA + t * d
        St = rot(18 + 55 * t) @ np.diag([1.0 + 0.9 * t, 0.28]) @ rot(18 + 55 * t).T
        col = tuple((1 - t) * np.array([0.17, 0.42, 0.69]) + t * np.array([0.75, 0.22, 0.17]))
        cov_ellipse(a, St, p, 0.55, ec=col, lw=3.2 * (1 - t) + 0.9)
        a.plot(*p, ".", color=col, ms=5)
    # the (1-t) weighting, drawn as a shrinking wedge under the path
    base = wA + np.array([0.0, -1.55])
    a.fill([base[0], base[0] + 6.4, base[0]], [base[1], base[1], base[1] - 0.62],
           color=GREY, alpha=0.3, lw=0)
    txt(a, 2.45, -2.55, r"weight $(1-t)$: early curvature counts more", color=GREY)
    a.plot(*wA, "o", color=BLACK, ms=8, zorder=6); a.plot(*wB, "o", color=BLACK, ms=8, zorder=6)
    txt(a, -0.28, -0.42, r"$w_A$", ha="right"); txt(a, 6.62, 1.9, r"$w_B$", ha="left")
    txt(a, 3.1, 3.35, r"$\Delta L_A=\frac{1}{2}\,\Delta^\top \bar H_A\,\Delta$,"
                      r"  $\bar H_A=2\int_0^1 (1-t)\,\nabla^2 L_A\,dt$")
    a.set_xlim(-2.6, 8.6); a.set_ylim(-3.0, 3.9); clean(a)

    # ---- (b) optimal merge: Sigma-weighted, not Euclidean ------------------
    b = ax[1]
    SA = rot(8) @ np.diag([3.2, 0.30]) @ rot(8).T          # A certain along ~x
    SB = rot(97) @ np.diag([3.0, 0.30]) @ rot(97).T        # B certain along ~y
    wa, wb = np.array([-2.4, -1.1]), np.array([2.4, 1.15])
    wbar = np.linalg.solve(SA + SB, SA @ wa + SB @ wb)     # exact Sigma-weighted merge
    wmid = 0.5 * (wa + wb)                                  # Euclidean average
    for S, c0, col in [(SA, wa, BLUE), (SB, wb, RED)]:
        w_, V_ = np.linalg.eigh(S)
        angS = np.degrees(np.arctan2(V_[1, -1], V_[0, -1]))
        b.add_patch(Ellipse(c0, 2 * np.sqrt(w_[-1]), 2 * np.sqrt(w_[0]), angle=angS,
                            fc=col, alpha=0.10, lw=0))
        cov_ellipse(b, S, c0, 1.0, ec=col, lw=2.0)
        cov_ellipse(b, S, c0, 1.7, ec=col, lw=1.2, alpha=0.45)
    # joint-loss level set THROUGH the Euclidean average, centered at the optimum:
    Stot = SA + SB
    lvl = np.sqrt((wmid - wbar) @ Stot @ (wmid - wbar))
    w_, V_ = np.linalg.eigh(np.linalg.inv(Stot))
    angT = np.degrees(np.arctan2(V_[1, -1], V_[0, -1]))
    b.add_patch(Ellipse(wbar, 2 * lvl * np.sqrt(w_[-1]), 2 * lvl * np.sqrt(w_[0]),
                        angle=angT, fill=False, ec=BLACK, lw=1.6, ls="--"))
    arrow(b, wa, wbar, BLUE, lw=2.2); arrow(b, wb, wbar, RED, lw=2.2)
    b.plot(*wa, "o", color=BLUE, ms=8, zorder=6); b.plot(*wb, "o", color=RED, ms=8, zorder=6)
    b.plot(*wbar, "*", color=GREEN, ms=24, zorder=7, mec=BLACK, mew=0.7)
    b.plot(*wmid, "X", color=GREY, ms=13, zorder=7, mec=BLACK, mew=0.7)
    txt(b, wa[0] + 0.35, wa[1] - 0.62, r"$\mathbf{w}_A^\ast$", color=BLUE, ha="left")
    txt(b, wb[0] + 0.42, wb[1] - 0.72, r"$\mathbf{w}_B^\ast$", color=RED, ha="left")
    b.annotate("Euclidean average", xy=wmid, xytext=(3.0, -1.55),
               color=GREY, fontsize=FS, ha="left",
               arrowprops=dict(arrowstyle="-", color=GREY, lw=1.1))
    b.annotate(r"$\Sigma$-weighted optimum $\bar w^\ast$", xy=wbar,
               xytext=(-2.4, 3.55), color=GREEN, fontsize=FS, ha="center",
               arrowprops=dict(arrowstyle="-", color=GREEN, lw=1.1))
    th = np.radians(250)                                    # bottom edge of the level set
    edge = wbar + lvl * (np.cos(th) * np.sqrt(w_[-1]) * V_[:, -1]
                         + np.sin(th) * np.sqrt(w_[0]) * V_[:, 0])
    b.annotate(r"level set of $L_A{+}L_B$", xy=edge, xytext=(-4.4, -3.1),
               fontsize=FS, ha="center",
               arrowprops=dict(arrowstyle="-", color=BLACK, lw=1.1))
    txt(b, 1.3, -3.6,
        r"$\bar w^\ast=(\Sigma_A{+}\Sigma_B)^{-1}(\Sigma_A w_A^\ast{+}\Sigma_B w_B^\ast)$",
        color=GREEN)
    txt(b, -5.05, 1.45, r"$\Sigma_A$", color=BLUE, ha="left")
    txt(b, 3.55, 2.85, r"$\Sigma_B$", color=RED, ha="left")
    b.set_xlim(-5.7, 7.2); b.set_ylim(-4.2, 4.2); clean(b)

    fig.text(0.03, 0.93, "(a)", fontsize=FS, fontweight="bold")
    fig.text(0.525, 0.93, "(b)", fontsize=FS, fontweight="bold")
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
