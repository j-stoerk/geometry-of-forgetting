# =====================================================================
#  Q4 retro-validation: the gate's realized advantage over unconditional
#  projection, across ALL six real streams of the paper, against one
#  pre-deployment statistic -- the stream's SHARE DENSITY (fraction of
#  task pairs whose measured similarity crosses that stream's gate
#  threshold).  No new training: Rotated-Digits overlaps are recomputed
#  exactly from the committed pipeline; the other five streams' pairwise
#  values come from the logged runs (provenance per block below).
#
#  Output: figures/fig_density.pdf + printed table.
# =====================================================================
import numpy as np
import matplotlib
matplotlib.use("Agg")
try:
    import mpl_style
except Exception:
    pass
import matplotlib.pyplot as plt

# ---- Rotated-Digits: recompute pairwise subspace overlaps exactly ---------
import experiments_rev as E
import experiments_compare as C

def rotdigits_density():
    rot_imgs = E.rotate_digits_once(C.ANGLES)
    dens = []
    for seed in range(C.N_SEEDS):
        rot, y, tr, te, _, _ = C.build(seed, rot_imgs)
        bases = []
        for a in C.ANGLES:
            X = rot[a][tr]
            w, V = np.linalg.eigh(X.T @ X / len(X))
            bases.append(V[:, -C.R:])
        ov = [np.linalg.norm(bases[i].T @ bases[j], "fro") ** 2 / C.R
              for i in range(5) for j in range(i + 1, 5)]
        dens.append(np.mean([o >= C.SSTAR for o in ov]))
    return float(np.mean(dens))

# ---- the six streams -------------------------------------------------------
# share density = fraction of (new task, old task) pairs the gate SHARED,
# read from each run's own gate log; realized gain = igfa - OGD avg accuracy
# (paired, five seeds) from the corresponding results table.
streams = [
    # name, share_density, realized_gain, significant, source
    ("Rotated-Digits", rotdigits_density(), 0.771 - 0.684, True,
     "recomputed here; gain: Table 2 (p=6e-5)"),
    ("ImageNet-R", 12 / 45, 0.788 - 0.781, True,
     "kaggle_vit_multiseed Y2 log: shared 12/45 pairs; gain: Table 3 (p=0.001)"),
    ("C100-Superclass", 1 / 10, 0.000, False,
     "Y4 log: shared 1/10 pairs; gain: igfa=OGD 0.786 (p=0.37)"),
    ("Split-CIFAR-100", 0 / 45, 0.000, False,
     "Y1 log: overlaps 0.31-0.50, all protected; igfa==OGD every seed"),
    ("CUB-200", 0 / 45, 0.000, False,
     "Y3 log: overlaps 0.34-0.54, all protected; igfa==OGD every seed"),
    ("AG-News (LoRA)", 0 / 6, 0.000, False,
     "probe_gate_progress.log: probe cos -0.20..+0.16, all below 0.25; tie n.s."),
]

if __name__ == "__main__":
    GREEN, GREY = "#27875a", "#7f8c8d"
    xs = [s[1] for s in streams]; ys = [s[2] for s in streams]
    from scipy import stats
    rho = stats.spearmanr(xs, ys).statistic
    print(f"{'stream':18s} {'share density':>14s} {'gate gain (acc)':>16s}  significant")
    for n, x, g, sig, src in streams:
        print(f"{n:18s} {x:14.3f} {g:+16.3f}  {'yes' if sig else 'tie'}   [{src}]")
    print(f"\nSpearman(share density, realized gain) = {rho:.2f}")
    print("zero-density streams tie EXACTLY (the gate provably reduces to OGD there);")
    print("both significant wins are the two highest-density streams.")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    offsets = {"Rotated-Digits": (-8, 6, "right"), "ImageNet-R": (10, 4, "left"),
               "C100-Superclass": (8, -4, "left"), "Split-CIFAR-100": (8, 16, "left"),
               "CUB-200": (8, 5, "left"), "AG-News (LoRA)": (8, -13, "left")}
    for n, x, g, sig, _ in streams:
        ax.scatter(x, g, s=(220 if sig else 90), marker="*" if sig else "o",
                   color=GREEN if sig else GREY,
                   edgecolor="black" if sig else "white", linewidth=1.0, zorder=3)
        dx, dy, ha = offsets[n]
        ax.annotate(n, (x, g), xytext=(dx, dy), textcoords="offset points",
                    fontsize=9.5, ha=ha,
                    color="#111" if sig else "#444",
                    fontweight="bold" if sig else "normal")
    ax.axhline(0, color="#bbb", lw=1.0, zorder=1)
    ax.set_xlabel("share density  (fraction of task pairs above the gate threshold)")
    ax.set_ylabel("realized gate gain over OGD  (avg. accuracy)")
    ax.set_xlim(-0.05, 0.68); ax.set_ylim(-0.015, 0.102)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig("figures/fig_density.pdf", bbox_inches="tight")
    print("wrote figures/fig_density.pdf")
