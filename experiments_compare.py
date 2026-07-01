# =====================================================================
# Head-to-head method comparison on the exact-A1 Rotated-Digits benchmark
# (frozen random-feature backbone, 5 rotations, domain-incremental).
#
# Produces the teaser figure fig_teaser.pdf (Accuracy vs Forgetting, coloured
# by method family) and comparison_results.json.  All numbers are measured
# multi-seed on the same frozen features; nothing is hand-set.
#
# Method families
#   Baseline   : naive
#   Corrective : EWC, Replay
#   Structural : OGD/GPM, Isotropic-Only (uniform merge, no geometry)
#   Merging    : TSV-Merge (Euclidean sign-elect task arithmetic),
#                TSV+Whitening (Sigma-orthogonal optimal merge, Thm 3)
#   Ours       : igfa (signed gate), Annealed igfa (protect-all warmup -> gate)
#
# Merges are offline (each rotation solved independently, then merged); the
# "forgetting" reported for a merge is the drop from its independent per-task
# accuracy to the merged accuracy.  The post-merge residual D = sum_t 1/2
# ||theta-W_t||^2_{Sigma_t} is logged to validate that the Sigma-metric merge
# (TSV+Whitening) uniquely minimises the theoretical distortion floor.
# =====================================================================
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
try:
    import mpl_style  # shared paper style (DejaVu Sans)
except Exception:
    pass
import matplotlib.pyplot as plt

import experiments_rev as E

FIGDIR = "figures"
N_SEEDS = 5
ANGLES = (0, 12, 24, 36, 48)
D, EPOCHS, LR, R, EWC_LAM, MEM, SSTAR = 300, 600, 0.3, 20, 3.0, 10, 0.65
WARM = 0.5  # annealed: fraction of each task's epochs run as protect-all before relaxing to the gate


def build(seed, rot_imgs):
    """Replicate the exp_benchmark2 frozen-feature setup for a given seed."""
    from sklearn.datasets import load_digits
    y = load_digits().target
    rng = np.random.default_rng(seed)
    C = 10
    Rmat = rng.standard_normal((64, D)) / np.sqrt(64)
    bph = rng.uniform(0, 2 * np.pi, D)

    def feats(imgs_rot):
        return np.maximum(0.0, np.cos(imgs_rot.reshape(len(imgs_rot), -1) @ Rmat + bph))

    rot = {a: feats(rot_imgs[a]) for a in ANGLES}
    allX = np.vstack(list(rot.values()))
    scale = np.sqrt(np.linalg.eigvalsh(allX.T @ allX / (len(y) * len(ANGLES)))[-1])
    rot = {a: rot[a] / scale for a in ANGLES}
    idx = np.arange(len(y)); rng.shuffle(idx)
    cut = int(0.7 * len(idx)); tr, te = idx[:cut], idx[cut:]
    return rot, y, tr, te, C, rng


def onehot(yv, C):
    Y = np.zeros((len(yv), C)); Y[np.arange(len(yv)), yv] = 1.0
    return Y


# ---------------------------------------------------------------- sequential
def run_sequential(method, rot, y, tr, te, C, rng):
    W = np.zeros((C, D)); occ = np.zeros((D, 0)); past = []
    fisher = np.zeros(D); W_star = np.zeros((C, D)); bufX, bufY = [], []

    def acc(Wm, a):
        pred = np.argmax(rot[a][te] @ Wm.T, axis=1)
        return float(np.mean(pred == y[te]))

    def basis_of(X, r):
        w, V = np.linalg.eigh(X.T @ X / len(X)); return V[:, -r:]

    hist = []
    for ti, a in enumerate(ANGLES):
        Xb = rot[a][tr]; Yb = onehot(y[tr], C)
        Bn = basis_of(Xb, R)
        # protect-all projector (OGD) and the igfa gated projector
        P_full = np.eye(D) - occ @ occ.T if occ.shape[1] else None
        if past:
            prot = [Bt for Bt in past if np.linalg.norm(Bt.T @ Bn, "fro") ** 2 / R < SSTAR]
            P_gate = (np.eye(D) - (lambda Q: Q @ Q.T)(np.linalg.qr(np.hstack(prot))[0])) if prot else np.eye(D)
        else:
            P_gate = None
        if method == "replay" and bufX:
            Xt = np.vstack([Xb] + bufX); Yt = np.vstack([Yb] + bufY)
        else:
            Xt, Yt = Xb, Yb
        A = Xt.T @ Xt / len(Xt); Bmat = Yt.T @ Xt / len(Xt)   # normal-equation GD
        for ep in range(EPOCHS):
            G = W @ A - Bmat
            if method == "ewc" and ti > 0:
                G = G + EWC_LAM * fisher[None, :] * (W - W_star)
            if method == "ogd" and P_full is not None:
                G = G @ P_full.T
            elif method == "igfa" and P_gate is not None:
                G = G @ P_gate.T
            elif method == "annealed":                       # protect-all warmup, then gate
                if ep < int(WARM * EPOCHS) and P_full is not None:
                    G = G @ P_full.T
                elif ep >= int(WARM * EPOCHS) and P_gate is not None:
                    G = G @ P_gate.T
            W = W - LR * G
        past.append(Bn)
        occ = np.linalg.qr(np.hstack([occ, Bn]))[0] if occ.shape[1] else np.linalg.qr(Bn)[0]
        if method == "ewc":
            fisher = fisher + np.diag(Xb.T @ Xb / len(Xb)); W_star = W.copy()
        if method == "replay":
            sel = rng.choice(len(tr), min(MEM * C, len(tr)), replace=False)
            bufX.append(Xb[sel]); bufY.append(Yb[sel])
        hist.append([acc(W, ANGLES[j]) for j in range(len(ANGLES))])
    hist = np.array(hist); final = hist[-1]
    diag = np.array([hist[t, t] for t in range(len(ANGLES))])
    return float(final.mean()), float(np.mean(diag[:-1] - final[:-1]))


# ---------------------------------------------------------------- offline merges
def run_merges(rot, y, tr, te, C):
    def acc(Wm, a):
        return float(np.mean(np.argmax(rot[a][te] @ Wm.T, axis=1) == y[te]))

    Wt, Sig = {}, {}
    for a in ANGLES:                                          # independent per-task solutions
        X = rot[a][tr]; Y = onehot(y[tr], C)
        S = X.T @ X / len(X); b = X.T @ Y / len(X)
        Wt[a] = np.linalg.solve(S + 1e-3 * np.eye(D), b).T   # C x d
        Sig[a] = S
    indep = np.mean([acc(Wt[a], a) for a in ANGLES])
    stack = np.stack([Wt[a] for a in ANGLES])                # K x C x d

    def resid(theta):                                        # sum_t 1/2 ||theta-W_t||^2_{Sigma_t}
        return float(sum(0.5 * np.trace((theta - Wt[a]) @ Sig[a] @ (theta - Wt[a]).T) for a in ANGLES))

    # (1) Isotropic-only: uniform average (Euclidean-optimal = no geometry)
    theta_iso = stack.mean(0)
    # (2) TSV-Merge: Euclidean sign-elect task arithmetic (TIES/TSV-style)
    sgn = np.sign(stack.sum(0))
    keep = (np.sign(stack) == sgn[None]) & (sgn[None] != 0)
    num = (stack * keep).sum(0); den = keep.sum(0)
    theta_tsv = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    # (3) TSV+Whitening: Sigma-orthogonal optimal merge (Theorem 3)
    SS = sum(Sig[a] for a in ANGLES); SW = sum(Sig[a] @ Wt[a].T for a in ANGLES)
    theta_sig = np.linalg.solve(SS + 1e-6 * np.eye(D), SW).T
    out = {}
    for name, th in [("isotropic", theta_iso), ("tsv", theta_tsv), ("tsv_white", theta_sig)]:
        merged = np.mean([acc(th, a) for a in ANGLES])
        out[name] = (float(merged), float(indep - merged), resid(th))
    return out, float(indep)


# ---------------------------------------------------------------- driver
def ci95(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return float(x.mean()), 0.0
    from scipy import stats
    return float(x.mean()), float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))


SEQ = ["naive", "ewc", "ogd", "replay", "igfa", "annealed"]
MERGE = ["isotropic", "tsv", "tsv_white"]

if __name__ == "__main__":
    rot_imgs = E.rotate_digits_once(ANGLES)
    accs = {m: [] for m in SEQ + MERGE}
    frgs = {m: [] for m in SEQ + MERGE}
    resid = {m: [] for m in MERGE}
    for s in range(N_SEEDS):
        rot, y, tr, te, C, rng = build(s, rot_imgs)
        for m in SEQ:
            a, f = run_sequential(m, rot, y, tr, te, C, rng)
            accs[m].append(a); frgs[m].append(f)
        mout, _ = run_merges(rot, y, tr, te, C)
        for m in MERGE:
            accs[m].append(mout[m][0]); frgs[m].append(mout[m][1]); resid[m].append(mout[m][2])
        print(f"seed {s} done")

    rows = {}
    print(f"\n{'method':12s}{'acc mean+/-CI':>18s}{'forget mean+/-CI':>20s}{'residual D':>14s}")
    for m in SEQ + MERGE:
        am, ah = ci95(accs[m]); fm, fh = ci95(frgs[m])
        rd = f"{np.mean(resid[m]):.3f}" if m in MERGE else "--"
        rows[m] = {"acc": (am, ah), "forget": (fm, fh), "residualD": (np.mean(resid[m]) if m in MERGE else None)}
        print(f"{m:12s}{am:>10.3f}+/-{ah:.3f}{fm:>13.3f}+/-{fh:.3f}{rd:>14s}")
    json.dump(rows, open("comparison_results.json", "w"), indent=2)

    # ------------------------------------------------ teaser figure
    GREY, ORANGE, BLUE, PURPLE, GREEN = "#7f8c8d", "#e08e0b", "#2b6cb0", "#8e44ad", "#27875a"
    spec = [  # (key, label, family, colour). Ownership is shown by "(ours)" in the NAME;
              # colour is the method CLASS (no separate "Ours" colour).
        ("naive",     "Naive",                 "Baseline",   GREY),
        ("ewc",       "EWC",                   "Corrective", ORANGE),
        ("replay",    "Replay (Bfr)",          "Corrective", ORANGE),
        ("ogd",       "OGD/GPM",               "Structural", BLUE),
        ("igfa",      "igfa (ours)",           "Structural", BLUE),
        ("annealed",  "Annealed igfa (ours)",  "Structural", BLUE),
        ("isotropic", "Isotropic-Only",        "Merging",    PURPLE),
        ("tsv",       "TSV-Merge",             "Merging",    PURPLE),
        ("tsv_white", "Σ-orth. merge (ours)",  "Merging",    PURPLE),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for key, lab, fam, col in spec:
        x = rows[key]["forget"][0]; yv = rows[key]["acc"][0]
        ours = "(ours)" in lab                         # our methods: bold star, class colour
        ax.scatter(x, yv, s=(300 if ours else 95), marker=("*" if ours else "o"), color=col,
                   zorder=4 if ours else 3, edgecolor=("black" if ours else "white"),
                   linewidth=(1.1 if ours else 0.8))
        off = (11, 7) if ours else (8, 5)              # label to the TOP-RIGHT of the point
        ax.annotate(lab, (x, yv), xytext=off, textcoords="offset points", ha="left", va="bottom",
                    fontsize=9, color="#111" if ours else "#333", fontweight="bold" if ours else "normal")
    ax.axvline(0, color="#444", lw=1.0, zorder=1)
    ax.set_xlabel("Forgetting rate  (lower is better $\\rightarrow$)")
    ax.set_ylabel("Average accuracy  ($\\uparrow$ better)")
    ax.set_title("Accuracy vs. forgetting on Rotated-Digits (exact A1)", fontsize=12, fontweight="bold")
    ax.set_xlim(0.245, -0.075)          # inverted: high forgetting left, room for labels right
    ax.set_ylim(0.665, 0.915)
    # legend: method class (colour) + a white star for our methods
    seen = {}
    for _, _, fam, col in spec:
        seen[fam] = col
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=c, label=f, ms=8) for f, c in seen.items()]
    handles.append(plt.Line2D([0], [0], marker="*", ls="", markerfacecolor="white",
                              markeredgecolor="#111", markeredgewidth=1.1, label="ours", ms=15))
    ax.legend(handles=handles, loc="upper left", ncol=3, fontsize=8.5)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_teaser.pdf"); plt.close(fig)
    print("\nwrote fig_teaser.pdf and comparison_results.json")
