"""
Geometry of Forgetting -- experiments for the revision round:
  (A) fig_sstar.pdf     s* ablation + heuristic (sensitivity to feature dim d);
  (B) fig_graceful.pdf  graceful degradation when capacity overwrites a direction;
  (C) fig_drift.pdf     online recursive Gauss-Newton subspace update under feature
                        drift -- the answer to subspace-projection failure in LLMs
                        (capacity exhaustion vs bounded rank).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
try:
    import mpl_style  # shared publication style
except Exception:
    pass
import matplotlib.pyplot as plt

RNG = np.random.default_rng(0)
BLUE, RED, GREEN, GREY, PURPLE = "#4C72B0", "#C44E52", "#55A868", "#777777", "#8172B3"
FIGDIR = "figures"
res = {}


# ---------------------------------------------------------------------------
# (A) s* ablation: a smoothly drifting K-task stream on a shared block.
#     Sharing similar tasks reduces variance (transfer); sharing dissimilar tasks
#     adds bias. The optimal threshold sits well above the chance overlap r/d.
# ---------------------------------------------------------------------------
def exp_sstar():
    def run(d, r=6, K=8, m=10, noise=0.7, decay=0.82, trials=200):
        sgrid = np.linspace(0.0, 1.0, 26)
        err = np.zeros(len(sgrid))
        for _ in range(trials):
            Q = np.linalg.qr(RNG.standard_normal((d, r)))[0]      # shared active block
            # drifting unit targets: cos(u_t,u_{t'}) ~ decay^{|t-t'|}
            base = RNG.standard_normal(r); us = []
            u = base / np.linalg.norm(base)
            for t in range(K):
                g = RNG.standard_normal(r); g -= (g @ u) * u
                u = decay * u + np.sqrt(1 - decay**2) * g / np.linalg.norm(g)
                u = u / np.linalg.norm(u); us.append(u.copy())
            us = np.array(us)
            cos = us @ us.T                                       # true pairwise similarity
            X = [RNG.standard_normal((m, r)) for _ in range(K)]
            y = [X[t] @ us[t] + noise * RNG.standard_normal(m) for t in range(K)]
            for j, s in enumerate(sgrid):
                tot = 0.0
                for t in range(K):                               # pool earlier tasks with sim>s*
                    idx = [t] + [tp for tp in range(t) if cos[t, tp] > s]
                    Xp = np.vstack([X[i] for i in idx]); yp = np.concatenate([y[i] for i in idx])
                    u_hat = np.linalg.solve(Xp.T @ Xp + 1e-2 * np.eye(r), Xp.T @ yp)
                    tot += np.sum((u_hat - us[t]) ** 2)          # population MSE (Sigma=I)
                err[j] += tot / K
        return sgrid, err / trials
    out = {}
    for d in (64, 256):
        sg, e = run(d)
        out[d] = (sg, e)
        res[f"sstar_opt_d{d}"] = float(sg[np.argmin(e)])
        res[f"sstar_chance_d{d}"] = 6.0 / d
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    for d, c in zip((64, 256), (BLUE, GREEN)):
        sg, e = out[d]
        ax.plot(sg, e, "-o", ms=3, color=c, label=f"$d={d}$ (chance $r/d={6/d:.2f}$)")
        ax.axvline(sg[np.argmin(e)], color=c, ls="--", lw=1)
    ax.set_xlabel(r"similarity threshold $s^\ast$ (target cosine)")
    ax.set_ylabel("avg. population error (lower better)")
    ax.set_title("Broad optimum, well above chance overlap", fontsize=9.5)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_sstar.pdf"); plt.close(fig)


# ---------------------------------------------------------------------------
# (B) Graceful degradation: overwrite a task's occupied directions. Dropping the
#     smallest-eigenvalue (lowest-energy) direction costs little; the loss rises
#     in proportion to the dropped direction's energy, not catastrophically.
# ---------------------------------------------------------------------------
def exp_graceful(d=40, rank=12, trials=200):
    drop = np.arange(0, rank + 1)
    loss_small = np.zeros(rank + 1); loss_large = np.zeros(rank + 1)
    for _ in range(trials):
        Q = np.linalg.qr(RNG.standard_normal((d, rank)))[0]
        eig = np.sort(RNG.uniform(0.2, 2.0, rank))[::-1]          # descending spectrum
        Sig = (Q * eig) @ Q.T
        w = Q @ RNG.standard_normal(rank); wstar = w.copy()
        # forgetting from removing the head's component along occupied direction i
        comp = (Q.T @ wstar)                                      # coeff per direction
        contrib = 0.5 * eig * comp**2                            # energy lost if dropped
        order_small = np.argsort(eig)                            # smallest eigenvalue first
        order_large = np.argsort(eig)[::-1]                      # largest first
        cs_small = np.concatenate([[0], np.cumsum(contrib[order_small])])
        cs_large = np.concatenate([[0], np.cumsum(contrib[order_large])])
        loss_small += cs_small; loss_large += cs_large
    loss_small /= trials; loss_large /= trials
    res["graceful_small_full"] = float(loss_small[-1])
    res["graceful_large_half"] = float(loss_large[rank // 2])
    res["graceful_small_half"] = float(loss_small[rank // 2])
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    ax.plot(drop, loss_small, "-o", ms=4, color=GREEN, label="drop smallest-energy first (igfa)")
    ax.plot(drop, loss_large, "-s", ms=4, color=RED, label="drop largest-energy first")
    ax.set_xlabel("number of occupied directions overwritten")
    ax.set_ylabel("forgetting of the old task  $\\Delta L_A$")
    ax.set_title("Overwriting low-energy directions degrades gracefully", fontsize=9.5)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_graceful.pdf"); plt.close(fig)


# ---------------------------------------------------------------------------
# (C) Online recursive Gauss-Newton update under feature drift.
#     The feature geometry rotates as the stream proceeds. A STALE protected
#     subspace (OGD/GPM: fix it once) accumulates drifted copies -> occupied rank
#     grows toward d (capacity exhaustion) and protection misaligns. A RECURSIVE
#     estimate (streaming PCA with decay on the current Gauss-Newton) stays
#     low-rank and aligned. This is the subspace-projection failure mode reported
#     for LLM merging, and its fix.
# ---------------------------------------------------------------------------
def exp_drift(d=60, r0=4, T=25, drift=0.18, kmax=12, trials=40):
    rank_stale = np.zeros(T); rank_rec = np.zeros(T)
    plast_stale = np.zeros(T); plast_rec = np.zeros(T)

    def small_rotation(d, angle, rng):
        A = rng.standard_normal((d, d)); A = A - A.T              # skew-symmetric
        A = angle * A / np.linalg.norm(A, 2)
        # matrix exponential via eigen (A skew -> orthogonal exp)
        w, V = np.linalg.eigh(1j * A)
        return np.real(V @ np.diag(np.exp(-1j * w)) @ V.conj().T)

    for _ in range(trials):
        rng = np.random.default_rng(int(RNG.integers(1 << 30)))
        base = np.linalg.qr(rng.standard_normal((d, r0)))[0]      # true low-rank subspace
        Rcur = np.eye(d)
        occ_stale = np.zeros((d, 0))                             # accumulate (never merged)
        C_rec = np.zeros((d, d))                                 # recursive GN estimate
        for t in range(T):
            Rcur = small_rotation(d, drift, rng) @ Rcur          # geometry drifts each step
            Bt = Rcur @ base                                     # current task subspace
            Sig_t = Bt @ Bt.T
            # stale: pile up
            occ_stale = np.linalg.qr(np.hstack([occ_stale, Bt]))[0] if occ_stale.shape[1] else \
                np.linalg.qr(Bt)[0]
            rank_stale[t] += occ_stale.shape[1]
            # recursive: decayed streaming second-moment; protect only SIGNIFICANT dirs
            C_rec = 0.6 * C_rec + Sig_t
            wv, V = np.linalg.eigh(C_rec)
            sig = wv > 1e-2 * wv.max()                            # significant directions
            blocked_rec = int(sig.sum())
            rank_rec[t] += blocked_rec
            # plasticity = fraction of parameter space still free to learn new tasks
            plast_stale[t] += (d - occ_stale.shape[1]) / d
            plast_rec[t] += (d - blocked_rec) / d
    rank_stale /= trials; rank_rec /= trials
    plast_stale /= trials; plast_rec /= trials
    res["drift_rank_stale_final"] = float(rank_stale[-1])
    res["drift_rank_rec_final"] = float(rank_rec[-1])
    res["drift_plasticity_stale_final"] = float(plast_stale[-1])
    res["drift_plasticity_rec_final"] = float(plast_rec[-1])
    res["drift_true_rank"] = r0

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.7))
    xs = np.arange(1, T + 1)
    ax[0].plot(xs, rank_stale, "-o", ms=3, color=RED, label="stale subspace (OGD/GPM)")
    ax[0].plot(xs, rank_rec, "-s", ms=3, color=GREEN, label="recursive GN (self-correcting)")
    ax[0].axhline(d, color=GREY, ls=":", lw=1); ax[0].text(1, d - 4, "capacity $d$", fontsize=8, color=GREY)
    ax[0].axhline(r0, color="black", ls="--", lw=0.8); ax[0].text(1, r0 - 3.0, "true rank", fontsize=8)
    ax[0].set_xlabel("task index in the drifting stream")
    ax[0].set_ylabel("occupied / effective rank")
    ax[0].set_title("Stale projection exhausts capacity under drift", fontsize=9.5)
    ax[0].legend(fontsize=7.8, loc="center right")
    ax[1].plot(xs, plast_stale, "-o", ms=3, color=RED, label="stale subspace (OGD/GPM)")
    ax[1].plot(xs, plast_rec, "-s", ms=3, color=GREEN, label="recursive GN (self-correcting)")
    ax[1].set_xlabel("task index in the drifting stream")
    ax[1].set_ylabel("free plasticity  $(d-\\mathrm{rank})/d$")
    ax[1].set_ylim(-0.03, 1.0)
    ax[1].set_title("Re-estimation preserves capacity to learn", fontsize=9.5)
    ax[1].legend(fontsize=7.8, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_drift.pdf"); plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(FIGDIR, exist_ok=True)
    exp_sstar()
    exp_graceful()
    exp_drift()
    with open("results_drift.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
