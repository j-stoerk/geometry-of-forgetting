"""
Roadmap prototypes: test two new ideas locally so the "is it better?" question
gets a real answer.
  (A) fig_feasibility.pdf  a pre-flight feasibility diagnostic: a cheap linear
                           task-probe estimates I(T;phi(X)) and predicts the
                           distortion floor BEFORE any continual training (#3).
  (B) fig_cheapgn.pdf      cheap online Gauss-Newton subspace via Frequent
                           Directions: O(l d) memory recovers the top subspace as
                           well as full O(d^2), and better than Oja per memory (#1).
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
from sklearn.linear_model import LogisticRegression

RNG = np.random.default_rng(0)
BLUE, RED, GREEN, GREY, PURPLE = "#4C72B0", "#C44E52", "#55A868", "#777777", "#8172B3"
FIGDIR = "figures"
res = {}


def hb(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


# ===========================================================================
# (A) Pre-flight feasibility diagnostic.
#     Theory (floor = ceiling): the class-IL achievable accuracy is bounded by
#     I(T;phi). A cheap PRE-FLIGHT estimate: fit a linear task-classifier on the
#     features; its test accuracy -> I_est, which predicts the floor with NO
#     continual-learning run. We validate (i) I_est ~ I_true, (ii) floor decreases
#     monotonically in I_est, so the probe forecasts solvability.
# ===========================================================================
def exp_feasibility(d=12, N=6000):
    seps = np.linspace(0.1, 5.0, 16)
    floor, I_true, I_est = [], [], []
    for sep in seps:
        rng = np.random.default_rng(int(RNG.integers(1 << 30)))
        u = rng.standard_normal(d); u /= np.linalg.norm(u)
        muA, muB = 0.5 * sep * u, -0.5 * sep * u
        Q = np.linalg.qr(rng.standard_normal((d, d)))[0]
        Sig = (Q * rng.uniform(0.5, 1.5, d)) @ Q.T
        wA = rng.standard_normal(d); wB = rng.standard_normal(d); delta = wA - wB
        nA = N // 2
        XA = rng.multivariate_normal(muA, Sig, nA); XB = rng.multivariate_normal(muB, Sig, N - nA)
        X = np.vstack([XA, XB]); T = np.concatenate([np.ones(nA), np.zeros(N - nA)])
        # exact posterior + true floor and MI (Gaussian, equal cov)
        Si = np.linalg.inv(Sig)
        la = -0.5 * np.einsum("ij,jk,ik->i", X - muA, Si, X - muA)
        lb = -0.5 * np.einsum("ij,jk,ik->i", X - muB, Si, X - muB)
        eta = 1 / (1 + np.exp(lb - la))
        floor.append(0.5 * np.mean(eta * (1 - eta) * (X @ delta) ** 2))
        I_true.append(np.log(2) - np.mean(hb(eta)))
        # PRE-FLIGHT diagnostic: a linear task-probe (no CL training)
        tr = rng.permutation(N)[: N // 2]; te = np.setdiff1d(np.arange(N), tr)
        clf = LogisticRegression(max_iter=200).fit(X[tr], T[tr])
        acc = clf.score(X[te], T[te])
        I_est.append(np.log(2) - hb(1 - acc))
    floor, I_true, I_est = map(np.array, (floor, I_true, I_est))
    res["feasibility_I_corr"] = float(np.corrcoef(I_true, I_est)[0, 1])
    res["feasibility_floor_vs_Iest_spearman"] = float(
        np.corrcoef(np.argsort(np.argsort(I_est)), np.argsort(np.argsort(-floor)))[0, 1])

    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.7))
    lim = max(I_true.max(), I_est.max()) * 1.05
    ax[0].plot([0, lim], [0, lim], "--", color=GREY, lw=1)
    ax[0].scatter(I_true, I_est, s=28, color=BLUE)
    ax[0].set_xlabel(r"true $I(T;\phi)$ (nats)"); ax[0].set_ylabel(r"probe estimate $\hat I$")
    ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim)
    ax[1].scatter(I_est, floor, s=28, color=GREEN)
    ax[1].set_xlabel(r"pre-flight $\hat I$ (no CL run)"); ax[1].set_ylabel("distortion floor")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_feasibility.pdf"); plt.close(fig)


# ===========================================================================
# (B) Cheap online Gauss-Newton subspace via Frequent Directions.
#     Compare top-r subspace recovery of a low-rank-plus-noise covariance:
#       full      : accumulate d x d covariance, eig  -- O(d^2) mem, O(d^3) eig
#       Oja       : streaming rank-r PCA              -- O(d r) mem
#       FD        : frequent-directions sketch l x d  -- O(d l) mem, bounded error
#     Metric: captured energy of the true top-r subspace (1 = perfect).
# ===========================================================================
def frequent_directions(stream, ell):
    d = stream.shape[1]; B = np.zeros((ell, d))      # l x d sketch (Liberty 2013)
    for x in stream:
        zero = np.where(~B.any(axis=1))[0]
        if len(zero) == 0:                           # sketch full -> densify
            s, Vt = np.linalg.svd(B, full_matrices=False)[1:]
            delta = s[ell // 2] ** 2
            B = np.diag(np.sqrt(np.maximum(s ** 2 - delta, 0.0))) @ Vt
            zero = np.where(~B.any(axis=1))[0]
        B[zero[0]] = x
    return B


def oja_pca(stream, r, lr=0.05):
    d = stream.shape[1]; W = np.linalg.qr(np.random.default_rng(0).standard_normal((d, r)))[0]
    for x in stream:
        W = W + lr * np.outer(x, x @ W)
        W = np.linalg.qr(W)[0]
    return W


def exp_cheap_gn(d=128, r=6, n=4000, trials=8):
    def energy(U, true_U):                       # fraction of true top-r energy captured
        return float(np.linalg.norm(true_U.T @ U, "fro") ** 2 / r)
    ells = [r, 2 * r, 3 * r, 4 * r, 6 * r]
    full_e = []; fd_e = {e: [] for e in ells}; oja_e = []
    for _ in range(trials):
        rng = np.random.default_rng(int(RNG.integers(1 << 30)))
        Vtrue = np.linalg.qr(rng.standard_normal((d, r)))[0]
        scale = np.linspace(3, 1, r)
        Z = rng.standard_normal((n, r)) * scale
        X = Z @ Vtrue.T + 0.15 * rng.standard_normal((n, d))      # low-rank + noise
        C = X.T @ X
        full_U = np.linalg.eigh(C)[1][:, -r:]; full_e.append(energy(full_U, Vtrue))
        for e in ells:
            B = frequent_directions(X, e)
            fd_U = np.linalg.svd(B, full_matrices=False)[2][:r].T
            fd_e[e].append(energy(fd_U, Vtrue))
        oja_e.append(energy(oja_pca(X, r), Vtrue))
    res["cheapgn_full"] = float(np.mean(full_e))
    res["cheapgn_oja"] = float(np.mean(oja_e))
    res["cheapgn_fd"] = {str(e): float(np.mean(fd_e[e])) for e in ells}

    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    ax.axhline(np.mean(full_e), color=GREY, ls="--", lw=1.4, label=f"full $O(d^2)$ ({np.mean(full_e):.3f})")
    ax.axhline(np.mean(oja_e), color=RED, ls=":", lw=1.4, label=f"Oja $O(dr)$ ({np.mean(oja_e):.3f})")
    mem = [e * d for e in ells]; fde = [np.mean(fd_e[e]) for e in ells]
    ax.plot([e for e in ells], fde, "-o", color=GREEN, ms=5, label="Frequent Directions $O(d\\ell)$")
    ax.set_xlabel(r"FD sketch rows $\ell$ (memory $\propto \ell d$)")
    ax.set_ylabel(f"captured top-{r} energy")
    ax.legend(fontsize=8, loc="lower right"); ax.set_ylim(min(np.mean(oja_e) - 0.05, 0.8), 1.01)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_cheapgn.pdf"); plt.close(fig)


# ===========================================================================
# (C) Soft gating smears the capacity knee -- and r_eff = sum_j (1-keep_j) is the
#     universal capacity coordinate. Hard allocation (keep=0) knees at T*r=d; the
#     functional controller (keep_j = 1/(1+beta*lambda_j)) protects each direction
#     partially, so capacity is consumed at rate sum(1-keep) and the knee moves to
#     r_eff ~ d. Plotting distortion vs r_eff collapses hard and soft onto one curve.
# ===========================================================================
def exp_soft_knee(d=40, r=2, T=22, trials=20, lr=0.4, steps=1500):
    def excess(w, S, ws): dv = w - ws; return 0.5 * dv @ S @ dv

    def run(beta):  # beta=None -> hard
        resid = np.zeros(T); cap = np.zeros(T)
        for _ in range(trials):
            rng = np.random.default_rng(int(RNG.integers(1 << 30)))
            w = np.zeros(d); Sig, wst = [], []
            Cacc = np.zeros((d, d))
            for t in range(T):
                Q = np.linalg.qr(rng.standard_normal((d, r)))[0]
                S = Q @ Q.T; ws = Q @ rng.standard_normal(r); ws /= np.linalg.norm(ws)
                # protection from accumulated geometry
                ev, U = np.linalg.eigh(Cacc); sig = ev > 1e-6 * max(ev.max(), 1e-9)
                Uk = U[:, sig]; lam = ev[sig]
                if Uk.shape[1]:
                    if beta is None:
                        keep = np.zeros(Uk.shape[1])
                    else:
                        keep = 1.0 / (1.0 + beta * lam / lam.max())
                    P = np.eye(d) - Uk @ np.diag(1 - keep) @ Uk.T
                    cap[t] += float(np.sum(1 - keep))             # r_eff (= rank when hard)
                else:
                    P = None
                for _ in range(steps):                            # projected GD on task t
                    g = S @ (w - ws)
                    if P is not None: g = P @ g
                    w = w - lr * g
                Sig.append(S); wst.append(ws); Cacc = GAMMA_ACC * Cacc + S
                resid[t] += np.mean([excess(w, Sig[j], wst[j]) for j in range(t + 1)])
        return resid / trials, cap / trials

    global GAMMA_ACC; GAMMA_ACC = 1.0
    curves = {"hard (keep=0)": run(None), r"soft $\beta=3$": run(3.0), r"soft $\beta=1$": run(1.0)}

    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.7))
    xs = np.arange(1, T + 1)
    for (lab, (rd, cp)), c in zip(curves.items(), (GREY, GREEN, BLUE)):
        ax[0].plot(xs, rd, "-o", ms=3, color=c, label=lab)
        ax[1].plot(cp, rd, "-o", ms=3, color=c, label=lab)
    ax[0].axvline(d / r, color="black", ls=":", lw=1)
    ax[0].text(d / r - 0.5, 0.82, "$T=d/r$", fontsize=8, ha="right",
               transform=ax[0].get_xaxis_transform())
    ax[0].set_xlabel("number of tasks $T$"); ax[0].set_ylabel("mean residual distortion")
    ax[0].legend(fontsize=8)
    ax[1].axvline(d, color="black", ls=":", lw=1); ax[1].text(d * 0.6, 0.1, "$r_{\\rm eff}=d$", fontsize=8)
    ax[1].set_xlabel(r"occupied capacity $r_{\rm eff}=\sum_j(1-\mathrm{keep}_j)$")
    ax[1].set_ylabel("mean residual distortion")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_softknee.pdf"); plt.close(fig)
    res["softknee_cap_at_T_hard"] = float(curves["hard (keep=0)"][1][-1])
    res["softknee_cap_at_T_soft1"] = float(curves[r"soft $\beta=1$"][1][-1])


# ===========================================================================
# (D) Per-direction gate beats the per-task gate (limitation #3).
#     A task shares a block with the past where SOME directions align (transfer)
#     and others conflict (forgetting). A per-task gate must share-all or
#     protect-all; a per-direction gate shares the aligned directions (variance
#     reduction) AND orthogonalizes the conflicting ones (no forgetting), getting
#     strictly the best of both.
# ===========================================================================
def exp_perdirection(k=10, noise=0.8, m=12, trials=4000):
    fracs = np.linspace(0.0, 1.0, 11)                 # fraction of shared directions that ALIGN
    protect = np.zeros(len(fracs)); share = np.zeros(len(fracs)); perdir = np.zeros(len(fracs))
    for fi, f in enumerate(fracs):
        nA = int(round(f * k))
        for _ in range(trials):
            rng = np.random.default_rng(int(RNG.integers(1 << 30)))
            ca = rng.standard_normal(k)
            cb = ca.copy(); cb[nA:] = -ca[nA:]        # first nA dirs align, rest conflict
            # one noisy coefficient estimate per task per direction (variance ~ noise^2/m)
            sd = noise / np.sqrt(m)
            ahat = ca + sd * rng.standard_normal(k)
            bhat = cb + sd * rng.standard_normal(k)
            # estimators of B's shared coefficients under each gate:
            est_protect = bhat                                   # isolate all (B alone)
            est_share = 0.5 * (ahat + bhat)                      # pool all (A+B)
            aligned = np.arange(k) < nA
            est_perdir = np.where(aligned, 0.5 * (ahat + bhat), bhat)  # pool aligned, isolate conflict
            protect[fi] += np.mean((est_protect - cb) ** 2)
            share[fi]   += np.mean((est_share - cb) ** 2)
            perdir[fi]  += np.mean((est_perdir - cb) ** 2)
    protect /= trials; share /= trials; perdir /= trials
    res["perdir_best_margin"] = float(np.min(np.minimum(protect, share) - perdir))
    res["perdir_always_best"] = bool(np.all(perdir <= np.minimum(protect, share) + 1e-9))

    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    ax.plot(fracs, protect, "-s", ms=4, color=RED, label="per-task: protect all (OGD)")
    ax.plot(fracs, share, "-o", ms=4, color=PURPLE, label="per-task: share all (naive)")
    ax.plot(fracs, perdir, "-^", ms=4, color=GREEN, label="per-direction gate (ours)")
    ax.set_xlabel("fraction of shared directions that align (transfer)")
    ax.set_ylabel("B's estimation error on the shared block")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_perdirection.pdf"); plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(FIGDIR, exist_ok=True)
    exp_feasibility()
    exp_cheap_gn()
    exp_soft_knee()
    exp_perdirection()
    with open("results_roadmap.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
