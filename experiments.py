"""
Geometry of Forgetting -- experimental backbone.

All experiments live in the linear-on-features regime: a frozen d-dimensional
feature map phi(x) with a trainable linear head w. This is EXACT for the
frozen-backbone / PEFT regime (the practically dominant CL setting), and the
first-order (NTK) model for full networks. Each task t is a regression problem
with feature second-moment Sigma_t = E[phi phi^T] and optimum w_t*; the per-task
excess loss is L_t(w) = 1/2 (w-w_t*)^T Sigma_t (w-w_t*).

Outputs (written to ./figures and ./results.json):
  fig_interference_validation.pdf   interference functional Delta^T Sigma Delta = forgetting
  fig_allocation_floor.pdf          lossless allocation + distortion floor vs overlap
  fig_signchange.pdf                similarity sign-change s*
  fig_capacity.pdf                  rate-distortion floor vs capacity (number of tasks)
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
BLUE, RED, GREEN, GREY = "#4C72B0", "#C44E52", "#55A868", "#888888"
FIGDIR = "figures"
results = {}


# ----------------------------------------------------------------------------
# core linear-CL primitives (exact)
# ----------------------------------------------------------------------------
def subspace_cov(d, rank, rng, eig=1.0, basis=None):
    """Second-moment matrix with given orthonormal active basis (d x rank)."""
    if basis is None:
        basis = np.linalg.qr(rng.standard_normal((d, rank)))[0][:, :rank]
    Sigma = basis @ (eig * np.eye(rank)) @ basis.T
    return Sigma, basis


def optimum_in(basis, rng):
    """Unit-norm task optimum living in the active subspace span(basis)."""
    w = basis @ rng.standard_normal(basis.shape[1])
    return w / np.linalg.norm(w)


def excess_loss(w, Sigma, wstar):
    dv = w - wstar
    return 0.5 * dv @ Sigma @ dv


def train(w, Sigma, wstar, P=None, lr=0.4, steps=4000):
    """Projected gradient descent on L_t. P (d x d) restricts the search space."""
    w = w.copy()
    for _ in range(steps):
        g = Sigma @ (w - wstar)
        if P is not None:
            g = P @ g
        w = w - lr * g
    return w


def null_projector(basis):
    """Projector onto the orthogonal complement of span(basis): I - QQ^T."""
    d = basis.shape[0]
    return np.eye(d) - basis @ basis.T


# ----------------------------------------------------------------------------
# Experiment 1: the interference functional IS the forgetting
# ----------------------------------------------------------------------------
def exp_interference_validation(d=24, rank=4, n_pairs=160):
    """
    (A) Linear head: measured backward-forgetting of task A after learning B,
        vs the predicted interference energy 1/2 Delta^T Sigma_A Delta. Exact.
    (B) Nonlinear 1-hidden-layer net: the same prediction computed from the
        *last-layer feature covariance* tracks measured forgetting to first order.
    """
    # ---- (A) exact linear head ----
    pred, meas = [], []
    for _ in range(n_pairs):
        Sa, Qa = subspace_cov(d, rank, RNG)
        Sb, Qb = subspace_cov(d, rank, RNG)
        wa, wb = optimum_in(Qa, RNG), optimum_in(Qb, RNG)
        w_after_a = train(np.zeros(d), Sa, wa)              # == wa
        w_after_b = train(w_after_a, Sb, wb)               # naive sequential
        forgetting = excess_loss(w_after_b, Sa, wa) - excess_loss(w_after_a, Sa, wa)
        delta = w_after_b - wa
        predicted = 0.5 * delta @ Sa @ delta
        meas.append(forgetting); pred.append(predicted)
    pred, meas = np.array(pred), np.array(meas)
    rel_err_linear = float(np.max(np.abs(pred - meas)) / (np.max(meas) + 1e-12))
    results["interference_linear_max_rel_err"] = rel_err_linear

    # ---- (B) nonlinear net, prediction from last-layer features ----
    def relu(z): return np.maximum(z, 0.0)

    def make_task(d_in, rng):
        Wt = rng.standard_normal((d_in,)) / np.sqrt(d_in)
        X = rng.standard_normal((256, d_in))
        y = X @ Wt
        return X, y

    d_in, h = 8, 64
    pred_nl, meas_nl = [], []
    for _ in range(60):
        rng = np.random.default_rng(int(RNG.integers(1 << 30)))
        W1 = rng.standard_normal((d_in, h)) / np.sqrt(d_in)   # frozen feature map
        b1 = rng.standard_normal(h) * 0.1
        feat = lambda X: relu(X @ W1 + b1)
        Xa, ya = make_task(d_in, rng); Xb, yb = make_task(d_in, rng)
        Fa, Fb = feat(Xa), feat(Xb)
        Sa = Fa.T @ Fa / len(Xa)
        # closed-form ridge heads (tiny ridge for stability)
        lam = 1e-3
        head_a = np.linalg.solve(Fa.T @ Fa + lam * np.eye(h), Fa.T @ ya)
        # learn B starting from head_a (a few GD steps -> realistic partial move)
        w = head_a.copy(); lr = 0.05
        for _ in range(400):
            w -= lr * (Fb.T @ (Fb @ w - yb)) / len(Xb)
        forget = 0.5 * np.mean((Fa @ w - ya) ** 2) - 0.5 * np.mean((Fa @ head_a - ya) ** 2)
        delta = w - head_a
        predicted = 0.5 * delta @ Sa @ delta
        meas_nl.append(forget); pred_nl.append(predicted)
    pred_nl, meas_nl = np.array(pred_nl), np.array(meas_nl)
    corr_nl = float(np.corrcoef(pred_nl, meas_nl)[0, 1])
    results["interference_nonlinear_corr"] = corr_nl

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.7))
    for a, (p, m, ttl, sub) in zip(
        ax,
        [(pred, meas, "Linear head (frozen features): exact",
          f"max rel. error $={rel_err_linear:.1e}$"),
         (pred_nl, meas_nl, "Nonlinear net: first-order prediction",
          f"Pearson $r={corr_nl:.3f}$")]):
        lim = max(p.max(), m.max()) * 1.05
        a.plot([0, lim], [0, lim], color=GREY, lw=1, ls="--", zorder=0)
        a.scatter(p, m, s=18, color=BLUE, alpha=0.8, edgecolor="none")
        a.set_xlabel(r"predicted  $\frac{1}{2}\,\Delta^\top \Sigma_A \Delta$")
        a.set_ylabel("measured forgetting of task A")
        a.set_title(ttl, fontsize=9.5)
        a.text(0.04, 0.92, sub, transform=a.transAxes, fontsize=9)
        a.set_xlim(0, lim); a.set_ylim(0, lim)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_interference_validation.pdf"); plt.close(fig)


# ----------------------------------------------------------------------------
# Experiment 2: lossless allocation + distortion floor vs subspace overlap
# ----------------------------------------------------------------------------
def block_bases(d, rng, k_shared, n_excl):
    """Orthonormal shared / A-exclusive / B-exclusive feature blocks."""
    tot = k_shared + 2 * n_excl
    M = np.linalg.qr(rng.standard_normal((d, tot)))[0][:, :tot]
    Sh = M[:, :k_shared]
    Aex = M[:, k_shared:k_shared + n_excl]
    Bex = M[:, k_shared + n_excl:tot]
    return Sh, Aex, Bex


def exp_allocation_floor(d=24, rank=6):
    # ---- (A) training curves on partially shared features with a CONFLICT on the
    #          shared dims: naive destroys A; allocation converts that irreversible
    #          forgetting into deferred (recoverable) under-fitting of B. ----
    Sh, Aex, Bex = block_bases(d, RNG, k_shared=2, n_excl=2)
    Qa = np.hstack([Sh, Aex]); Qb = np.hstack([Sh, Bex])
    Sa = Qa @ Qa.T; Sb = Qb @ Qb.T
    us = Sh @ RNG.standard_normal(2)
    wa = us + Aex @ RNG.standard_normal(2)
    wb = -us + Bex @ RNG.standard_normal(2)              # conflict on shared dims
    w_a = train(np.zeros(d), Sa, wa)
    Pnull = null_projector(Qa)                          # IGFA: orthogonalize against A

    steps_track, lr = 1200, 0.4
    naive_A, naive_B, alloc_A, alloc_B = [], [], [], []
    wn, wo = w_a.copy(), w_a.copy()
    for _ in range(steps_track):
        wn = wn - lr * (Sb @ (wn - wb))
        wo = wo - lr * (Pnull @ (Sb @ (wo - wb)))
        naive_A.append(excess_loss(wn, Sa, wa)); naive_B.append(excess_loss(wn, Sb, wb))
        alloc_A.append(excess_loss(wo, Sa, wa)); alloc_B.append(excess_loss(wo, Sb, wb))
    results["lossless_naive_forgetA"] = float(naive_A[-1])
    results["lossless_alloc_forgetA"] = float(alloc_A[-1])
    results["lossless_alloc_lossB"] = float(alloc_B[-1])
    results["lossless_naive_lossB"] = float(naive_B[-1])

    # ---- (B) distortion floor vs overlap: shared dims carry CONFLICTING unit
    #          targets, averaged over random realizations. ----
    ks = np.arange(0, rank + 1)
    overlap = ks / rank
    n_tr = 40
    naive_f = np.zeros(len(ks)); alloc_f = np.zeros(len(ks))
    alloc_b = np.zeros(len(ks)); merge_floor = np.zeros(len(ks))
    for i, k in enumerate(ks):
        nex = rank - k
        for _ in range(n_tr):
            Sh, Aex, Bex = block_bases(d, RNG, k_shared=k, n_excl=nex)
            Qa = np.hstack([b for b in [Sh, Aex] if b.shape[1]])
            Qb = np.hstack([b for b in [Sh, Bex] if b.shape[1]])
            Sa = Qa @ Qa.T; Sb = Qb @ Qb.T
            uc = Sh @ RNG.standard_normal(k) if k else np.zeros(d)
            wa = uc + (Aex @ RNG.standard_normal(nex) if nex else 0)
            wb = -uc + (Bex @ RNG.standard_normal(nex) if nex else 0)  # conflict on shared
            wa = wa / np.linalg.norm(wa); wb = wb / np.linalg.norm(wb)
            w_a = train(np.zeros(d), Sa, wa)
            wn = train(w_a, Sb, wb)
            wo = train(w_a, Sb, wb, P=null_projector(Qa))
            wJ = np.linalg.pinv(Sa + Sb) @ (Sa @ wa + Sb @ wb)
            naive_f[i] += excess_loss(wn, Sa, wa)
            alloc_f[i] += excess_loss(wo, Sa, wa)
            alloc_b[i] += excess_loss(wo, Sb, wb)
            merge_floor[i] += excess_loss(wJ, Sa, wa) + excess_loss(wJ, Sb, wb)
    naive_f /= n_tr; alloc_f /= n_tr; alloc_b /= n_tr; merge_floor /= n_tr
    results["floor_at_disjoint"] = float(merge_floor[0])
    results["floor_at_full_overlap"] = float(merge_floor[-1])
    results["interference_at_full_overlap"] = float(naive_f[-1])

    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.8))
    t = np.arange(steps_track)
    ax[0].plot(t, naive_A, color=RED, label="task A — naive (forgets)")
    ax[0].plot(t, alloc_A, color=GREEN, label="task A — allocation (preserved)")
    ax[0].plot(t, naive_B, color=RED, ls=":", label="task B — naive")
    ax[0].plot(t, alloc_B, color=GREEN, ls=":", label="task B — allocation")
    ax[0].set_xlabel("gradient steps on task B"); ax[0].set_ylabel("excess loss")
    ax[0].set_title("Allocation relocates cost:\nirreversible forgetting $\\to$ deferred plasticity",
                    fontsize=9.5)
    ax[0].legend(fontsize=7.2, loc="center right"); ax[0].set_ylim(bottom=-0.7)

    ax[1].plot(overlap, naive_f, "-o", color=RED, ms=4,
               label="interference (naive forgets A = alloc. defers B)")
    ax[1].plot(overlap, alloc_f, "-s", color=GREEN, ms=4, label="A forgetting under allocation")
    ax[1].plot(overlap, merge_floor, "--", color="black", lw=1.6,
               label="distortion floor (optimal merge)")
    ax[1].set_xlabel(r"feature-support overlap  $k/r$")
    ax[1].set_ylabel("loss")
    ax[1].set_title("Interference is unavoidable iff supports overlap", fontsize=9.5)
    ax[1].legend(fontsize=7.0, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_allocation_floor.pdf"); plt.close(fig)


# ----------------------------------------------------------------------------
# Experiment 3: similarity sign-change s*
# ----------------------------------------------------------------------------
def exp_signchange(k=6, m=5, noise=0.6, lam=0.05, trials=600):
    """
    The sign-change is a multitask bias-variance effect on a shared feature block.
    With exact, full-data least squares the two strategies are an exact relocation
    of the same cost; the asymmetry appears only under finite data. We compare:

      SHARE (pool)        : tie the shared-block coefficient across A and B and fit
                            it from BOTH tasks' data -> halves estimation variance,
                            but is biased when the task targets differ.
      ORTHOGONALIZE (iso) : estimate each task's coefficient from its own data only.

    Targets u_A, u_B on the shared block have cosine similarity s. Sharing wins
    (variance reduction beats bias) above a threshold s*; isolating wins below it.
    """
    sims = np.linspace(-1.0, 1.0, 21)
    share_tot = np.zeros(len(sims)); orth_tot = np.zeros(len(sims))
    I = lam * np.eye(k)
    for _ in range(trials):
        uA = RNG.standard_normal(k); uA /= np.linalg.norm(uA)
        up = RNG.standard_normal(k); up -= (up @ uA) * uA; up /= np.linalg.norm(up)
        XA = RNG.standard_normal((m, k)); XB = RNG.standard_normal((m, k))
        nA = noise * RNG.standard_normal(m); nB = noise * RNG.standard_normal(m)
        for j, s in enumerate(sims):
            uB = s * uA + np.sqrt(max(0.0, 1 - s * s)) * up
            yA = XA @ uA + nA; yB = XB @ uB + nB
            uA_iso = np.linalg.solve(XA.T @ XA + I, XA.T @ yA)
            uB_iso = np.linalg.solve(XB.T @ XB + I, XB.T @ yB)
            Xp = np.vstack([XA, XB]); yp = np.concatenate([yA, yB])
            u_pool = np.linalg.solve(Xp.T @ Xp + I, Xp.T @ yp)
            orth_tot[j] += 0.5 * (np.sum((uA_iso - uA) ** 2) + np.sum((uB_iso - uB) ** 2))
            share_tot[j] += 0.5 * (np.sum((u_pool - uA) ** 2) + np.sum((u_pool - uB) ** 2))
    share_tot /= trials; orth_tot /= trials
    net = orth_tot - share_tot
    idx = np.where(np.diff(np.sign(net)))[0]
    s_star = (float(sims[idx[0]] - net[idx[0]] * (sims[idx[0] + 1] - sims[idx[0]]) /
                    (net[idx[0] + 1] - net[idx[0]])) if len(idx) else float("nan"))
    results["sign_change_s_star"] = s_star

    fig, ax = plt.subplots(figsize=(5.3, 3.9))
    ax.axhline(0, color=GREY, lw=0.8)
    ax.plot(sims, share_tot, "-o", color=RED, ms=3, label="share / pool: $L_A+L_B$")
    ax.plot(sims, orth_tot, "-s", color=GREEN, ms=3, label="orthogonalize / isolate: $L_A+L_B$")
    ax.plot(sims, net, "-", color=BLUE, lw=2, label="net benefit of sharing")
    if not np.isnan(s_star):
        ax.axvline(s_star, color="black", ls="--", lw=1.2)
        ax.text(s_star + 0.03, ax.get_ylim()[1] * 0.62, fr"$s^\ast={s_star:.2f}$", fontsize=9)
    ax.set_xlabel(r"shared-target similarity  $s$")
    ax.set_ylabel("post-merge loss / benefit")
    ax.set_title("Sign-change: share above $s^\\ast$, orthogonalize below", fontsize=9.5)
    ax.legend(fontsize=8.2, loc="upper right", frameon=True, framealpha=0.9,
              facecolor="white", edgecolor="none").set_zorder(5)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_signchange.pdf"); plt.close(fig)


# ----------------------------------------------------------------------------
# Experiment 4: rate-distortion floor vs capacity (number of tasks)
# ----------------------------------------------------------------------------
def exp_capacity(rank=2, n_tasks=16, trials=16):
    """
    Stream of T random rank-`rank` tasks under orthogonal allocation in dimension
    d. Earlier tasks are never forgotten, but once the cumulative occupied rank
    T*rank exceeds the capacity d there is no room left to fit new tasks, so the
    mean residual distortion turns on at the knee T*rank ~ d. This is the geometric
    analogue of an N + kappa*M capacity budget: d trades against the number of tasks.
    """
    dims = [16, 28, 40]
    curves = {}; knee_vals = {}
    for d in dims:
        mean_resid = np.zeros(n_tasks)
        for _ in range(trials):
            Sig, wst = [], []
            occupied = np.zeros((d, 0)); w = np.zeros(d)
            for t in range(n_tasks):
                Q = np.linalg.qr(RNG.standard_normal((d, rank)))[0][:, :rank]
                S = Q @ Q.T; ws = optimum_in(Q, RNG)
                P = null_projector(occupied) if occupied.shape[1] else None
                w = train(w, S, ws, P=P, steps=2500)
                Sig.append(S); wst.append(ws)
                stacked = np.hstack([occupied, Q]) if occupied.shape[1] else Q
                occupied = np.linalg.qr(stacked)[0][:, :min(stacked.shape[1], d)]
                mean_resid[t] += np.mean([excess_loss(w, Sig[j], wst[j]) for j in range(t + 1)])
        curves[d] = mean_resid / trials
        knee_vals[d] = d / rank
    results["capacity_dims"] = dims
    results["capacity_rank"] = rank
    results["capacity_final_resid"] = {str(d): float(curves[d][-1]) for d in dims}

    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    xs = np.arange(1, n_tasks + 1)
    for d, c in zip(dims, [BLUE, GREEN, RED]):
        ax.plot(xs, curves[d], "-o", ms=3, color=c, label=f"$d={d}$ (knee $T={d//rank}$)")
        ax.axvline(knee_vals[d], color=c, ls=":", lw=1)
    ax.set_xlabel("number of tasks in the stream  $T$")
    ax.set_ylabel("mean residual distortion")
    ax.set_title(f"Capacity floor turns on at $T\\,r \\approx d$  ($r={rank}$)", fontsize=9.5)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_capacity.pdf"); plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(FIGDIR, exist_ok=True)
    exp_interference_validation()
    exp_allocation_floor()
    exp_signchange()
    exp_capacity()
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
