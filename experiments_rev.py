"""
Geometry of Forgetting -- revision experiments addressing reviewer feedback.

  (A) fig_breakdown.pdf    when does the linearized functional break down?
                           full-backprop MLPs, depth x learning-rate sweep.
  (B) fig_benchmark.pdf    real-data Split-Digits task-incremental benchmark,
                           frozen random-feature backbone (A1 exact);
                           naive / EWC / OGD-GPM / replay / IGFA.
  (C) fig_diagnostics.pdf  per-pair Delta^T Sigma Delta vs measured forgetting
                           across a 6-task sequence, and the Sigma spectrum.

Numbers are written to results_rev.json and quoted verbatim in the paper.
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
from sklearn.datasets import load_digits
from scipy.ndimage import rotate as _rotate

RNG = np.random.default_rng(1)
BLUE, RED, GREEN, ORANGE, PURPLE, GREY = \
    "#4C72B0", "#C44E52", "#55A868", "#DD8452", "#8172B3", "#777777"
FIGDIR = "figures"
res = {}


# ===========================================================================
# (A) Linearization breakdown: full-backprop MLPs of varying depth / step size
# ===========================================================================
def exp_breakdown():
    """
    Train task A, freeze the last-layer feature covariance Sigma_A and head w_A,
    then train task B with FULL backprop (every layer moves). The first-order
    prediction uses Sigma_A and the head displacement only. As depth and step
    size grow, the feature map drifts and the prediction degrades -- we quantify
    exactly how much.
    """
    def relu(z): return np.maximum(z, 0.0)

    d_in, h = 10, 48

    def make_task(rng):
        w = rng.standard_normal(d_in) / np.sqrt(d_in)
        X = rng.standard_normal((300, d_in))
        return X, X @ w

    def init_net(depth, rng):
        dims = [d_in] + [h] * depth
        Ws = [rng.standard_normal((dims[i], dims[i + 1])) / np.sqrt(dims[i])
              for i in range(depth)]
        bs = [np.zeros(dims[i + 1]) for i in range(depth)]
        v = rng.standard_normal(h) / np.sqrt(h)               # linear read-out
        return Ws, bs, v

    def forward(Ws, bs, v, X):
        a = X; acts = []
        for W, b in zip(Ws, bs):
            a = relu(a @ W + b); acts.append(a)
        return a @ v, acts  # acts[-1] = last-layer features

    def train(Ws, bs, v, X, y, lr, steps):
        n = len(X)
        for _ in range(steps):
            # forward with cache
            a = X; pre = []; cache = [X]
            for W, b in zip(Ws, bs):
                z = a @ W + b; a = relu(z); pre.append(z); cache.append(a)
            pred = a @ v
            g = (pred - y) / n
            dv = cache[-1].T @ g
            da = np.outer(g, v)
            dWs = [None] * len(Ws); dbs = [None] * len(bs)
            for i in reversed(range(len(Ws))):
                dz = da * (pre[i] > 0)
                dWs[i] = cache[i].T @ dz
                dbs[i] = dz.sum(0)
                da = dz @ Ws[i].T
            for i in range(len(Ws)):
                Ws[i] -= lr * dWs[i]; bs[i] -= lr * dbs[i]
            v -= lr * dv
        return Ws, bs, v

    depths = [1, 2, 3]
    lrs = [0.02, 0.05, 0.1, 0.2, 0.4]
    corr = {dp: [] for dp in depths}
    relerr = {dp: [] for dp in depths}
    for dp in depths:
        for lr in lrs:
            preds, meas = [], []
            for _ in range(40):
                rng = np.random.default_rng(int(RNG.integers(1 << 30)))
                Ws, bs, v = init_net(dp, rng)
                Xa, ya = make_task(rng); Xb, yb = make_task(rng)
                Ws, bs, v = train(Ws, bs, v, Xa, ya, lr, 300)
                _, acts_a = forward(Ws, bs, v, Xa)
                Fa = acts_a[-1]; Sa = Fa.T @ Fa / len(Xa); v0 = v.copy()
                la0 = 0.5 * np.mean((Fa @ v0 - ya) ** 2)
                Ws, bs, v = train(Ws, bs, v, Xb, yb, lr, 300)
                _, acts_a2 = forward(Ws, bs, v, Xa)
                la1 = 0.5 * np.mean((acts_a2[-1] @ v - ya) ** 2)
                measured = la1 - la0
                dv = v - v0
                predicted = 0.5 * dv @ Sa @ dv          # frozen-feature first-order
                meas.append(measured); preds.append(predicted)
            preds, meas = np.array(preds), np.array(meas)
            c = np.corrcoef(preds, meas)[0, 1]
            re = np.median(np.abs(preds - meas) / (np.abs(meas) + 1e-9))
            corr[dp].append(c); relerr[dp].append(re)
    res["breakdown_lrs"] = lrs
    res["breakdown_corr_depth1"] = [float(x) for x in corr[1]]
    res["breakdown_corr_depth3"] = [float(x) for x in corr[3]]
    res["breakdown_relerr_depth3"] = [float(x) for x in relerr[3]]

    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.7))
    for dp, c in zip(depths, [BLUE, GREEN, RED]):
        ax[0].plot(lrs, corr[dp], "-o", ms=4, color=c, label=f"depth {dp}")
        ax[1].plot(lrs, relerr[dp], "-o", ms=4, color=c, label=f"depth {dp}")
    ax[0].set_ylabel("Pearson $r$ (predicted vs measured)")
    ax[0].set_title("First-order fidelity vs step size", fontsize=9.5)
    ax[0].set_ylim(0, 1.02); ax[0].legend(fontsize=8)
    ax[1].set_ylabel("median relative error")
    ax[1].set_title("Prediction error grows with depth, step", fontsize=9.5)
    ax[1].legend(fontsize=8)
    for a in ax:
        a.set_xlabel("learning rate (feature drift)")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_breakdown.pdf"); plt.close(fig)


# ===========================================================================
# (B) Split-Digits task-incremental benchmark, frozen random-feature backbone
# ===========================================================================
def random_features(X, d, rng):
    R = rng.standard_normal((X.shape[1], d)) / np.sqrt(X.shape[1])
    b = rng.uniform(0, 2 * np.pi, d)
    Phi = np.maximum(0.0, np.cos(X @ R + b))                # frozen ReLU-cos features
    # normalize so the top eigenvalue of the global second-moment is ~1 (stable GD)
    lam_max = np.linalg.eigvalsh(Phi.T @ Phi / len(Phi))[-1]
    return Phi / np.sqrt(lam_max)


def exp_benchmark(d=300, epochs=600, lr=0.3, seed=7, make_fig=True):
    data = load_digits()
    X = data.data / 16.0
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    y = data.target
    rng = np.random.default_rng(seed)
    Phi = random_features(X, d, rng)                        # frozen feature map (A1 exact)
    C = 10
    tasks = [(2 * t, 2 * t + 1) for t in range(5)]          # 5 tasks, 2 classes each
    # train/test split per class
    idx_tr, idx_te = {}, {}
    for c in range(C):
        ic = np.where(y == c)[0]; rng.shuffle(ic)
        cut = int(0.7 * len(ic)); idx_tr[c] = ic[:cut]; idx_te[c] = ic[cut:]

    def task_data(classes, which):
        idx = np.concatenate([{"tr": idx_tr, "te": idx_te}[which][c] for c in classes])
        Yp = np.zeros((len(idx), C));
        for c in classes:
            Yp[y[idx] == c, c] = 1.0
        return Phi[idx], Yp, y[idx]

    def acc(W, classes):
        Xte, _, yte = task_data(classes, "te")
        logits = Xte @ W.T
        mask = np.full(C, -1e9); mask[list(classes)] = 0
        pred = np.argmax(logits + mask, axis=1)
        return float(np.mean(pred == yte))

    def grad(W, Xb, Yb):
        return (Xb @ W.T - Yb).T @ Xb / len(Xb)            # squared-error grad, C x d

    def basis_of(Xb, r):
        S = Xb.T @ Xb / len(Xb)
        w, V = np.linalg.eigh(S)
        return V[:, -r:]                                   # top-r eigenvectors (d x r)

    def run(method, r=20, ewc_lam=3.0, mem=10, sstar=0.5):
        W = np.zeros((C, d))
        occupied = np.zeros((d, 0))                        # OGD / IGFA subspace
        past_bases = []                                    # per-task bases (IGFA)
        fisher = np.zeros(d); W_star = np.zeros((C, d))    # EWC
        buffer_X, buffer_Y = [], []                        # replay
        hist = []                                          # acc matrix rows per task
        for ti, classes in enumerate(tasks):
            Xb, Yb, _ = task_data(classes, "tr")
            Bn = basis_of(Xb, r)
            # choose protected subspace
            if method == "ogd" and occupied.shape[1]:
                P = np.eye(d) - occupied @ occupied.T
            elif method == "igfa" and past_bases:
                protect = []
                for Bt in past_bases:
                    sim = np.linalg.norm(Bt.T @ Bn, "fro") ** 2 / r   # subspace overlap
                    if sim < sstar:                        # conflicting -> protect
                        protect.append(Bt)
                if protect:
                    Q = np.linalg.qr(np.hstack(protect))[0]
                    P = np.eye(d) - Q @ Q.T
                else:
                    P = np.eye(d)
            else:
                P = None
            # Xt,Yt are constant across this task's epochs -> use the
            # normal-equation gradient G = W A - B (identical to full-batch GD,
            # ~8x cheaper since A,B are precomputed once per task).
            if method == "replay" and buffer_X:
                Xt = np.vstack([Xb] + buffer_X); Yt = np.vstack([Yb] + buffer_Y)
            else:
                Xt, Yt = Xb, Yb
            A = Xt.T @ Xt / len(Xt); Bmat = Yt.T @ Xt / len(Xt)
            for ep in range(epochs):
                G = W @ A - Bmat
                if method == "ewc" and ti > 0:
                    G = G + ewc_lam * fisher[None, :] * (W - W_star)
                if P is not None:
                    G = G @ P.T
                W = W - lr * G
            # post-task bookkeeping
            past_bases.append(Bn)
            occupied = np.linalg.qr(np.hstack([occupied, Bn]))[0] if occupied.shape[1] else \
                np.linalg.qr(Bn)[0]
            if method == "ewc":
                fisher = fisher + np.diag(Xb.T @ Xb / len(Xb)); W_star = W.copy()
            if method == "replay":
                sel = rng.choice(len(Xb), min(mem, len(Xb)), replace=False)
                buffer_X.append(Xb[sel]); buffer_Y.append(Yb[sel])
            hist.append([acc(W, tasks[j]) for j in range(len(tasks))])
        hist = np.array(hist)                              # [task_trained, task_eval]
        final = hist[-1]                                   # acc on each task at end
        avg_acc = float(final.mean())
        diag = np.array([hist[t, t] for t in range(len(tasks))])
        forget = float(np.mean(diag[:-1] - final[:-1]))    # mean drop on earlier tasks
        return avg_acc, forget, final

    methods = ["naive", "ewc", "ogd", "replay", "igfa"]
    labels = {"naive": "Naive", "ewc": "EWC", "ogd": "OGD/GPM",
              "replay": "Replay (10/task)", "igfa": "igfa (ours)"}
    out = {}
    for m in methods:
        a, f, _ = run(m)
        out[m] = (a, f)
    res["benchmark"] = {m: {"avg_acc": out[m][0], "forgetting": out[m][1]} for m in methods}
    if not make_fig:
        return out

    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6))
    xs = np.arange(len(methods))
    cols = [GREY, ORANGE, BLUE, PURPLE, GREEN]
    ax[0].bar(xs, [out[m][0] for m in methods], color=cols)
    ax[0].set_xticks(xs); ax[0].set_xticklabels([labels[m] for m in methods], rotation=25, ha="right", fontsize=8)
    ax[0].set_ylabel("avg. final accuracy"); ax[0].set_ylim(0, 1.02)
    ax[0].set_title("Retention across 5 tasks", fontsize=9.5)
    for i, m in enumerate(methods):
        ax[0].text(i, out[m][0] + 0.02, f"{out[m][0]:.2f}", ha="center", fontsize=7.5)
    ax[1].bar(xs, [out[m][1] for m in methods], color=cols)
    ax[1].set_xticks(xs); ax[1].set_xticklabels([labels[m] for m in methods], rotation=25, ha="right", fontsize=8)
    ax[1].set_ylabel("avg. forgetting (lower better)")
    ax[1].set_title("Forgetting (frozen-feature, A1 exact)", fontsize=9.5)
    for i, m in enumerate(methods):
        ax[1].text(i, out[m][1] + 0.005, f"{out[m][1]:.2f}", ha="center", fontsize=7.5)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_benchmark.pdf"); plt.close(fig)


# ===========================================================================
# (C) Diagnostics: per-pair interference vs forgetting, and the Sigma spectrum
# ===========================================================================
def exp_diagnostics(d=300):
    data = load_digits()
    X = data.data / 16.0; X = (X - X.mean(0)) / (X.std(0) + 1e-6); y = data.target
    rng = np.random.default_rng(3)
    Phi = random_features(X, d, rng)
    tasks = [(2 * t, 2 * t + 1) for t in range(5)]
    C = 10
    # train naive sequentially; record per-task head and Sigma at the moment it finishes
    W = np.zeros((C, d)); lr = 0.3
    snap_W, snap_S, snap_cls = [], [], []
    for classes in tasks:
        idx = np.concatenate([np.where(y == c)[0] for c in classes])
        Xb = Phi[idx]; Yb = np.zeros((len(idx), C))
        for c in classes:
            Yb[y[idx] == c, c] = 1.0
        for _ in range(400):
            W = W - lr * ((Xb @ W.T - Yb).T @ Xb / len(Xb))
        snap_W.append(W.copy()); snap_S.append(Xb.T @ Xb / len(Xb)); snap_cls.append(classes)
    # interference vs forgetting for every (earlier task a, later state b>a)
    pred, meas = [], []
    for a in range(len(tasks) - 1):
        Sa = snap_S[a]; Wa = snap_W[a]; cls = snap_cls[a]; rows = list(cls)
        idx = np.concatenate([np.where(y == c)[0] for c in cls])
        Xb = Phi[idx]; Yb = np.zeros((len(idx), C))
        for c in cls:
            Yb[y[idx] == c, c] = 1.0
        # restrict the loss to the task's own output columns (the quantity the
        # functional predicts): 0.5 * sum_c mean_samples (residual_c)^2
        la_then = 0.5 * np.mean((Xb @ Wa[rows].T - Yb[:, rows]) ** 2, axis=0).sum()
        for b in range(a + 1, len(tasks)):
            Wb = snap_W[b]
            dW = Wb[rows] - Wa[rows]
            predicted = 0.5 * np.sum([dW[i] @ Sa @ dW[i] for i in range(len(rows))])
            la_now = 0.5 * np.mean((Xb @ Wb[rows].T - Yb[:, rows]) ** 2, axis=0).sum()
            pred.append(predicted); meas.append(la_now - la_then)
    pred, meas = np.array(pred), np.array(meas)
    corr = float(np.corrcoef(pred, meas)[0, 1]) if len(pred) > 1 else float("nan")
    res["diagnostics_corr"] = corr

    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6))
    lim = max(pred.max(), meas.max()) * 1.1
    if not np.isfinite(lim) or lim <= 0:
        lim = 1.0
    ax[0].plot([0, lim], [0, lim], "--", color=GREY, lw=1)
    ax[0].scatter(pred, meas, s=40, color=BLUE)
    ax[0].set_xlabel(r"predicted $\frac{1}{2}\sum_c \Delta_c^\top \Sigma_A \Delta_c$")
    ax[0].set_ylabel("measured forgetting")
    ax[0].set_title(f"Split-Digits, frozen features: $r={corr:.3f}$", fontsize=9.5)
    ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim)
    # spectrum of one task's Sigma
    w = np.linalg.eigvalsh(snap_S[0])[::-1]
    ax[1].semilogy(np.arange(1, len(w) + 1), np.maximum(w, 1e-12), color=PURPLE)
    eff = (w.sum() ** 2) / (np.sum(w ** 2))
    res["diagnostics_eff_rank"] = float(eff)
    ax[1].axvline(eff, color=GREY, ls="--", lw=1)
    ax[1].annotate(f"eff. rank $\\approx${eff:.0f}", xy=(eff, w[0] * 0.2),
                   xytext=(len(w) * 0.34, w[0] * 0.4), fontsize=8.5,
                   ha="left", va="center",
                   arrowprops=dict(arrowstyle="->", color=GREY, lw=0.9))
    ax[1].set_xlabel("index"); ax[1].set_ylabel(r"eigenvalue of $\Sigma_A$")
    ax[1].set_title("Feature spectrum sets removable budget", fontsize=9.5)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_diagnostics.pdf"); plt.close(fig)


# ===========================================================================
# (D) Rotated-Digits domain-incremental benchmark: SIMILAR tasks, where IGFA's
#     sharing gate is expected to beat unconditional orthogonal projection.
# ===========================================================================
def rotate_digits_once(angles=(0, 12, 24, 36, 48)):
    """Pre-compute the (slow, seed-independent) rotated raw images once, so a
    multi-seed sweep does not repeat the scipy rotation for every seed."""
    imgs = (load_digits().data / 16.0).reshape(-1, 8, 8)
    return {a: np.stack([_rotate(im, a, reshape=False, order=1) for im in imgs])
            for a in angles}


def exp_benchmark2(d=300, epochs=600, lr=0.3, angles=(0, 12, 24, 36, 48),
                   seed=11, make_fig=True, rot_imgs=None):
    data = load_digits()
    imgs = (data.data / 16.0).reshape(-1, 8, 8)
    y = data.target
    rng = np.random.default_rng(seed)
    C = 10
    # one frozen feature map shared across all rotations (the fixed backbone)
    R = rng.standard_normal((64, d)) / np.sqrt(64)
    bphase = rng.uniform(0, 2 * np.pi, d)

    def feats(imgs_rot):
        Xp = np.maximum(0.0, np.cos(imgs_rot.reshape(len(imgs_rot), -1) @ R + bphase))
        return Xp

    # build per-rotation feature sets, normalized by a shared global scale
    if rot_imgs is None:
        rot_imgs = {a: np.stack([_rotate(im, a, reshape=False, order=1) for im in imgs])
                    for a in angles}
    rot = {a: feats(rot_imgs[a]) for a in angles}
    scale = np.sqrt(np.linalg.eigvalsh(np.vstack(list(rot.values())).T @
                                       np.vstack(list(rot.values())) /
                                       (len(imgs) * len(angles)))[-1])
    rot = {a: rot[a] / scale for a in angles}
    # train/test split (shared indices across rotations)
    idx = np.arange(len(imgs)); rng.shuffle(idx)
    cut = int(0.7 * len(idx)); tr, te = idx[:cut], idx[cut:]

    def task_train(a):
        Yp = np.zeros((len(tr), C)); Yp[np.arange(len(tr)), y[tr]] = 1.0
        return rot[a][tr], Yp

    def acc(W, a):
        pred = np.argmax(rot[a][te] @ W.T, axis=1)            # domain-IL: full 10-way
        return float(np.mean(pred == y[te]))

    def basis_of(Xb, r):
        w, V = np.linalg.eigh(Xb.T @ Xb / len(Xb))
        return V[:, -r:]

    # report mean pairwise subspace similarity (justifies the gate firing)
    bases = {a: basis_of(rot[a][tr], 20) for a in angles}
    sims = [np.linalg.norm(bases[angles[i]].T @ bases[angles[j]], "fro") ** 2 / 20
            for i in range(len(angles)) for j in range(i + 1, len(angles))]
    res["rot_mean_subspace_sim"] = float(np.mean(sims))

    def run(method, r=20, ewc_lam=3.0, mem=10, sstar=0.65):
        W = np.zeros((C, d)); occupied = np.zeros((d, 0)); past = []
        fisher = np.zeros(d); W_star = np.zeros((C, d)); bufX, bufY = [], []
        hist = []
        for ti, a in enumerate(angles):
            Xb, Yb = task_train(a)
            Bn = basis_of(Xb, r)
            if method == "ogd" and occupied.shape[1]:
                P = np.eye(d) - occupied @ occupied.T
            elif method == "igfa" and past:
                protect = [Bt for Bt in past
                           if np.linalg.norm(Bt.T @ Bn, "fro") ** 2 / r < sstar]
                if protect:
                    Q = np.linalg.qr(np.hstack(protect))[0]; P = np.eye(d) - Q @ Q.T
                else:
                    P = np.eye(d)
            else:
                P = None
            if method == "replay" and bufX:
                Xt = np.vstack([Xb] + bufX); Yt = np.vstack([Yb] + bufY)
            else:
                Xt, Yt = Xb, Yb
            A = Xt.T @ Xt / len(Xt); Bmat = Yt.T @ Xt / len(Xt)   # normal-equation GD (identical, ~8x faster)
            for ep in range(epochs):
                G = W @ A - Bmat
                if method == "ewc" and ti > 0:
                    G = G + ewc_lam * fisher[None, :] * (W - W_star)
                if P is not None:
                    G = G @ P.T
                W = W - lr * G
            past.append(Bn)
            occupied = np.linalg.qr(np.hstack([occupied, Bn]))[0] if occupied.shape[1] \
                else np.linalg.qr(Bn)[0]
            if method == "ewc":
                fisher = fisher + np.diag(Xb.T @ Xb / len(Xb)); W_star = W.copy()
            if method == "replay":
                sel = rng.choice(len(tr), min(mem * C, len(tr)), replace=False)
                bufX.append(Xb[sel]); bufY.append(Yb[sel])
            hist.append([acc(W, angles[j]) for j in range(len(angles))])
        hist = np.array(hist); final = hist[-1]
        diag = np.array([hist[t, t] for t in range(len(angles))])
        return float(final.mean()), float(np.mean(diag[:-1] - final[:-1]))

    methods = ["naive", "ewc", "ogd", "replay", "igfa"]
    out = {m: run(m) for m in methods}
    res["benchmark2"] = {m: {"avg_acc": out[m][0], "forgetting": out[m][1]} for m in methods}
    if not make_fig:
        return out

    labels = {"naive": "Naive", "ewc": "EWC", "ogd": "OGD/GPM",
              "replay": "Replay", "igfa": "igfa (ours)"}
    cols = [GREY, ORANGE, BLUE, PURPLE, GREEN]
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6))
    xs = np.arange(len(methods))
    ax[0].bar(xs, [out[m][0] for m in methods], color=cols)
    ax[0].set_xticks(xs); ax[0].set_xticklabels([labels[m] for m in methods], rotation=25, ha="right", fontsize=8)
    ax[0].set_ylabel("avg. final accuracy"); ax[0].set_ylim(0, 1.0)
    ax[0].set_title("Rotated-Digits (similar tasks): retention", fontsize=9.5)
    for i, m in enumerate(methods):
        ax[0].text(i, out[m][0] + 0.015, f"{out[m][0]:.2f}", ha="center", fontsize=7.5)
    ax[1].bar(xs, [out[m][1] for m in methods], color=cols)
    ax[1].set_xticks(xs); ax[1].set_xticklabels([labels[m] for m in methods], rotation=25, ha="right", fontsize=8)
    ax[1].set_ylabel("avg. forgetting (lower better)")
    ax[1].set_title(f"igfa shares aligned tasks (mean sim {np.mean(sims):.2f})", fontsize=9.5)
    for i, m in enumerate(methods):
        ax[1].text(i, out[m][1], f"{out[m][1]:.2f}", ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_benchmark2.pdf"); plt.close(fig)


# ===========================================================================
# (E) Resolving the deep/large-step breakdown with the segment-averaged curvature
#     (the architecture-general forgetting identity).
# ===========================================================================
def exp_breakdown_fix():
    """
    The frozen-feature predictor uses the curvature of L_A at w_A only. The exact
    architecture-general identity uses the secant-averaged Hessian H_bar along the
    displacement w_A -> w_B. We compare, in the failing regime (depth 1-3, large
    learning rate), the correlation of measured forgetting with:
      P_init : local quadratic at w_A           (the frozen-feature formula)
      P_avg  : segment-averaged curvature        (the general identity)
    estimated from K forward evaluations of L_A along the segment (no full Hessian).
    """
    def relu(z): return np.maximum(z, 0.0)
    d_in, h = 10, 48

    def make_task(rng):
        w = rng.standard_normal(d_in) / np.sqrt(d_in)
        X = rng.standard_normal((300, d_in)); return X, X @ w

    def init(depth, rng):
        dims = [d_in] + [h] * depth
        Ws = [rng.standard_normal((dims[i], dims[i+1])) / np.sqrt(dims[i]) for i in range(depth)]
        bs = [np.zeros(dims[i+1]) for i in range(depth)]
        v = rng.standard_normal(h) / np.sqrt(h)
        return Ws, bs, v

    def flat(Ws, bs, v): return np.concatenate([p.ravel() for p in Ws] + [b for b in bs] + [v])

    def unflat(vec, depth):
        dims = [d_in] + [h] * depth; Ws = []; i = 0
        for k in range(depth):
            n = dims[k] * dims[k+1]; Ws.append(vec[i:i+n].reshape(dims[k], dims[k+1])); i += n
        bs = []
        for k in range(depth):
            bs.append(vec[i:i+dims[k+1]]); i += dims[k+1]
        v = vec[i:i+h]; return Ws, bs, v

    def loss(vec, depth, X, y):
        Ws, bs, v = unflat(vec, depth); a = X
        for W, b in zip(Ws, bs):
            a = relu(a @ W + b)
        return 0.5 * np.mean((a @ v - y) ** 2)

    def train(Ws, bs, v, X, y, lr, steps):
        n = len(X)
        for _ in range(steps):
            a = X; pre = []; cache = [X]
            for W, b in zip(Ws, bs):
                z = a @ W + b; a = relu(z); pre.append(z); cache.append(a)
            g = (a @ v - y) / n; dv = cache[-1].T @ g; da = np.outer(g, v)
            dWs = [None]*len(Ws); dbs = [None]*len(bs)
            for i in reversed(range(len(Ws))):
                dz = da * (pre[i] > 0); dWs[i] = cache[i].T @ dz; dbs[i] = dz.sum(0); da = dz @ Ws[i].T
            for i in range(len(Ws)):
                Ws[i] -= lr*dWs[i]; bs[i] -= lr*dbs[i]
            v -= lr*dv
        return Ws, bs, v

    depths = [1, 2, 3]; lr = 0.4; K = 21
    r_init = {}; r_avg = {}
    for dp in depths:
        meas, Pin, Pav = [], [], []
        for _ in range(60):
            rng = np.random.default_rng(int(RNG.integers(1 << 30)))
            Ws, bs, v = init(dp, rng)
            Xa, ya = make_task(rng); Xb, yb = make_task(rng)
            Ws, bs, v = train(Ws, bs, v, Xa, ya, lr, 300); wA = flat(Ws, bs, v)
            Ws, bs, v = train(Ws, bs, v, Xb, yb, lr, 300); wB = flat(Ws, bs, v)
            dW = wB - wA
            ts = np.linspace(0, 1, K)
            phi = np.array([loss(wA + t*dW, dp, Xa, ya) for t in ts])  # L_A along segment
            measured = phi[-1] - phi[0]
            dt = ts[1] - ts[0]
            dphi0 = (phi[1] - phi[0]) / dt                            # phi'(0)
            d2 = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dt**2           # phi''(t) interior
            t_in = ts[1:-1]
            P_init = dphi0 + 0.5 * d2[0]                              # local quadratic at w_A
            P_avg = dphi0 + np.sum((1 - t_in) * d2) * dt              # segment-averaged
            meas.append(measured); Pin.append(P_init); Pav.append(P_avg)
        meas = np.array(meas); Pin = np.array(Pin); Pav = np.array(Pav)
        r_init[dp] = float(np.corrcoef(Pin, meas)[0, 1])
        r_avg[dp] = float(np.corrcoef(Pav, meas)[0, 1])
    res["breakdown_fix_r_init"] = r_init
    res["breakdown_fix_r_avg"] = r_avg

    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    xs = np.arange(len(depths)); w = 0.36
    ax.bar(xs - w/2, [r_init[d] for d in depths], w, color=RED, label=r"$P_{\rm init}$: frozen $\Sigma_A$ (old)")
    ax.bar(xs + w/2, [r_avg[d] for d in depths], w, color=GREEN, label=r"$P_{\rm avg}$: segment-averaged $\bar H_A$")
    for i, d in enumerate(depths):
        ax.text(i - w/2, r_init[d] + 0.02, f"{r_init[d]:.2f}", ha="center", fontsize=7.5)
        ax.text(i + w/2, r_avg[d] + 0.02, f"{r_avg[d]:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(xs); ax.set_xticklabels([f"depth {d}" for d in depths])
    ax.set_ylabel("Pearson $r$ (predicted vs measured)"); ax.set_ylim(0, 1.08)
    ax.set_title(f"Segment-averaged curvature restores the identity (lr={lr})", fontsize=9.5)
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_breakdown_fix.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(FIGDIR, exist_ok=True)
    exp_breakdown()
    exp_breakdown_fix()
    exp_benchmark()
    exp_benchmark2()
    exp_diagnostics()
    with open("results_rev.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
