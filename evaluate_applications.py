# =====================================================================
#  Application-domain probes for the interference functional (exact-A1, fast, synthetic).
#    A/B  PINNs: does 1/2 Delta^T Sigma_PDE Delta predict the physics-loss spike, and does
#         null-space projection fit data without violating the PDE?
#    C    Process drift (dry-electrode extrusion sim): subspace-velocity fault detection.
#    D    Sigma -> policy gradients: Fisher-metric IGFA gate retains an old RL environment.
#    E    Meta-learn a feature map with a Tr(Sigma_A Sigma_B) penalty to shrink the floor.
#  MuJoCo (D) and MAML/Omniglot (E) full versions need GPU; here we test the cores.
# =====================================================================
import numpy as np
rng = np.random.default_rng(0)
def hdr(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# =====================================================================
#  A / B  PINN: 1D Poisson u''=f with a frozen sin random-feature basis (A1 exact).
#  The PDE constraint acts on phi''(x); Sigma_PDE = E[phi'' phi''^T].
# =====================================================================
def AB_pinn():
    hdr("A/B  PINN (1D Poisson): interference functional predicts & prevents PDE violation")
    d = 80
    a = rng.uniform(1.0, 3.2, d); b = rng.uniform(0, 2 * np.pi, d)
    phi = lambda x: np.sin(a * x[:, None] + b)                 # frozen features  (N x d)
    phi2 = lambda x: -(a ** 2) * np.sin(a * x[:, None] + b)    # phi''  (the PDE operator on phi)
    xc = np.linspace(0, 1, 150)                                # collocation (physics) points
    xd = rng.uniform(0, 1, 40)                                 # interior sensor points
    P2c, Pd = phi2(xc), phi(xd)
    P2c = P2c / np.sqrt(np.linalg.eigvalsh(P2c.T @ P2c / len(xc))[-1])   # unit-scale operators (stable)
    Pd = Pd / np.sqrt(np.linalg.eigvalsh(Pd.T @ Pd / len(xd))[-1])
    wstar = rng.standard_normal(d) / np.sqrt(d)
    f = P2c @ wstar                                            # PDE source  f = u*''
    ud = Pd @ wstar + 0.10 * rng.standard_normal(len(xd))      # NOISY sensor data
    S_pde = P2c.T @ P2c / len(xc); S_dat = Pd.T @ Pd / len(xd)
    Lpde = lambda w: 0.5 * np.mean((P2c @ w - f) ** 2)
    Ldat = lambda w: 0.5 * np.mean((Pd @ w - ud) ** 2)
    gpde = lambda w: S_pde @ w - P2c.T @ f / len(xc)
    gdat = lambda w, idx: Pd[idx].T @ (Pd[idx] @ w - ud[idx]) / len(idx)

    # a near-optimal-but-not-exact PDE head (perturbed within range(Sigma_PDE) -> genuine linear term)
    Vp = np.linalg.eigh(S_pde)[1][:, -40:]
    w = wstar + 0.3 * Vp @ rng.standard_normal(40)
    # ---- A: predict the PDE-loss jump caused by data-gradient updates ----
    pred_q, pred_full, meas = [], [], []
    for _ in range(200):
        idx = rng.choice(len(xd), rng.integers(10, len(xd)), replace=False)
        D = -rng.uniform(0.3, 3.0) * gdat(w, idx)             # a data-only update
        pred_q.append(0.5 * D @ S_pde @ D)                    # interference energy (quadratic only)
        pred_full.append(gpde(w) @ D + 0.5 * D @ S_pde @ D)   # full identity (Thm 1/2)
        meas.append(Lpde(w + D) - Lpde(w))
    rq = np.corrcoef(pred_q, meas)[0, 1]; rf = np.corrcoef(pred_full, meas)[0, 1]
    print(f"  A: 1/2 D^T Sigma_PDE D vs measured PDE-loss jump: r={rq:.4f} (energy only); "
          f"full identity r={rf:.5f} (max rel err {np.max(np.abs(np.array(pred_full)-meas)/(np.abs(meas)+1e-9)):.1e})")
    # ---- B: fit the noisy data WITHOUT violating the PDE (null-space projection) ----
    U = np.linalg.eigh(S_pde)[1][:, -25:]                     # physics subspace U_PDE
    res = {}
    for mode in ["naive (data only)", "weighted (data+PDE)", "projected (ours)"]:
        w2 = w.copy()
        for _ in range(500):
            g = gdat(w2, np.arange(len(xd)))
            if mode.startswith("weighted"): g = g + 1.0 * gpde(w2)
            if mode.startswith("projected"): g = g - U @ (U.T @ g)   # remove physics-active part
            w2 = w2 - 0.5 * g
        res[mode] = (Ldat(w2), Lpde(w2))
        print(f"  B {mode:22s}: L_data={res[mode][0]:.4f}   L_PDE={res[mode][1]:.4f}")
    print(f"  -> projection cuts the physics violation by "
          f"{res['naive (data only)'][1]/max(res['projected (ours)'][1],1e-9):.0f}x vs naive, no loss weight to tune.")
    return dict(rA_energy=float(rq), rA_full=float(rf),
                pred=np.array(pred_full), meas=np.array(meas))


# =====================================================================
#  C  Process drift: simulated dry-electrode extrusion sensor stream + fault detection
# =====================================================================
def C_extrusion():
    hdr("C  Process drift (dry-electrode extrusion sim): subspace-velocity fault detection")
    names = ["torque", "screw_rpm", "temp_z1", "temp_z2", "temp_z3", "pressure", "film_thick", "line_speed"]
    dim = len(names)
    def run(seed):
        rg = np.random.default_rng(seed)
        # 4 regimes (material batches / recalibrations); each = a correlated setpoint + covariance
        def regime():
            L = rg.standard_normal((dim, 4)) * 0.6           # low-rank correlation (shared drivers)
            mu = rg.uniform(-1, 1, dim)
            return mu, L
        regs = [regime() for _ in range(4)]
        bounds = sorted(rg.choice(range(20, 90), 3, replace=False))
        T = 110
        C = np.zeros((dim, dim)); Up = None; vel = []
        for t in range(T):
            k = sum(t >= bb for bb in bounds)
            mu, L = regs[k]
            drift = 0.02 * (t - (bounds[k-1] if k else 0))    # slow within-regime drift
            X = mu + drift + (rg.standard_normal((48, 4)) @ L.T) + 0.05 * rg.standard_normal((48, dim))
            C = 0.6 * C + X.T @ X / len(X)
            U = np.linalg.eigh(C)[1][:, -4:]
            vel.append(np.linalg.norm((U @ U.T) @ (Up @ Up.T) - (Up @ Up.T) @ (U @ U.T), "fro") if Up is not None else 0.0)
            Up = U
        vel = np.array(vel); thr = vel.mean() + 1.5 * vel.std()
        pk = [t for t in range(1, T - 1) if vel[t] > thr and vel[t] >= vel[t-1] and vel[t] >= vel[t+1]]
        hit = sum(any(abs(p - b) <= 3 for p in pk) for b in bounds)
        fp = sum(1 for p in pk if not any(abs(p - b) <= 3 for b in bounds))
        return hit / len(bounds), (hit / max(len(pk), 1))
    R = [run(s) for s in range(20)]
    rec = np.mean([r[0] for r in R]); prec = np.mean([r[1] for r in R])
    print(f"  {dim}-sensor extruder stream, 3 hidden batch-swaps/recalibrations, N=20 runs:")
    print(f"  fault-detection recall={rec:.2f}  precision={prec:.2f}  (unsupervised, from subspace velocity only)")
    print("  -> a drop-in unsupervised fault detector for continuous manufacturing (real logs needed to confirm).")
    return dict(recall=float(rec), precision=float(prec))


# =====================================================================
#  D  Sigma -> policy gradients: Fisher-metric IGFA gate retains an old RL environment
# =====================================================================
def D_rl_fisher():
    hdr("D  Sigma -> policy gradients: Fisher-metric gate protects an old environment")
    # Linear-Gaussian policy pi(a|s)=N(theta^T s, sig^2). For reward -(a-w^T s)^2 the EXPECTED
    # return is J(theta)=-(theta-w)^T F (theta-w) - sig^2 with F=E[s s^T] (= Fisher up to 1/sig^2):
    # policy-gradient interference is governed by the Fisher exactly as Sigma governs it in
    # supervised learning. (Full stochastic PPO/MuJoCo adds estimator noise -> separate Kaggle run.)
    ds = 14
    sh = np.linalg.qr(rng.standard_normal((ds, 3)))[0]        # 3 state dims SHARED by both envs (where they conflict)
    def env(seed):
        rg = np.random.default_rng(seed)
        uniq = np.linalg.qr(rg.standard_normal((ds, 2)))[0]
        feat = np.concatenate([sh, uniq], 1)                  # 5-dim excited subspace (3 shared + 2 unique)
        return feat @ feat.T, feat @ rg.standard_normal(5)    # (Fisher = E[s s^T], optimal policy)
    F1, w1 = env(1); F2, w2 = env(2)
    J = lambda th, F, w: float(-(th - w) @ F @ (th - w))      # expected return (<=0, higher=better)
    def learn(th0, mode, steps=1500, lr=0.06):
        th = th0.copy(); U = np.linalg.eigh(F1)[1][:, -5:]
        for _ in range(steps):
            g = -2 * F2 @ (th - w2)                           # expected policy gradient on env2 (ascent)
            if mode == "ewc": g = g - 3.0 * F1 @ (th - th0)   # Fisher-weighted anchor to env1 solution
            if mode == "gated": g = g - U @ (U.T @ g)         # route PG orthogonal to old-env Fisher subspace
            th = th + lr * g
        return th
    th1 = np.zeros(ds)                                        # train env1 to optimum first
    for _ in range(1500): th1 = th1 - 0.06 * (2 * F1 @ (th1 - w1))
    print(f"  after env1:  env1 return={J(th1,F1,w1):+.3f}  env2 return={J(th1,F2,w2):+.3f}")
    out = {}
    for mode in ["naive", "ewc", "gated"]:
        th2 = learn(th1, mode)
        out[mode] = (J(th2, F1, w1), J(th2, F2, w2))
        print(f"  after env2 [{mode:5s}]: env1={out[mode][0]:+.3f} (retention)  env2={out[mode][1]:+.3f} (new)")
    print("  -> the Fisher-subspace gate retains env1 (~0) while still solving env2, unlike naive PG.")
    return out


# =====================================================================
#  E  Meta-learn a feature map with a Tr(Sigma_A Sigma_B) penalty to shrink the floor
# =====================================================================
def E_meta_floor():
    hdr("E  Meta-learned feature map: Tr(Sigma_A Sigma_B) penalty drives task subspaces apart")
    din, d, K = 40, 24, 4
    Ms = [np.linalg.qr(rng.standard_normal((din, 6)))[0] for _ in range(K)]   # overlapping input supports
    Ms = [M @ M.T for M in Ms]                                # input covariances (rank 6, share dims)
    def overlaps(W):
        Sig = [W @ M @ W.T for M in Ms]
        Bs = [np.linalg.qr(np.linalg.eigh(S)[1][:, -6:])[0] for S in Sig]
        ov = [np.linalg.norm(Bs[i].T @ Bs[j], "fro") ** 2 / 6 for i in range(K) for j in range(i+1, K)]
        floor = sum(np.trace(Sig[i] @ Sig[j]) for i in range(K) for j in range(i+1, K))
        return float(np.mean(ov)), float(floor)
    W0 = np.linalg.qr(rng.standard_normal((d, din)).T)[0].T   # random (baseline) feature map
    ov0, fl0 = overlaps(W0)
    # optimise W to MINIMISE sum_{A<B} Tr(Sigma_A Sigma_B) while staying orthonormal (variance-preserving)
    W = W0.copy(); lr = 0.05
    for _ in range(400):
        Sig = [W @ M @ W.T for M in Ms]
        G = np.zeros_like(W)
        for i in range(K):
            for j in range(K):
                if i != j:
                    G += 2 * (Sig[i] @ W @ Ms[j] + Sig[j] @ W @ Ms[i])   # d/dW Tr(Sig_i Sig_j)
        W = W - lr * G / (np.linalg.norm(G) + 1e-9)
        W = np.linalg.qr(W.T)[0].T                            # keep orthonormal rows
    ov1, fl1 = overlaps(W)
    print(f"  baseline feature map:   mean inter-task overlap s={ov0:.3f}   floor proxy Tr={fl0:.3f}")
    print(f"  meta-penalised map:     mean inter-task overlap s={ov1:.3f}   floor proxy Tr={fl1:.3f}")
    print(f"  -> penalty cuts overlap {ov0/max(ov1,1e-6):.1f}x and the floor {fl0/max(fl1,1e-6):.1f}x: features "
          f"become near-disjoint, pushing IGFA toward the lossless regime (Cor. 1).")
    return dict(ov_base=ov0, ov_meta=ov1, floor_base=fl0, floor_meta=fl1)


# =====================================================================
#  Figure: three application panels for the SI (matches house figure style)
# =====================================================================
def make_figure(rA, rD, rE):
    import matplotlib
    matplotlib.use("Agg")
    try:
        import mpl_style  # shared publication style (DejaVu Sans)
    except Exception:
        pass
    import matplotlib.pyplot as plt
    BLUE, RED, GREEN, GREY = "#2b6cb0", "#c0392b", "#27875a", "#7f8c8d"
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.9))

    a = ax[0]                                                  # (a) PINN: predicted vs measured
    lim = max(rA["meas"].max(), rA["pred"].max()) * 1.05
    a.plot([0, lim], [0, lim], "--", color=GREY, lw=1.4, zorder=1)
    a.scatter(rA["pred"], rA["meas"], s=26, color=BLUE, alpha=0.75, zorder=3)
    a.set_xlabel(r"predicted  $\nabla L_{\rm PDE}^\top\Delta+\frac{1}{2}\Delta^\top\Sigma_{\rm PDE}\Delta$")
    a.set_ylabel("measured PDE-loss change")

    b = ax[1]                                                  # (b) RL: old-env regret per method
    modes = ["naive", "ewc", "gated"]
    labs = ["naive", "EWC", "gate (ours)"]
    cols = [RED, GREY, GREEN]
    old_reg = [max(-rD[m][0], 0.0) for m in modes]             # regret = -return, lower better
    new_reg = [max(-rD[m][1], 0.0) for m in modes]
    x = np.arange(3); w = 0.38
    b.bar(x - w / 2, old_reg, w, color=cols, edgecolor="white")
    b.bar(x + w / 2, new_reg, w, color=cols, alpha=0.45, edgecolor="white", hatch="//")
    for xi, (o, n) in enumerate(zip(old_reg, new_reg)):
        b.text(xi - w / 2, o + 0.15, f"{o:.1f}", ha="center", fontsize=9)
        b.text(xi + w / 2, n + 0.15, f"{n:.1f}", ha="center", fontsize=9)
    b.set_xticks(x, labs, fontsize=10)
    b.set_ylabel(r"regret $-J$  ($\downarrow$ better)")
    b.set_ylim(0, 12.6)
    import matplotlib.patches as mpatches
    b.legend(handles=[mpatches.Patch(fc="#555", label="old env (retention)"),
                      mpatches.Patch(fc="#555", alpha=0.45, hatch="//", label="new env (adaptation)")],
             fontsize=9)

    c = ax[2]                                                  # (c) meta-learned features shrink floor
    pairs = [("overlap $\\bar s$", rE["ov_base"], rE["ov_meta"]),
             (r"floor  $\mathrm{Tr}(\Sigma_A\Sigma_B)$", rE["floor_base"], rE["floor_meta"])]
    x = np.arange(2); w = 0.38
    c.bar(x - w / 2, [p[1] for p in pairs], w, color=GREY, edgecolor="white", label="random features")
    c.bar(x + w / 2, [p[2] for p in pairs], w, color=GREEN, edgecolor="white", label="meta-penalized")
    for xi, (_, base, meta) in enumerate(pairs):
        c.text(xi - w / 2, base * 1.25, f"{base:.2f}", ha="center", fontsize=9)
        c.text(xi + w / 2, meta * 1.25, f"{meta:.3g}", ha="center", fontsize=9)
    c.set_yscale("log"); c.set_ylim(4e-4, 30)
    c.set_xticks(x, [p[0] for p in pairs], fontsize=10)
    c.legend(fontsize=9)

    for axis in ax:
        axis.spines[["top", "right"]].set_visible(False)
    for tag, axis in zip("abc", ax):
        axis.text(-0.12, 1.08, f"({tag})", transform=axis.transAxes, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig("figures/fig_applications.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig_applications.pdf")


if __name__ == "__main__":
    results = {}
    for fn in [AB_pinn, C_extrusion, D_rl_fisher, E_meta_floor]:
        try: results[fn.__name__] = fn()
        except Exception as e:
            import traceback; print(f"{fn.__name__} FAILED: {e}"); traceback.print_exc()
    try:
        make_figure(results["AB_pinn"], results["D_rl_fisher"], results["E_meta_floor"])
    except Exception as e:
        import traceback; print(f"figure FAILED: {e}"); traceback.print_exc()
    print("\n" + "=" * 70 + "\nDONE.")
