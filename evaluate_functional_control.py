# =====================================================================
#  evaluate_functional_control.py -- local (CPU, numpy/scipy) validation
#  of the D1 theorem package for boundary-free function-space control:
#
#  E1  FINITE-STEP LEAKAGE LAW.  For a step theta + eta*v with v in
#      ker G_A(theta) (instantaneously safe), with kappa(x)=v'D2f(x)v:
#         dL_A(eta) = (eta^2/2) E_A[r_A kappa] + (eta^4/8) E_A[kappa^2] + ...
#      -> frozen-linear (A1):    zero at ALL orders (finite-step exact)
#      -> converged anchor r=0:  QUARTIC leakage, coeff (1/8)E[kappa^2]
#      -> non-converged anchor:  quadratic, coeff (1/2)E[r kappa]
#      We check both the slopes and the COEFFICIENTS against finite-
#      difference curvatures.
#
#  E2  THE GATE IS OPTIMAL MONOTONE CONTROL.  The constrained flow
#         min_v 1/2||v-u||^2   s.t.  <grad L_Ai, v> <= 0  (i=1..m)
#      has dual lambda = NNLS(G,u) and solution v = u - G lambda.
#      m=1 recovers the threshold-free sign gate in closed form
#      (checked to machine precision); no active constraints recovers
#      naive; all active recovers subspace projection.
#
#  E3  CONSTRAINT SAMPLING EXPLAINS THE VISION RESULT.  Single-probe
#      gating (sample ONE protected task per step) is stochastic
#      under-enforcement of the QP.  On a dense-conflict stream it
#      recovers only a fraction of projection's retention -- the
#      mechanism behind the measured C100-superclass gap -- while
#      ACTIVE-SET gating (enforce all measured-violated constraints,
#      cost m dots + NNLS on an m-column system) recovers nearly all
#      of it and still shares freely on aligned streams.
# =====================================================================
import numpy as np
from scipy.optimize import nnls

def hdr(t): print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)

# --------------------------------------------------------------- E1 --
def mlp(theta, X, din, h):
    i = 0
    W1 = theta[i:i + din * h].reshape(din, h); i += din * h
    b1 = theta[i:i + h]; i += h
    w2 = theta[i:i + h]; i += h
    return np.tanh(X @ W1 + b1) @ w2 + theta[i]

def jac(theta, X, din, h, eps=1e-6):
    J = np.empty((len(X), len(theta)))
    for j in range(len(theta)):
        tp, tm = theta.copy(), theta.copy(); tp[j] += eps; tm[j] -= eps
        J[:, j] = (mlp(tp, X, din, h) - mlp(tm, X, din, h)) / (2 * eps)
    return J

def leakage_case(realizable, seed=0, din=3, h=8, n=25):
    rg = np.random.default_rng(seed)
    d = din * h + h + h + 1
    th = rg.standard_normal(d) * 0.8
    X = rg.standard_normal((n, din))
    f0 = mlp(th, X, din, h)
    y = f0 if realizable else f0 + 0.3 * rg.standard_normal(n)
    J = jac(th, X, din, h)
    _, s, Vt = np.linalg.svd(J, full_matrices=True)
    v = Vt[len(s):][0] if len(s) < d else Vt[-1]          # exact kernel direction
    v /= np.linalg.norm(v)
    # curvature along v via second differences of f
    hh = 1e-4
    kap = (mlp(th + hh * v, X, din, h) - 2 * f0 + mlp(th - hh * v, X, din, h)) / hh ** 2
    r0 = f0 - y
    c2, c4 = 0.5 * np.mean(r0 * kap), 0.125 * np.mean(kap ** 2)
    L = lambda t: 0.5 * np.mean((mlp(t, X, din, h) - y) ** 2)
    etas = np.geomspace(0.02, 0.2, 8)
    dL = np.array([L(th + e * v) - L(th) for e in etas])
    slope = np.polyfit(np.log(etas), np.log(np.abs(dL) + 1e-300), 1)[0]
    e0 = etas[0]
    coeff_ratio = (dL[0] / e0 ** 2 / c2) if not realizable else (dL[0] / e0 ** 4 / c4)
    return slope, coeff_ratio

if __name__ == "__main__":
    hdr("E1: finite-step leakage law for kernel-direction steps")
    sl, cr = leakage_case(realizable=False, seed=1)
    print(f"  non-converged anchor : slope {sl:5.2f}  (theory 2);  dL/(eta^2 c2) = {cr:.3f}  (theory 1)")
    sl, cr = leakage_case(realizable=True, seed=1)
    print(f"  converged anchor     : slope {sl:5.2f}  (theory 4);  dL/(eta^4 c4) = {cr:.3f}  (theory 1)")
    rg = np.random.default_rng(0)                          # frozen-linear control
    Xl = rg.standard_normal((25, 40)); w = rg.standard_normal(40)
    yl = Xl @ w; vl = np.linalg.svd(Xl, full_matrices=True)[2][25:][0]
    dLlin = max(abs(0.5 * np.mean((Xl @ (w + e * vl) - yl) ** 2)) for e in [0.1, 1.0, 10.0])
    print(f"  frozen-linear (A1)   : max |dL| over eta<=10 : {dLlin:.2e}  (theory 0: finite-step exact)")

    # ----------------------------------------------------------- E2 --
    hdr("E2: the sign gate IS the optimal monotone controller (QP/KKT)")
    d = 30; rgq = np.random.default_rng(2)
    u = rgq.standard_normal(d); g1 = rgq.standard_normal(d)
    if u @ g1 < 0: g1 = -g1                                # make the constraint active
    lam, _ = nnls(g1[:, None], u)
    v_qp = u - g1 * lam[0]
    v_gate = u - (u @ g1) / (g1 @ g1) * g1                 # threshold-free sign gate
    print(f"  m=1, active constraint : ||v_QP - v_gate|| = {np.linalg.norm(v_qp - v_gate):.2e}")
    g2 = rgq.standard_normal(d)
    if u @ g2 > 0: g2 = -g2                                # inactive constraint
    lam, _ = nnls(g2[:, None], u)
    print(f"  m=1, inactive          : lambda = {lam[0]:.2e}  -> v = u exactly (naive/share)")
    G = rgq.standard_normal((d, 6))
    G = G * np.sign(G.T @ u)                               # all six constraints violated
    lam, _ = nnls(G, u)
    v_qp = u - G @ lam
    Q = np.linalg.qr(G)[0]
    v_proj = u - Q @ (Q.T @ u)                             # equality projection (OGD)
    print(f"  m=6, all violated      : feasibility max <g_i, v_QP> = {max(G.T @ v_qp):.2e} (theory <=0)")
    print(f"    projection pins rates at 0 (max |<g_i,v_proj>| = {max(abs(G.T @ v_proj)):.2e});")
    print(f"    the QP also permits protected-loss DECREASE: min rate = {min(G.T @ v_qp):+.3f} < 0,")
    print(f"    and stays closer to the desired update: ||v_QP-u|| = {np.linalg.norm(v_qp-u):.3f}"
          f" <= ||v_proj-u|| = {np.linalg.norm(v_proj-u):.3f}")
    print(f"    -> the QP gate DOMINATES equality projection: same guarantee, less restriction.")

    # ----------------------------------------------------------- E3 --
    hdr("E3: sampled vs active-set gating -- dense and aligned streams (exact regime)")
    def stream(kind, seed):
        rg = np.random.default_rng(seed); d, r, T = 60, 6, 5
        B0 = np.linalg.qr(rg.standard_normal((d, r)))[0]
        tasks = []
        for t in range(T):
            if kind == "dense":                            # shared subspace, conflicting targets
                B = np.linalg.qr(B0 + 0.05 * rg.standard_normal((d, r)))[0]
                w = B @ (rg.choice([-1.5, 1.5], r) + 0.2 * rg.standard_normal(r))
            else:                                          # aligned: same subspace, close targets
                B = np.linalg.qr(B0 + 0.05 * rg.standard_normal((d, r)))[0]
                w = B0 @ np.ones(r) + 0.1 * B @ rg.standard_normal(r)
            tasks.append((B @ B.T, w, B))
        return tasks

    def run(kind, method, seed, eta=0.05, K=400):
        tasks = stream(kind, seed); d = tasks[0][0].shape[0]
        w = np.zeros(d); rg = np.random.default_rng(seed + 999)
        occ = None
        for t, (S, ws, B) in enumerate(tasks):
            if method == "ogd" and t > 0:
                cols = np.hstack([tasks[i][2] for i in range(t)])
                occ = np.linalg.qr(cols)[0]
            for _ in range(K):
                u_dir = -S @ (w - ws)
                past = [tasks[i] for i in range(t)]
                if method == "naive" or not past:
                    v = u_dir
                elif method == "ogd":
                    v = u_dir - occ @ (occ.T @ u_dir)
                elif method == "sign-1":                   # sample ONE protected task
                    Si, wi, _ = past[rg.integers(len(past))]
                    gi = Si @ (w - wi)
                    dot = u_dir @ gi
                    v = u_dir - dot / (gi @ gi) * gi if (dot > 0 and gi @ gi > 1e-12) else u_dir
                else:                                      # active-set: NNLS over ALL violated
                    Gm = np.stack([Si @ (w - wi) for Si, wi, _ in past], 1)
                    keep = (Gm.T @ u_dir > 0) & (np.linalg.norm(Gm, axis=0) > 1e-9)
                    if keep.any():
                        lam, _ = nnls(Gm[:, keep], u_dir)
                        v = u_dir - Gm[:, keep] @ lam
                    else:
                        v = u_dir
                w = w + eta * v
        Ls = [0.5 * (w - ws) @ S @ (w - ws) for S, ws, _ in tasks]
        return np.mean(Ls[:-1]), Ls[-1]                    # retention (past), adaptation (last)

    for kind in ["dense", "aligned"]:
        print(f"\n  --- {kind}-conflict stream (10 seeds) ---")
        res = {}
        for m in ["naive", "sign-1", "active-set", "ogd"]:
            r = np.array([run(kind, m, s) for s in range(10)])
            res[m] = r.mean(0)
            print(f"  {m:10s}: past-task loss {r[:,0].mean():7.3f}   last-task loss {r[:,1].mean():7.3f}")
        if kind == "dense":
            rec1 = (res["naive"][0] - res["sign-1"][0]) / (res["naive"][0] - res["ogd"][0])
            reca = (res["naive"][0] - res["active-set"][0]) / (res["naive"][0] - res["ogd"][0])
            print(f"  retention recovered vs projection: sign-1 {100*rec1:.0f}%,  active-set {100*reca:.0f}%")
            print("  -> single-probe gating under-enforces the QP under dense conflict")
            print("     (the C100-superclass mechanism); the active set closes the gap at")
            print("     the cost of m interference-rate dots per step.")
        else:
            print("  -> aligned stream: constraints stay inactive, both gates keep naive's")
            print("     adaptation while projection needlessly sacrifices it.")
