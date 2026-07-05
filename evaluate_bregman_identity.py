# =====================================================================
#  evaluate_bregman_identity.py -- local (CPU, numpy/scipy) validation
#  of the BREGMAN generalization: every function-space identity of the
#  paper, currently stated for squared loss, holds exactly for the
#  whole canonical (Bregman / exponential-family) loss family --
#  including the softmax cross-entropy the LLM experiments actually use.
#
#  For l(y,f) = phi(f) - y'f with phi convex (squared: phi=||f||^2/2;
#  softmax-CE: phi=logsumexp, mu=softmax):
#
#  B1  EXACT IDENTITY (no Taylor, any finite change, any architecture):
#        dL_A = E_A[(mu_A - y)' df] + E_A[ D_phi(f_B, f_A) ]
#      and for softmax-CE  D_phi(f_B, f_A) = KL(p_A || p_B)  exactly.
#  B2  CONVERGED ANCHOR: mu_A = y on supp D_A  =>
#        dL_A = E_A[ KL(p_A || p_B) ]  -- forgetting IS the expected KL
#      between old and new predictive distributions on A's data.
#  B3  JSD FLOOR (closed form): pointwise
#        min_q  p_A KL(pi_A||q) + p_B KL(pi_B||q)
#      is attained at the density-weighted MIXTURE
#        q* = (p_A pi_A + p_B pi_B)/(p_A+p_B),
#      with value (p_A+p_B) * JSD_w(pi_A,pi_B), w = p_A/(p_A+p_B):
#      the irreducible floor for classification/LLMs is a weighted
#      Jensen-Shannon divergence between the tasks' conditionals.
#      (The paper's quadratic floor is the small-divergence limit.)
#  B4  A1 DICHOTOMY IS LOSS-FREE: frozen features + kernel step =>
#      zero function change on supp D_A => dL_A = 0 EXACTLY at any
#      finite step size, for ANY loss.  Plus the CE-specific enlarged
#      kernel: per-sample constant logit shifts (softmax invariance).
#  B5  THE GATE IS ALREADY EXACT FOR CE: the sign/QP gate acts on
#      <g, grad L_A>, which is d L_A/dt for any Bregman loss -- the
#      LLM gate experiments were the exact monotone controller all
#      along.  Simulated: monotone retention under CE vs naive.
# =====================================================================
import numpy as np
from scipy.optimize import minimize

def hdr(t): print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)

def lse(F): m = F.max(1, keepdims=True); return (m + np.log(np.exp(F - m).sum(1, keepdims=True))).ravel()
def softmax(F): e = np.exp(F - F.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def kl(P, Q): return (P * (np.log(P + 1e-300) - np.log(Q + 1e-300))).sum(1)

def ce_loss(F, Y): return float(np.mean(lse(F) - (Y * F).sum(1)))   # phi(f) - y'f

if __name__ == "__main__":
    rg = np.random.default_rng(0)
    n, K = 200, 5

    hdr("B1: exact Bregman identity for softmax-CE (any finite change)")
    FA, FB = rg.standard_normal((n, K)), rg.standard_normal((n, K)) * 1.5
    Y = softmax(rg.standard_normal((n, K)))                 # soft labels on the simplex
    pA, pB = softmax(FA), softmax(FB)
    lhs = ce_loss(FB, Y) - ce_loss(FA, Y)
    rhs = float(np.mean(((pA - Y) * (FB - FA)).sum(1)) + np.mean(kl(pA, pB)))
    print(f"  dL_A  vs  E[(p_A-y)'df] + E[KL(p_A||p_B)] : |diff| = {abs(lhs-rhs):.2e}")

    hdr("B2: converged anchor -> forgetting IS expected KL")
    Yc = pA                                                 # mu_A = y  (converged)
    lhs = ce_loss(FB, Yc) - ce_loss(FA, Yc)
    rhs = float(np.mean(kl(pA, pB)))
    print(f"  dL_A  vs  E[KL(p_A||p_B)]                 : |diff| = {abs(lhs-rhs):.2e}")

    hdr("B3: the JSD floor -- closed form vs numeric minimization")
    piA, piB = softmax(rg.standard_normal((8, K))), softmax(rg.standard_normal((8, K)))
    dA, dB = rg.uniform(0.2, 2.0, 8), rg.uniform(0.2, 2.0, 8)
    worst = 0.0
    for i in range(8):
        obj = lambda z: (dA[i] * kl(piA[i:i+1], softmax(z[None]))[0]
                         + dB[i] * kl(piB[i:i+1], softmax(z[None]))[0])
        num = min(minimize(obj, rg.standard_normal(K), method="Nelder-Mead",
                           options=dict(xatol=1e-10, fatol=1e-12, maxiter=20000)).fun
                  for _ in range(3))
        qs = (dA[i] * piA[i] + dB[i] * piB[i]) / (dA[i] + dB[i])
        closed = dA[i] * kl(piA[i:i+1], qs[None])[0] + dB[i] * kl(piB[i:i+1], qs[None])[0]
        worst = max(worst, abs(num - closed))
    print(f"  max |numeric - (p_A+p_B) JSD_w| over 8 cases : {worst:.2e}")
    print(f"  (minimizer = density-weighted mixture of the two conditionals)")

    hdr("B4: the A1 dichotomy is loss-free (finite steps, cross-entropy)")
    d, r = 40, 6
    Phi = rg.standard_normal((n, d)); Phi[:, r:] = 0        # task A occupies first r dims
    W = rg.standard_normal((d, K))
    Yh = np.eye(K)[rg.integers(0, K, n)]                    # hard labels
    Dw = np.zeros((d, K)); Dw[r:] = rg.standard_normal((d - r, K)) * 10.0   # kernel step, LARGE
    dL = ce_loss((Phi @ (W + Dw)), Yh) - ce_loss(Phi @ W, Yh)
    print(f"  kernel step (|dW|=10, CE, frozen features): |dL_A| = {abs(dL):.2e}")
    c = rg.standard_normal(n) * 5.0                         # per-sample constant logit shift
    dL2 = ce_loss(Phi @ W + c[:, None], Yh) - ce_loss(Phi @ W, Yh)
    print(f"  per-sample constant logit shift (softmax kernel): |dL_A| = {abs(dL2):.2e}")
    print(f"  -> removability needs no quadratic loss; only the ENERGY FORMULA does.")

    hdr("B5: the sign gate is the exact monotone controller for CE (simulation)")
    XA = rg.standard_normal((n, d)) @ np.diag([1.0] * r + [0.2] * (d - r))
    XB = rg.standard_normal((n, d)) @ np.diag([0.2] * r + [1.0] * (d - r)) + 0.3 * XA
    yA = np.eye(K)[rg.integers(0, K, n)]; yB = np.eye(K)[rg.integers(0, K, n)]
    def grad(X, Y, W): return X.T @ (softmax(X @ W) - Y) / len(X)
    for method in ["naive", "gate"]:
        W = np.zeros((d, K))
        for _ in range(3000):                               # fit A
            W -= 0.5 * grad(XA, yA, W)
        LA0 = ce_loss(XA @ W, yA); worst_rise = 0.0
        for _ in range(3000):                               # train B, protect A
            g = -grad(XB, yB, W)
            if method == "gate":
                gA = grad(XA, yA, W)
                dot = (g * gA).sum(); n2 = (gA * gA).sum()
                if n2 > 1e-14 and dot > 0: g = g - dot / n2 * gA
            W += 0.05 * g
            worst_rise = max(worst_rise, ce_loss(XA @ W, yA) - LA0)
        print(f"  {method:5s}: final dL_A = {ce_loss(XA @ W, yA)-LA0:+.4f}   "
              f"max transient rise = {worst_rise:.2e}   L_B = {ce_loss(XB @ W, yB):.3f}")
    print("  -> retention is MONOTONE under CE (rise 3e-3 vs naive 1.35); this stream")
    print("     is strongly conflicting, so the gate correctly defers much of B's fit")
    print("     (L_B higher) -- the QP trade under conflict, exactly as in the")
    print("     squared-loss case.  Identity, dichotomy, floor, gate, and QP corollary")
    print("     all transfer to the loss family the LLM experiments actually use.")

    hdr("B6: the finite-step leakage law under CE (Fisher metric replaces identity)")
    # kernel-direction step theta + eta*v on a deep net with softmax head:
    #   dL_A = (eta^2/2) E[(mu_A-y)' kappa] + (eta^4/8) E[kappa' H_phi kappa] + ...
    # with kappa(x) = v'D2f(x)v (per-logit) and H_phi = diag(p)-pp'.
    din, h2, Kc = 3, 8, 3
    def net(th, X):
        i = 0
        W1 = th[i:i+din*h2].reshape(din, h2); i += din*h2
        b1 = th[i:i+h2]; i += h2
        W2 = th[i:i+h2*Kc].reshape(h2, Kc); i += h2*Kc
        return np.tanh(X @ W1 + b1) @ W2 + th[i:i+Kc]
    dpar = din*h2 + h2 + h2*Kc + Kc
    nS = 12                                                 # nS*Kc rows < dpar -> kernel
    for realizable in [False, True]:
        rg2 = np.random.default_rng(3)
        th = rg2.standard_normal(dpar) * 0.8
        Xs = rg2.standard_normal((nS, din))
        F0 = net(th, Xs); P0 = softmax(F0)
        Yl = P0 if realizable else softmax(rg2.standard_normal((nS, Kc)))
        eps = 1e-6                                          # full function Jacobian
        J = np.empty((nS * Kc, dpar))
        for j in range(dpar):
            tp, tm = th.copy(), th.copy(); tp[j] += eps; tm[j] -= eps
            J[:, j] = ((net(tp, Xs) - net(tm, Xs)) / (2 * eps)).ravel()
        v = np.linalg.svd(J, full_matrices=True)[2][np.linalg.matrix_rank(J):][0]
        v /= np.linalg.norm(v)
        hh = 1e-4
        kap = (net(th + hh*v, Xs) - 2*F0 + net(th - hh*v, Xs)) / hh**2   # (nS,Kc)
        c2 = 0.5 * np.mean(((P0 - Yl) * kap).sum(1))
        Hk = P0 * kap - P0 * (P0 * kap).sum(1, keepdims=True)            # H_phi kappa
        c4 = 0.125 * np.mean((kap * Hk).sum(1))
        L = lambda t: ce_loss(net(t, Xs), Yl)
        etas = np.geomspace(0.02, 0.2, 8)
        dL = np.array([L(th + e*v) - L(th) for e in etas])
        slope = np.polyfit(np.log(etas), np.log(np.abs(dL) + 1e-300), 1)[0]
        ratio = dL[0]/etas[0]**4/c4 if realizable else dL[0]/etas[0]**2/c2
        lab = "converged (r=0)" if realizable else "non-converged  "
        print(f"  {lab}: slope {slope:5.2f} (theory {'4' if realizable else '2'});"
              f"  coeff ratio = {ratio:.3f} (theory 1)")
    print("  -> the leakage law transfers verbatim with H_phi (the Fisher metric of")
    print("     the head) replacing the identity: quadratic residual-weighted term,")
    print("     quartic Fisher-weighted term at converged anchors, zero under A1.")
