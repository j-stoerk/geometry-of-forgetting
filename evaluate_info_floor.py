# =====================================================================
#  evaluate_info_floor.py -- local validation of the INFORMATION FLOOR:
#  under cross-entropy, the T-task irreducible floor is EXACTLY a
#  conditional mutual information -- the paper's mutual-information
#  sandwich (Thm mi, quadratic loss: bounds) closes to an identity.
#
#  Setting: task t has input density p_t(x) and conditional pi_t(y|x);
#  one shared predictor q(y|x) serves all T tasks (uniform task prior).
#
#  THEOREM (information floor).  The excess joint risk of the best
#  single predictor over the task oracle is
#      D*_T  =  sum_x sum_t p_t(x) KL(pi_t || pibar_x)
#            =  T * I(T ; Y | X),
#  attained at the density-weighted mixture pibar = sum_t w_t pi_t,
#  w_t = p_t / sum_s p_s.  Equivalently  D*_T/T = H(Y|X) - H(Y|X,T):
#  the floor is exactly the label-entropy cost of hiding the task.
#
#  Corollaries checked here:
#   C1  exactness: closed form == T * I(T;Y|X)  (machine precision)
#   C2  the mixture is the true minimizer (vs numeric minimization)
#   C3  dichotomy = zero-CMI: conditionals agreeing on overlapping
#       support  =>  floor 0 (removability as an information statement)
#   C4  T=2 reduces to the weighted-JSD floor (ledger's jsd_floor)
#   C5  a trained shared softmax head converges to the floor from above
# =====================================================================
import numpy as np
from scipy.optimize import minimize
from interference_ledger import jsd_floor

def hdr(t): print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)
def sm(F): e = np.exp(F - F.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
def xlogx(p): return np.where(p > 0, p * np.log(p + 1e-300), 0.0)
def H(p): return -xlogx(p).sum(-1)

rg = np.random.default_rng(0)
T, K, NX = 4, 5, 40                                        # tasks, labels, input atoms

def make_tasks(agree=False):
    dens = rg.uniform(0.1, 1.0, (T, NX))
    dens /= dens.sum(1, keepdims=True)                     # each task: unit input mass
    if agree:
        pi = np.tile(sm(rg.standard_normal((1, NX, K))), (T, 1, 1))
    else:
        pi = sm(rg.standard_normal((T, NX, K)))
    return dens, pi

def closed_floor(dens, pi):
    P = dens.sum(0)                                        # total mass per atom
    w = dens / P                                           # task weights per atom (T,NX)
    mix = np.einsum("tx,txk->xk", w, pi)
    kl = (xlogx(pi) - pi * np.log(mix + 1e-300)[None]).sum(-1)   # KL(pi_t || mix) per (t,x)
    return float((dens * kl).sum()), mix

def cmi(dens, pi):
    """I(T;Y|X) under: T~Unif, X|T~dens_t, Y|X,T~pi_t."""
    pX = dens.sum(0) / T                                   # p(x)
    w = dens / dens.sum(0)                                 # p(t|x)
    pY_X = np.einsum("tx,txk->xk", w, pi)                  # p(y|x)
    HY_X = (pX * H(pY_X)).sum()
    HY_XT = (pX[None] * w * H(pi)).sum()
    return float(HY_X - HY_XT)

if __name__ == "__main__":
    hdr("C1: the floor IS T * I(T;Y|X)  (exact identity)")
    dens, pi = make_tasks()
    fl, mix = closed_floor(dens, pi)
    print(f"  closed-form floor = {fl:.12f}")
    print(f"  T * I(T;Y|X)      = {T * cmi(dens, pi):.12f}")
    print(f"  |difference|      = {abs(fl - T * cmi(dens, pi)):.2e}")

    hdr("C2: the density-weighted mixture is the minimizer (numeric check)")
    worst = 0.0
    for x in rg.choice(NX, 6, replace=False):
        obj = lambda z: float(sum(dens[t, x] * (xlogx(pi[t, x]) - pi[t, x]
                              * np.log(sm(z[None])[0] + 1e-300)).sum() for t in range(T)))
        num = min(minimize(obj, rg.standard_normal(K), method="Nelder-Mead",
                           options=dict(xatol=1e-10, fatol=1e-13, maxiter=30000)).fun
                  for _ in range(3))
        cl = float(sum(dens[t, x] * (xlogx(pi[t, x]) - pi[t, x]
                       * np.log(mix[x] + 1e-300)).sum() for t in range(T)))
        worst = max(worst, abs(num - cl))
    print(f"  max |numeric - closed| over 6 atoms : {worst:.2e}")

    hdr("C3: removability = zero conditional mutual information")
    dens2, pi2 = make_tasks(agree=True)                    # same conditionals, any densities
    fl2, _ = closed_floor(dens2, pi2)
    print(f"  agreeing conditionals: floor = {fl2:.2e}   I(T;Y|X) = {cmi(dens2, pi2):.2e}")

    hdr("C4: T=2 reduces to the ledger's weighted-JSD floor")
    dens4, pi4 = make_tasks()
    d2, p2 = dens4[:2], pi4[:2]
    fla, _ = closed_floor(d2, p2)
    flb = jsd_floor(p2[0], p2[1], d2[0], d2[1]) * NX       # ledger averages over atoms
    print(f"  closed T=2 floor = {fla:.10f}   ledger jsd_floor = {flb:.10f}"
          f"   |diff| = {abs(fla-flb):.2e}")

    hdr("C5: a trained shared head converges to the floor from above")
    Z = rg.standard_normal((NX, K)) * 0.01                 # per-atom logits (full capacity)
    dens, pi = make_tasks()
    bayes = (dens * H(pi)).sum()
    for it in range(20000):
        q = sm(Z)
        grad = np.einsum("tx,txk->xk", dens, q[None] - pi)  # d(total CE)/d logits
        Z -= 2.0 * grad
    risk = -(dens[:, :, None] * pi * np.log(sm(Z) + 1e-300)[None]).sum()
    fl, _ = closed_floor(dens, pi)
    print(f"  trained excess risk = {risk - bayes:.10f}   floor = {fl:.10f}"
          f"   gap = {risk - bayes - fl:.2e}  (>=0, ->0)")
    print("\n  -> the T-task cross-entropy floor equals T * I(task; label | input):")
    print("     the mutual-information SANDWICH of the quadratic theorem closes to an")
    print("     IDENTITY under the loss LLMs train.  Removable <=> Y independent of")
    print("     task given input.  Estimable from micro-caches: per-domain vs mixture")
    print("     log-likelihoods.")
