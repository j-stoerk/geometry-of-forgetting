# =====================================================================
#  evaluate_replay_policy.py -- replay as a BUDGETED policy driven by
#  geometry (roadmap item 4), sharpened by the information floor:
#
#  PRINCIPLE.  Full replay (joint retraining) lands every task exactly
#  AT its floor share -- never below (the floor is the single-predictor
#  limit; rehearsal cannot beat it).  Therefore
#      replay value of task t  =  L_t  -  floor_t   (recoverable excess),
#  and the optimal fixed-budget allocation is the fractional knapsack:
#  greedy in (L_t - floor_t)/cost_t.
#
#  Two regimes where the floor-proportional heuristic misallocates:
#    - a task AT its floor: zero replay value however large the floor
#      (conflict already balanced -- rehearsal buys nothing);
#    - a ZERO-floor task damaged by capacity/dynamics coupling: full
#      replay value despite floor = 0 (all damage recoverable).
#  The heuristic works when damage scales with conflict -- the regime
#  of the paper's measured 28% win -- and fails outside it.
#
#  P1: full replay attains the floor exactly (per-task decomposition).
#  P2: budgeted allocation on a 3-regime protected set: greedy-by-
#      recoverable-excess == enumerated optimum; floor-proportional and
#      forgetting-proportional both waste budget on the at-floor task.
# =====================================================================
import numpy as np
from itertools import product

def hdr(t): print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)
def sm(F): e = np.exp(F - F.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
def xlogx(p): return np.where(p > 0, p * np.log(p + 1e-300), 0.0)

rg = np.random.default_rng(0)
K, NX = 4, 30

def ce_excess(q, pi, dens):
    """Excess risk of predictor q over task (dens, pi)'s Bayes risk."""
    return float((dens[:, None] * (xlogx(pi) - pi * np.log(q + 1e-300))).sum())

if __name__ == "__main__":
    hdr("P1: full replay (joint optimum) attains each task's floor share EXACTLY")
    dens = rg.uniform(0.1, 1.0, (3, NX)); dens /= dens.sum(1, keepdims=True)
    pi = sm(rg.standard_normal((3, NX, K)))
    mix = np.einsum("tx,txk->xk", dens / dens.sum(0), pi)   # joint optimum
    for t in range(3):
        floor_t = ce_excess(mix, pi[t], dens[t])
        # train jointly to convergence (exact expected-gradient descent)
        Z = np.zeros((NX, K))
        for _ in range(30000):
            Z -= 2.0 * np.einsum("tx,txk->xk", dens, sm(Z)[None] - pi)
        trained_t = ce_excess(sm(Z), pi[t], dens[t])
        print(f"  task {t}: floor share {floor_t:.10f}   after full replay {trained_t:.10f}"
              f"   |diff| = {abs(floor_t-trained_t):.2e}")
    print("  -> rehearsal converges TO the floor, never below it: replay's value is")
    print("     the excess ABOVE the floor, not the floor itself.")

    hdr("P2: budgeted replay -- three regimes, five policies (20 random instances)")
    # protected task t: current excess L_t, floor f_t, unit cost c_t; spending
    # b in [0,c_t] recovers (b/c_t)*(L_t - f_t)  (linear response, saturates at floor)
    def retained_after(L, f, c, alloc):
        return sum(Li - min(b / ci, 1.0) * (Li - fi) for Li, fi, ci, b in zip(L, f, c, alloc))
    def alloc_prop(w, c, B):
        w = np.maximum(np.asarray(w, float), 0)
        a = B * w / w.sum() if w.sum() > 0 else np.full(len(w), B / len(w))
        return np.minimum(a, c)                             # cannot overspend a task
    def alloc_greedy(dens_val, c, B):
        a = np.zeros(len(c)); left = B
        for i in np.argsort(-np.asarray(dens_val)):
            take = min(c[i], left); a[i] = take; left -= take
            if left <= 0: break
        return a
    wins, results = 0, {p: [] for p in
        ["uniform", "prop-floor (paper)", "prop-forgetting", "greedy-recoverable", "optimum"]}
    for inst in range(20):
        r = np.random.default_rng(inst)
        #        at-floor task      zero-floor damaged     mixed
        f = [r.uniform(1.0, 2.0),  0.0,                   r.uniform(0.2, 0.6)]
        L = [f[0],                 r.uniform(0.8, 1.5),   f[2] + r.uniform(0.5, 1.0)]
        c = [r.uniform(0.5, 1.5) for _ in range(3)]
        B = 0.6 * sum(c)                                    # budget covers ~60%
        rec_density = [(Li - fi) / ci for Li, fi, ci in zip(L, f, c)]
        pols = {
            "uniform":            alloc_prop([1, 1, 1], c, B),
            "prop-floor (paper)": alloc_prop(f, c, B),
            "prop-forgetting":    alloc_prop(L, c, B),
            "greedy-recoverable": alloc_greedy(rec_density, c, B),
        }
        # enumerated optimum on a fine grid (3 tasks, simplex of spends)
        best = np.inf
        for a0 in np.linspace(0, min(c[0], B), 25):
            for a1 in np.linspace(0, min(c[1], B - a0), 25):
                a2 = min(c[2], B - a0 - a1)
                best = min(best, retained_after(L, f, c, [a0, a1, a2]))
        for p, a in pols.items():
            results[p].append(retained_after(L, f, c, a))
        results["optimum"].append(best)
        wins += results["greedy-recoverable"][-1] <= best + 1e-9
    print(f"  {'policy':22s} {'mean retained excess (lower=better)':>36s}")
    for p, v in results.items():
        print(f"  {p:22s} {np.mean(v):36.3f}")
    print(f"  greedy-recoverable matches the enumerated optimum in {wins}/20 instances")
    print("\n  -> the at-floor task makes floor-proportional AND forgetting-proportional")
    print("     waste budget on unrecoverable loss; the zero-floor damaged task is")
    print("     invisible to the floor heuristic yet fully recoverable.  The optimal")
    print("     policy is greedy in (excess above floor)/cost -- all three quantities")
    print("     measurable pre-flight from the ledger.")
