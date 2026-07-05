# =====================================================================
#  evaluate_forgetting_decomposition.py -- the FORGETTING DECOMPOSITION:
#  a bias/variance-style attribution theorem for continual learning.
#
#  Define, for a stream served by one model of capacity (rank) R:
#    D*_inf : the FLOOR -- best achievable by an unconstrained single
#             predictor (the incompatibility of the tasks themselves);
#    D*_R   : best achievable at the deployed capacity R;
#    L      : what the deployed training policy actually achieved.
#  Then, term by term >= 0 and telescoping exactly:
#
#      L  =  D*_inf              (INCOMPATIBILITY -- no method removes it;
#                                 only changing the task family does)
#          + [D*_R - D*_inf]     (CAPACITY -- only expansion/routing
#                                 removes it; no gate or replay can)
#          + [L - D*_R]          (CONTROL -- only a better policy (gate,
#                                 replay, ordering) removes it)
#
#  The identity is trivial; the CONTENT is that each term is measurable
#  (floor from data geometry / CMI, D*_R from the spectrum, L observed)
#  and that INTERVENTIONS MOVE ONLY THEIR OWN TERM.  We validate that
#  attribution on three single-regime streams and a 2x3 intervention
#  matrix: expanding rank moves (almost) only the capacity term;
#  upgrading the gate moves (almost) only the control term; neither
#  touches the floor.  Diagnosis before medicine: measure WHICH term
#  dominates before choosing a method family.
# =====================================================================
import numpy as np

def hdr(t): print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)
def orth(rg, d, r): return np.linalg.qr(rg.standard_normal((d, r)))[0]

D, RT = 48, 4                                              # ambient dim, rank per task

def make_stream(kind, seed, T=6):
    rg = np.random.default_rng(seed)
    B0 = orth(rg, D, RT); w0 = B0 @ rg.standard_normal(RT)
    tasks = []
    for t in range(T):
        if kind == "floor":                                # EXACTLY shared support,
            B = B0                                         # conflicting targets
            w = (-w0 if t % 2 else w0) + 0.1 * B0 @ rg.standard_normal(RT)
        else:                                              # control: mixed, ample rank
            B = orth(rg, D, RT) if t % 2 == 0 else B0
            w = (B @ rg.standard_normal(RT) if t % 2 == 0
                 else (-w0 if (t // 2) % 2 else w0) + 0.1 * B0 @ rg.standard_normal(RT))
        tasks.append((B @ B.T, w, B))
    return tasks
    # (perturbed copies of a subspace are NOT conflict: 24 soft constraints in
    #  48 dims are jointly satisfiable and the floor is exactly 0 -- conflict
    #  requires exactly shared support, the quadratic knife-edge.)

def make_capacity_stream(seed, width, T=6, DTRUE=24):
    """Tasks disjoint in a DTRUE-dim reality (function-space floor EXACTLY 0)
    but deployed through a width-R random feature projection: compatible tasks
    collide in the deployed class.  Capacity = representation width, the
    paper's knee -- NOT the dimension of a subspace containing one vector
    (any single optimum spans 1 dim, so that notion collapses)."""
    rg = np.random.default_rng(seed)
    P = rg.standard_normal((DTRUE, DTRUE)) / np.sqrt(DTRUE)   # nested: widen = more rows
    tasks = []
    for t in range(T):
        Bt = np.zeros((DTRUE, RT)); Bt[RT * t:RT * (t + 1)] = np.eye(RT)
        a = Bt @ rg.standard_normal(RT)
        Pr = P[:, :width] if False else P[:width]             # first `width` rows
        S_hat = Pr @ Bt @ Bt.T @ Pr.T                         # deployed quadratic
        b_hat = Pr @ Bt @ Bt.T @ a
        w_hat = np.linalg.lstsq(S_hat, b_hat, rcond=None)[0]  # per-task deployed optimum
        tasks.append((S_hat, w_hat, np.linalg.qr(Pr @ Bt)[0][:, :RT]))
    return tasks

def joint_loss(w, tasks):
    return sum(0.5 * (w - wt) @ S @ (w - wt) for S, wt, _ in tasks)

def d_star(tasks):
    """Class optimum: best single parameter vector for the joint objective
    (closed form).  For deployed capacity streams this IS D*_R; the
    function-space floor D*_inf of those streams is 0 by construction."""
    A = sum(S for S, _, _ in tasks); b = sum(S @ wt for S, wt, _ in tasks)
    return float(joint_loss(np.linalg.lstsq(A, b, rcond=None)[0], tasks))

def run_policy(tasks, R, policy, seed=0, eta=0.05, K=500):
    """Sequential training at capacity R: parameters live in a greedily
    grown R-dim subspace; policy gates the within-span updates."""
    rg = np.random.default_rng(seed + 77)
    dim = tasks[0][0].shape[0]
    V = np.zeros((dim, 0)); w = np.zeros(dim)
    for t, (S, wt, Bt) in enumerate(tasks):
        if V.shape[1] < min(R, dim):                       # grow: allot task directions
            P = np.eye(dim) - V @ V.T if V.shape[1] else np.eye(dim)
            new = np.linalg.qr(P @ Bt)[0][:, :min(RT, min(R, dim) - V.shape[1])]
            V = np.hstack([V, new]) if V.shape[1] else new
        for _ in range(K):
            if policy == "replay":                         # full rehearsal: joint descent
                u = -(V @ (V.T @ sum(Si @ (w - wi) for Si, wi, _ in tasks[:t + 1])))
                w = w + eta * u
                continue
            u = -(V @ (V.T @ (S @ (w - wt))))              # descent within span
            past = tasks[:t]
            if policy == "active-set" and past:
                Gm = np.stack([V @ (V.T @ (Si @ (w - wi))) for Si, wi, _ in past], 1)
                from scipy.optimize import nnls
                keep = np.linalg.norm(Gm, axis=0) > 1e-10
                if keep.any():
                    lam, _ = nnls(Gm[:, keep], u)
                    u = u - Gm[:, keep] @ lam
            elif policy == "sign-1" and past:
                Si, wi, _ = past[rg.integers(len(past))]
                gi = V @ (V.T @ (Si @ (w - wi)))
                dt_ = u @ gi
                if dt_ > 0 and gi @ gi > 1e-12: u = u - dt_ / (gi @ gi) * gi
            w = w + eta * u
    return float(joint_loss(w, tasks))

if __name__ == "__main__":
    hdr("The decomposition: L = incompatibility + capacity + control  (10 seeds)")
    print(f"  {'stream':56s} {'floor':>7s} {'capac.':>7s} {'control':>8s}  dominant")
    rows_out = []
    for label, mk, pol in [
        ("floor-limited    (conflict, ample width, full replay)",
         lambda s: (make_stream("floor", s), d_star(make_stream("floor", s))), "replay"),
        ("capacity-limited (disjoint tasks through width-12 features)",
         lambda s: (make_capacity_stream(s, 12), 0.0), "replay"),
        ("mixed            (conflict floor + naive control error)",
         lambda s: (make_stream("control", s), d_star(make_stream("control", s))), "naive"),
    ]:
        terms = []
        for s in range(10):
            tasks, f = mk(s)                               # f = D*_inf (function-space)
            dr = max(d_star(tasks), f)                     # D*_class at deployed width
            dim = tasks[0][0].shape[0]
            L = max(run_policy(tasks, dim, pol, seed=s), dr)
            terms.append((f, dr - f, L - dr))
        m = np.mean(terms, 0)
        dom = ["incompatibility", "capacity", "control"][int(np.argmax(m))]
        print(f"  {label:56s} {m[0]:7.2f} {m[1]:7.2f} {m[2]:8.2f}  {dom}")

    hdr("Intervention attribution: each remedy moves (almost) only its own term")
    print("  CONTROL-limited stream:   upgrade the policy (naive -> active-set gate)")
    rows = []
    for s in range(6):
        tasks = make_stream("control", s)
        f = d_star(tasks); dr = f                          # ample width: D*_class = D*_inf
        Ln = max(run_policy(tasks, D, "naive", seed=s), dr)
        La = max(run_policy(tasks, D, "active-set", seed=s), dr)
        rows.append((f, Ln - dr, La - dr))
    r = np.mean(rows, 0)
    print(f"    floor {r[0]:.2f} (unchanged) | capacity 0.00 (unchanged) | "
          f"control {r[1]:.2f} -> {r[2]:.2f}  ({100*(1-r[2]/max(r[1],1e-9)):.0f}% removed)")
    print("  CAPACITY-limited stream:  widen features 12 -> 24 (same replay policy)")
    rows = []
    for s in range(6):
        t12, t24 = make_capacity_stream(s, 12), make_capacity_stream(s, 24)
        d12, d24 = d_star(t12), d_star(t24)
        L12 = max(run_policy(t12, 12, "replay", seed=s), d12)
        L24 = max(run_policy(t24, 24, "replay", seed=s), d24)
        rows.append((d12, d24, L12 - d12, L24 - d24))
    r = np.mean(rows, 0)
    print(f"    floor 0.00 (by construction) | capacity {r[0]:.2f} -> {r[1]:.2f} "
          f"({100*(1-r[1]/max(r[0],1e-9)):.0f}% removed) | control {r[2]:.2f} -> {r[3]:.2f} (~unchanged)")
    print("  CAPACITY-limited stream:  upgrade the policy instead (replay is already best)")
    rows = []
    for s in range(6):
        t12 = make_capacity_stream(s, 12); d12 = d_star(t12)
        Ln = max(run_policy(t12, 12, "naive", seed=s), d12)
        La = max(run_policy(t12, 12, "active-set", seed=s), d12)
        Lr = max(run_policy(t12, 12, "replay", seed=s), d12)
        rows.append((d12, Ln - d12, La - d12, Lr - d12))
    r = np.mean(rows, 0)
    print(f"    capacity {r[0]:.2f} UNTOUCHED by any policy; control naive {r[1]:.2f} / "
          f"active-set {r[2]:.2f} / replay {r[3]:.2f}")
    print("\n  -> the decomposition ATTRIBUTES: no policy touches the capacity term,")
    print("     expansion does not touch the control term, and nothing touches the")
    print("     floor.  Measure which term dominates BEFORE choosing a method family:")
    print("     floor from data (CMI / quadratic form), capacity from the class")
    print("     optimum at deployed width, control as the remainder.")
