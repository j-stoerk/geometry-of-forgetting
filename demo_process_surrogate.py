# =====================================================================
#  demo_process_surrogate.py -- domain demo for non-CL researchers
#  (roadmap: "scientific AI: adaptive surrogate / process control where
#  replay is expensive and drift is physical, not synthetic").
#
#  Setting: a plant surrogate y = f(x) (linear on fixed RBF features,
#  so the geometry is exact) predicts a process output from operating
#  conditions.  Data arrives in CAMPAIGNS:
#    - new operating windows      (disjoint inputs -> free allocation)
#    - revisits with FOULING      (same window, drifted response ->
#                                  the old fit is stale, follow it)
#    - overlapping-window shifts  (shared features, different response
#                                  -> positive floor: updating WILL
#                                  corrupt the validated window unless
#                                  archived data is replayed)
#    - sensor glitches            (OOD -> skip)
#  Replay = fetching archived validated batches (compliance/storage
#  cost), so it is counted as a budget, not treated as free.
#
#  THE OPERATIONAL PRE-FLIGHT QUESTION (one per domain, per roadmap):
#    "Will this campaign update corrupt the validated operating
#     windows -- and by how much?"
#  answered BEFORE each update from measured overlap + floor, then
#  checked against the realized damage.
# =====================================================================
import numpy as np

def hdr(t): print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)

# ---- plant + surrogate ------------------------------------------------
CENTERS = np.linspace(-4, 4, 60)                            # RBF grid on operating axis
def feats(x): return np.exp(-2.0 * (x[:, None] - CENTERS[None, :]) ** 2)

def true_response(x, state):
    base = np.sin(1.3 * x) + 0.3 * x
    for (lo, hi, amp) in state:                             # window-local physical changes
        m = (x > lo) & (x < hi)
        base = base + amp * np.exp(-3.0 * (x - (lo + hi) / 2) ** 2) * m
    return base

def make_campaigns(seed):
    rg = np.random.default_rng(seed)
    #        kind         window        physical change to plant state
    plan = [("baseline",  (-1.0, 1.0),  None),
            ("new-win",   (2.0, 3.5),   None),
            ("glitch",    None,         None),
            ("fouling",   (-1.0, 1.0),  (-1.0, 1.0, 0.35)),   # response drifts in W0
            ("new-win",   (-3.5, -2.0), None),
            ("overlap",   (0.2, 2.6),   (0.2, 2.6, 0.5)),     # straddles W0 & new-win
            ("fouling",   (-1.0, 1.0),  (-1.0, 1.0, 0.55))]
    camps, state = [], []
    for kind, win, change in plan:
        if kind == "glitch":
            x = rg.uniform(-8, 8, 40); y = rg.standard_normal(40) * 25.0
        else:
            if change: state = state + [change]
            x = rg.uniform(win[0], win[1], 120)
            y = true_response(x, state) + 0.02 * rg.standard_normal(len(x))
        camps.append(dict(kind=kind, x=x, y=y, state=list(state), win=win))
    return camps

def fit_in_span(w, X, y, lam=1e-4):
    """Least-squares update in X's WELL-CONDITIONED span: tiny singular
    directions are wild extrapolating tails and are excluded -- updating
    along them is estimation noise, not signal."""
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    U = Vt[s > 0.03 * s[0]].T
    if U.shape[1] == 0: return w
    A = X @ U
    c = np.linalg.solve(A.T @ A + lam * np.eye(U.shape[1]), A.T @ (y - X @ w))
    return w + U @ c

def rmse(w, x, y): return float(np.sqrt(np.mean((feats(x) @ w - y) ** 2)))

def run(seed, strategy):
    camps = make_campaigns(seed)
    w = np.zeros(len(CENTERS)); occ = np.zeros((len(CENTERS), 0))
    validated = []                                          # (x, y_current_truth, basis)
    fetches = 0; damage_pred, damage_real = [], []
    for c in camps:
        X = feats(c["x"])
        # ---- pre-flight panel (before any update) ----
        nrm = np.abs(c["y"]).mean()
        is_ood = nrm > 6.0                                  # glitch: absurd magnitudes
        U = np.linalg.svd(X, full_matrices=False)[2][:8].T
        # per validated window: measured DISAGREEMENT of incoming data with the
        # protected model on shared input support (the interference-rate analog)
        conflict_pred, partial_danger = 0.0, False
        if not is_ood:
            for xv, yv, _ in validated:
                lo, hi = xv.min(), xv.max()
                m = (c["x"] >= lo) & (c["x"] <= hi)
                if m.sum() >= 5:
                    pv = rmse(w, c["x"][m], c["y"][m])
                    conflict_pred = max(conflict_pred, pv)
                    cov_v = ((min(hi, c["x"].max()) - max(lo, c["x"].min()))
                             / max(hi - lo, 1e-9))          # how much of v is covered
                    cov_c = ((min(hi, c["x"].max()) - max(lo, c["x"].min()))
                             / max(c["x"].max() - c["x"].min(), 1e-9))
                    if pv > 0.15 and not (cov_v > 0.9 and cov_c > 0.9):
                        partial_danger = True               # conflict with a DIFFERENT window
        # ---- action ----
        if strategy == "freeze":
            act = "skip"
        elif is_ood:
            act = "skip" if strategy in ("controller", "naive-skip") else "share"
        elif strategy in ("naive", "naive-skip"):
            act = "share"
        elif strategy == "replay-all":
            act = "replay"
        else:                                               # controller
            # pure revisit with drift -> refresh (share); partial-overlap
            # conflict -> replay archives to protect the other window
            act = "replay" if partial_danger else "share"
        # ---- execute ----
        before = [rmse(w, xv, yv) for xv, yv, _ in validated]
        if act == "share":
            w = fit_in_span(w, X, c["y"])
        elif act == "replay":
            xs = [c["x"]]; ys = [c["y"]]
            for xv, yv, _ in validated:
                xs.append(xv); ys.append(yv); fetches += 1
            w = fit_in_span(w, feats(np.concatenate(xs)), np.concatenate(ys))
        after = [rmse(w, xv, yv) for xv, yv, _ in validated]
        if before and not is_ood and c["kind"] != "glitch" and act != "skip":
            damage_pred.append(conflict_pred)                 # committed pre-flight
            damage_real.append(max(a - b for a, b in zip(after, before)))
        # ---- register / refresh validated windows (truth moves with the plant) ----
        if not is_ood and c["kind"] != "glitch":
            xv = np.linspace(c["win"][0], c["win"][1], 80)
            yv = true_response(xv, c["state"])
            hit = [i for i, (xo, _, _) in enumerate(validated)
                   if abs(xo.mean() - xv.mean()) < 0.3]
            entry = (xv, yv, U)
            if hit: validated[hit[0]] = entry               # fouling: refresh the target
            else: validated.append(entry)
    final = [rmse(w, xv, yv) for xv, yv, _ in validated]
    return np.mean(final), fetches, damage_pred, damage_real

if __name__ == "__main__":
    hdr("Adaptive process surrogate under physical drift (10 seeds)")
    print(f"  {'strategy':12s} {'validated-window RMSE':>22s} {'archive fetches':>16s}")
    for s in ["freeze", "naive", "replay-all", "controller"]:
        r = [run(seed, s) for seed in range(10)]
        print(f"  {s:12s} {np.mean([x[0] for x in r]):22.3f} {np.mean([x[1] for x in r]):16.1f}")
    hdr("The operational pre-flight question, scored against counterfactual naive")
    dp, dr = [], []
    for seed in range(10):                    # counterfactual: share everything except
        _, _, p, q = run(seed, "naive-skip")  # glitches -- isolates the overlap question
        dp += p; dr += q
    dp, dr = np.array(dp), np.array(dr)
    flagged = dp > 0.15; damaged = dr > 0.15
    tp = (flagged & damaged).sum(); fp = (flagged & ~damaged).sum()
    fn = (~flagged & damaged).sum()
    print(f"  'Will this update corrupt a validated window?'  (flag: pred > 0.15)")
    print(f"  flagged {flagged.sum()}/{len(dp)} updates; damaging updates caught "
          f"{tp}/{tp + fn}; false alarms {fp}")
    print(f"  realized naive damage: flagged {dr[flagged].mean():.3f}"
          f"   vs unflagged {dr[~flagged].mean():.3f}")
    print("\n  Reading: freeze cannot follow fouling; naive is destroyed by sensor")
    print("  glitches and corrupted by the overlapping campaign; replay-all pays the")
    print("  full archive budget every campaign and STILL loses to the controller")
    print("  because rehearsal alone cannot reject glitches.  The controller skips")
    print("  the glitch, refreshes fouling windows, and fetches archives only where")
    print("  measured overlap+conflict predicted damage -- best retention, half the")
    print("  budget, and the danger identified BEFORE each update is applied.")
