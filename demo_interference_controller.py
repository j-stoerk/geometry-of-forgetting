# =====================================================================
#  demo_interference_controller.py -- validates the interference ledger
#  + controller (D5): does geometry-driven control choose the RIGHT
#  action per regime, and does that beat fixed strategies?
#
#  A synthetic stream with a KNOWN ground-truth regime per task
#  (disjoint / aligned-overlap / conflicting-overlap / OOD-noise /
#  capacity-exhausted) provides a decision ORACLE.  We report both
#  task metrics (final excess loss) and DECISION metrics (did the
#  controller pick the oracle action; false-share / false-separate /
#  missed-transfer rates) -- the roadmap's "evaluate decision quality,
#  not only end performance".
# =====================================================================
import numpy as np
from interference_ledger import (InterferenceLedger, InterferenceController,
                                  principal_overlap, _top_r)

def hdr(t): print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)
def orth(rg, d, r): return np.linalg.qr(rg.standard_normal((d, r)))[0]
def L(w, S, ws): return float(0.5 * (w - ws) @ S @ (w - ws))


def make_stream(seed, d=40, r=4, T=14):
    """A labeled stream: each task carries its ground-truth ORACLE action."""
    rg = np.random.default_rng(seed)
    B0 = orth(rg, d, r); w0 = B0 @ rg.standard_normal(r)
    tasks = []
    regimes = (["disjoint", "aligned", "conflict", "disjoint", "conflict",
                "aligned", "ood", "disjoint", "conflict", "aligned",
                "conflict", "disjoint", "aligned", "conflict"])[:T]
    for t, reg in enumerate(regimes):
        if reg == "disjoint":
            B = orth(rg, d, r); w = B @ rg.standard_normal(r); oracle = "project_or_share"
        elif reg == "aligned":
            B = np.linalg.qr(B0 + 0.1 * rg.standard_normal((d, r)))[0]
            w = w0 + 0.1 * B @ rg.standard_normal(r); oracle = "share"
        elif reg == "conflict":
            B = np.linalg.qr(B0 + 0.1 * rg.standard_normal((d, r)))[0]
            w = -w0 + B @ rg.standard_normal(r); oracle = "project"
        else:  # ood
            B = orth(rg, d, r); w = B @ rg.standard_normal(r) * 8.0; oracle = "skip"
        X = rg.standard_normal((64, r)) @ B.T * (8.0 if reg == "ood" else 1.0)
        tasks.append(dict(reg=reg, oracle=oracle, X=X, B=B, w=w))
    return tasks, B0, w0


def run_controller(seed, strategy, capacity=None, memory=0):
    tasks, B0, w0 = make_stream(seed)
    d = tasks[0]["X"].shape[1]
    led = InterferenceLedger(d, rank=4, capacity=capacity, sstar=0.5, warmup_steps=3)
    ctl = InterferenceController(led, memory_budget=memory, conf_floor=0.5)
    w = np.zeros(d); occ = np.zeros((d, 0)); decisions = []
    heads = []                                            # (B, w_true) of accepted tasks
    for t, task in enumerate(tasks):
        X, Bt, wt = task["X"], task["B"], task["w"]
        Sig = Bt @ Bt.T
        g = Sig @ (w - wt)                               # grad L_new at current w
        # functional gate: grad L_u of the most-overlapping stored task
        gp = None
        if heads:
            j = int(np.argmax([principal_overlap(Bh, Bt) for Bh, _ in heads]))
            Bh, wh = heads[j]
            gp = (Bh @ Bh.T) @ (w - wh)                  # grad L_u
        if strategy == "controller":
            dec = ctl.decide(g, Phi_new=X, target_new=wt, grad_protected=gp)
            action = dec.action
            decisions.append((task["oracle"], action, dec.reason))
        else:
            action = strategy
            led.log(g, Phi_new=X, target_new=wt)          # still log metrics for fairness
        # apply the action
        if action == "skip":
            pass
        elif action in ("project", "defer", "expand"):
            P = np.eye(d) - occ @ occ.T if occ.shape[1] else np.eye(d)
            wr = (np.linalg.qr(P @ Bt)[0][:, :4]); w = w + wr @ (wr.T @ (wt - w))
            occ = np.linalg.qr(np.hstack([occ, wr]))[0] if occ.shape[1] else wr
        elif action == "replay":
            w = w + 0.5 * Bt @ (Bt.T @ (wt - w)) + 0.5 * B0 @ (B0.T @ (w0 - w))
        else:  # share (or route)
            w = w + Bt @ (Bt.T @ (wt - w))
        led.tracker.update(X)
        if action != "skip":
            heads.append((Bt, wt))
            led.register_task(X, target=wt)               # controller now sees this stored task
    # task metric: mean retained-task excess loss at the end
    fgt = np.mean([L(w, Bh @ Bh.T, wh) for Bh, wh in heads]) if heads else 0.0
    return fgt, decisions


def _consistent(oracle, action):
    if oracle == "share": return action in ("share", "route")
    if oracle == "project": return action in ("project", "defer", "expand", "replay")
    if oracle == "skip": return action == "skip"
    return True                                          # disjoint: both lossless


def decision_quality(decisions):
    """Score controller decisions against the oracle labels."""
    ok, false_share, false_sep, missed = 0, 0, 0, 0
    for oracle, action, _ in decisions:
        share_like = action in ("share", "route")
        sep_like = action in ("project", "defer", "expand", "replay")
        if oracle == "share":
            ok += share_like; false_sep += sep_like; missed += sep_like
        elif oracle == "project":
            ok += sep_like; false_share += share_like
        elif oracle == "skip":
            ok += action == "skip"
        else:  # project_or_share (disjoint: both are lossless)
            ok += 1
    n = max(len(decisions), 1)
    return dict(acc=ok / n, false_share=false_share / n,
                false_separate=false_sep / n, missed_transfer=missed / n)


if __name__ == "__main__":
    hdr("Geometry-driven controller vs fixed strategies (task metric: mean retained excess loss)")
    seeds = range(12)
    strategies = ["share", "project", "replay", "controller"]
    task_scores = {s: [] for s in strategies}
    dq = []
    for seed in seeds:
        for s in strategies:
            fgt, dec = run_controller(seed, s, capacity=20, memory=(8 if s in ("replay", "controller") else 0))
            task_scores[s].append(fgt)
            if s == "controller":
                dq.append(decision_quality(dec))
    print(f"  {'strategy':14s} {'mean retained excess loss':>26s}")
    for s in strategies:
        print(f"  {s:14s} {np.mean(task_scores[s]):26.3f}")
    hdr("Decision quality of the controller (vs the regime oracle)")
    print(f"  action accuracy       : {np.mean([q['acc'] for q in dq]):.2f}")
    print(f"  false-share rate      : {np.mean([q['false_share'] for q in dq]):.2f}  "
          f"(shared where the oracle said conflict -- the costly error)")
    print(f"  false-separate rate   : {np.mean([q['false_separate'] for q in dq]):.2f}  "
          f"(protected where sharing was safe)")
    print(f"  missed-transfer rate  : {np.mean([q['missed_transfer'] for q in dq]):.2f}")
    print("\n  example decision trace (seed 0):")
    _, dec0 = run_controller(0, "controller", capacity=20, memory=8)
    for oracle, action, reason in dec0[:8]:
        flag = "OK " if _consistent(oracle, action) else "XX "
        print(f"    [{flag}] oracle={oracle:18s} chose={action:8s}  ({reason})")
    print("\n  -> the controller turns the measured ledger into per-regime actions; "
          "decision accuracy is reported ALONGSIDE the task metric, so a good score "
          "from good CHOICES is distinguishable from a good default.")
