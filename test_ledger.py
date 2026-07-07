# Smoke tests for the interference-ledger toolkit (pytest or plain python).
import numpy as np
from interference_ledger import (GeometryTracker, InterferenceLedger,
                                  InterferenceController, principal_overlap,
                                  kl_interference, jsd_floor, qp_gate,
                                  constraints_from_tasks, AutonomousController)

rg = np.random.default_rng(0)
def orth(d, r): return np.linalg.qr(rg.standard_normal((d, r)))[0]

def test_tracker_recovers_subspace():
    d, r = 30, 4
    B = orth(d, r)
    tr = GeometryTracker(d, rank=r, decay=0.8)
    for _ in range(40):
        tr.update(rg.standard_normal((64, r)) @ B.T)
    assert principal_overlap(tr.subspace(r), B) > 0.99

def test_fd_sketch_matches_recursive():
    d, r = 30, 4
    B = orth(d, r)
    a = GeometryTracker(d, rank=r, decay=0.8)
    b = GeometryTracker(d, rank=r, decay=0.8, method="fd", ell=2 * r)
    for _ in range(40):
        X = rg.standard_normal((64, r)) @ B.T
        a.update(X); b.update(X)
    assert principal_overlap(a.subspace(r), b.subspace(r)) > 0.98

def test_drift_velocity_spikes_at_boundary():
    d, r = 30, 4
    tr = GeometryTracker(d, rank=r, decay=0.8)
    B1, B2 = orth(d, r), orth(d, r)
    v_steady, v_switch = [], []
    for i in range(60):
        tr.update(rg.standard_normal((64, r)) @ (B1 if i < 30 else B2).T)
        (v_steady if 10 < i < 30 or i > 45 else v_switch).append(tr.drift_velocity())
    assert max(v_switch[20:24] or [1]) > 3 * (np.mean(v_steady) + 1e-9)

def test_ledger_metrics_sane():
    d = 30
    led = InterferenceLedger(d, rank=4, capacity=12, warmup_steps=2)
    B = orth(d, 4)
    led.register_task(rg.standard_normal((64, 4)) @ B.T, target=B @ np.ones(4))
    led.tracker.update(rg.standard_normal((64, 4)) @ B.T)
    g = B @ rg.standard_normal(4)
    row = led.log(g, Phi_new=rg.standard_normal((64, 4)) @ B.T)
    assert row.interference_energy > 0
    assert 0.9 < row.max_overlap <= 1.0 + 1e-9
    assert 0 <= row.ood_ratio <= 1 + 1e-9

def test_controller_regimes():
    d = 30
    led = InterferenceLedger(d, rank=4, capacity=40, warmup_steps=1)
    ctl = InterferenceController(led, memory_budget=4, conf_floor=0.0)
    B = orth(d, 4); w = B @ np.ones(4)
    for _ in range(5):
        led.tracker.update(rg.standard_normal((64, 4)) @ B.T)
    led.register_task(rg.standard_normal((64, 4)) @ B.T, target=w)
    gp = B @ B.T @ (np.zeros(d) - w)                       # grad of protected loss
    # decide() takes the NEW task's gradient g; the update is -g, so
    # rate = <g, grad_protected> < 0 means the step ASCENDS the protected loss.
    conflict = ctl.decide(-gp, grad_protected=gp)           # rate < 0: conflict
    share = ctl.decide(gp * 1.0, grad_protected=gp)         # rate > 0: aligned/transfer
    assert conflict.action in ("project", "replay", "defer", "expand")
    assert share.action in ("share", "route")

def test_bregman_helpers():
    sm = lambda F: np.exp(F - F.max(1, keepdims=True)) / np.exp(F - F.max(1, keepdims=True)).sum(1, keepdims=True)
    p, q = sm(rg.standard_normal((40, 5))), sm(rg.standard_normal((40, 5)))
    assert kl_interference(p, p) < 1e-12
    assert kl_interference(p, q) > 0
    d1, d2 = rg.uniform(0.5, 2, 40), rg.uniform(0.5, 2, 40)
    assert jsd_floor(p, p, d1, d2) < 1e-12
    assert jsd_floor(p, q, d1, d2) > 0

def test_autonomous_detects_boundary_not_ood():
    rg = np.random.default_rng(0)
    d, r = 40, 4
    B1, B2 = orth(d, r), orth(d, r)
    ctl = AutonomousController(d, rank=r)
    ev = []
    for _ in range(12):                                     # task 1
        ev.append(ctl.observe(rg.standard_normal((32, r)) @ B1.T).event)
    ev.append(ctl.observe(rg.standard_normal((32, r)) @ orth(d, r).T * 6).event)  # 1 OOD blip
    for _ in range(12):                                     # task 2 (persistent shift)
        ev.append(ctl.observe(rg.standard_normal((32, r)) @ B2.T).event)
    assert "boundary" in ev                                 # the persistent shift is caught
    assert ctl.n_anchors_seen if False else len(ctl.anchors) >= 1
    # the single OOD batch did not create a task
    assert ev.count("boundary") <= 2

def test_autonomous_bounded_anchors():
    rg = np.random.default_rng(1)
    d, r, cap = 40, 4, 5
    ctl = AutonomousController(d, rank=r, cap=cap)
    for t in range(20):                                     # 20 distinct tasks
        B = orth(d, r)
        for _ in range(6):
            ctl.observe(rg.standard_normal((32, r)) @ B.T)
    assert len(ctl.anchors) <= cap                          # bounded cost

def test_qp_gate_recovers_all_methods():
    r = np.random.default_rng(3)
    u = r.standard_normal(30)
    assert np.allclose(qp_gate(u), u)                       # naive
    g = r.standard_normal(30); g = g if u @ g > 0 else -g
    v = u - (u @ g) / (g @ g) * g                           # sign gate closed form
    assert np.allclose(qp_gate(u, G=g[:, None]), v, atol=1e-12)
    Q = np.linalg.qr(r.standard_normal((30, 6)))[0]
    assert np.allclose(qp_gate(u, G_eq=Q), u - Q @ (Q.T @ u), atol=1e-12)  # OGD
    B1, B2 = Q[:, :3], np.linalg.qr(r.standard_normal((30, 3)))[0]
    E = constraints_from_tasks([B1, B2], B_new=B1, sstar=0.5)   # igfa filter:
    assert E.shape[1] == 3                                  # aligned B1 dropped

if __name__ == "__main__":
    for name, fn in sorted({k: v for k, v in globals().items()
                            if k.startswith("test_")}.items()):
        fn(); print(f"  {name}: OK")
    print("all ledger smoke tests passed")
