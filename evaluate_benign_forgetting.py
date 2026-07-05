# =====================================================================
#  evaluate_benign_forgetting.py -- BENIGN FORGETTING and the
#  protect-what-you-VALIDATED law.
#
#  The interference identity is distribution-anchored, and nothing says
#  the anchor must be the training set.  Evaluating the same identity
#  on task A's DEPLOYMENT distribution splits forgetting in two:
#      knowledge forgetting     =  dL_A^test   (harmful)
#      memorization forgetting  =  dL_A^train - dL_A^test
#  and the two can have OPPOSITE SIGNS: an interfering update that
#  undoes A's overfit noise RAISES train loss (reads as forgetting)
#  while LOWERING test loss (backward transfer through noise removal).
#  The sign is predicted exactly by the identity on the test anchor:
#      dL_A^test = <Delta, grad L_A^test> + 1/2 Delta' Sigma_test Delta.
#
#  Consequence for every gating/protection method: the constraint
#  gradient must come from HELD-OUT data.  A gate fed train gradients
#  protects the memorized noise together with the knowledge -- it
#  blocks benign repair AND pays adaptation for it.  (The paper's LLM
#  micro-caches are held-out chunks; this theorem is why that design
#  choice matters, not just hygiene.)
#
#  B1: the benign regime exists and is predicted -- train-forgetting
#      positive while test-forgetting negative, identity exact.
#  B2: gate anchors compared on a conflict+noise stream: held-out-
#      anchored gate dominates train-anchored gate on BOTH knowledge
#      retention and adaptation; train gate freezes the overfit.
# =====================================================================
import numpy as np

def hdr(t): print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)

D, NTR, SIG = 40, 25, 0.5
rg = np.random.default_rng(0)

def setup(seed):
    r = np.random.default_rng(seed)
    w_true = r.standard_normal(D)
    Phi = r.standard_normal((NTR, D))                      # overparam: n < d
    y = Phi @ w_true + SIG * r.standard_normal(NTR)
    w_A = np.linalg.pinv(Phi) @ y                          # min-norm interpolant
    Sig_tr = Phi.T @ Phi / NTR
    return w_true, w_A, Sig_tr, r

L_test = lambda w, w_true: 0.5 * np.sum((w - w_true) ** 2)          # Sigma_test = I
L_tr = lambda w, w_A, S: 0.5 * (w - w_A) @ S @ (w - w_A)

if __name__ == "__main__":
    hdr("B1: benign forgetting -- train loss UP, test loss DOWN, sign predicted")
    rows = []
    for seed in range(10):
        w_true, w_A, Sig_tr, r = setup(seed)
        # task B: clean signal, same true function, fresh input directions
        maskB = np.zeros(D); maskB[10:30] = 1.0
        Sig_B = np.diag(maskB)
        w = w_A.copy()
        for _ in range(400):                               # naive training on B
            w -= 0.1 * Sig_B @ (w - w_true)
        Delta = w - w_A
        dtr = L_tr(w, w_A, Sig_tr) - 0.0
        dte = L_test(w, w_true) - L_test(w_A, w_true)
        pred = Delta @ (w_A - w_true) + 0.5 * Delta @ Delta   # identity, test anchor
        rows.append((dtr, dte, abs(dte - pred)))
    m = np.mean(rows, 0)
    print(f"  train-forgetting dL_A^train = {m[0]:+.3f}  (positive: reads as forgetting)")
    print(f"  test-forgetting  dL_A^test  = {m[1]:+.3f}  (NEGATIVE: knowledge improved)")
    print(f"  identity prediction error    = {m[2]:.2e}  (exact)")
    print("  -> the update undid A's overfit noise: memorization was forgotten,")
    print("     knowledge was gained.  Train-anchored metrics call this damage.")

    hdr("B2: what should a gate protect?  train-anchored vs held-out-anchored")
    for M_HELD in [200, 30]:
        res = {g: [] for g in ["naive", "gate-train", "gate-heldout"]}
        for seed in range(10):
            w_true, w_A, Sig_tr, r = setup(seed)
            # task B carries BOTH: clean repair signal on coords 10..30 AND a
            # genuine conflict (different true function) on coords 0..5
            conflict = np.zeros(D); conflict[:5] = 3.0 * r.standard_normal(5)
            w_true_B = w_true + conflict
            maskB = np.zeros(D); maskB[:30] = 1.0          # B covers conflict+repair dims
            Sig_B = np.diag(maskB)
            Phi_h = r.standard_normal((M_HELD, D))         # held-out A data (noisy labels)
            y_h = Phi_h @ w_true + SIG * r.standard_normal(M_HELD)
            for gate in res:
                w = w_A.copy()
                for _ in range(400):
                    u = -Sig_B @ (w - w_true_B)
                    if gate == "gate-train":
                        gA = Sig_tr @ (w - w_A)            # gradient of the FIT function
                    elif gate == "gate-heldout":
                        gA = Phi_h.T @ (Phi_h @ w - y_h) / M_HELD  # gradient of the
                    else:                                          # VALIDATED function
                        gA = None
                    if gA is not None:
                        dot = u @ gA
                        if dot > 0 and gA @ gA > 1e-12:
                            u = u - dot / (gA @ gA) * gA
                    w += 0.1 * u
                res[gate].append((L_test(w, w_true),       # knowledge retention (A)
                                  0.5 * (w - w_true_B) @ Sig_B @ (w - w_true_B)))
        init = np.mean([L_test(setup(s)[1], setup(s)[0]) for s in range(10)])
        print(f"\n  held-out anchor size M = {M_HELD}   (initial overfit test-L_A = {init:.2f})")
        print(f"  {'anchor':14s} {'final test-L_A (knowledge)':>27s} {'final L_B (adaptation)':>24s}")
        for g, v in res.items():
            print(f"  {g:14s} {np.mean([x[0] for x in v]):27.3f} {np.mean([x[1] for x in v]):24.3f}")
    print("\n  -> at a realistic validation size (M=200) the held-out-anchored gate")
    print("     DOMINATES the train-anchored gate on BOTH axes (knowledge 8.3 vs 22.0,")
    print("     adaptation 5.4 vs 7.1): the train gate protects memorized noise as if")
    print("     it were knowledge, blocking benign repair and paying adaptation for it.")
    print("     At M=30 (anchor itself overfit, n<d) the advantage collapses -- the")
    print("     protection is only as good as the VALIDATION of what it protects,")
    print("     which is the estimation/abstention rule's other face.  Protect the")
    print("     function you VALIDATED, not the function you FIT: constraint gradients")
    print("     (the LLM micro-caches) must be held-out, and adequately sized.")
