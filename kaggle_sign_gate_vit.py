# =====================================================================
#  kaggle_sign_gate_vit.py -- can the threshold-free functional sign
#  gate REPLACE the s*-threshold gate on the vision streams?
#
#  Run inside the kaggle_vit_multiseed.py notebook environment (it reads
#  the cached ViT features in ./feat_cache).  Self-contained otherwise.
#
#  Streams: Split-CIFAR-100, ImageNet-R, CUB (task-incremental, 10 tasks)
#  and the CIFAR-100 superclass similar-task stream.  Methods:
#    naive       : plain Adam on the shared linear head
#    ogd         : protect-all (null-space of per-task feature bases)
#    igfa (s*)   : the paper's threshold gate (s* = 0.60)
#    func-sign   : threshold-free -- per step, sample one past task,
#                  compute grad L_u from a MICRO-CACHE (2 features per
#                  class), and remove the update's component only if it
#                  would increase L_u.  No threshold, no subspaces.
#  Prediction: func-sign inherits the ImageNet-R win without s*, ties
#  the exact-tie streams, and needs no per-stream calibration.
#  5 seeds, mean +/- std, paired t-tests.  ~15 min (features cached).
# =====================================================================
import os, json
import numpy as np
import torch, torch.nn.functional as F
from scipy import stats

SEEDS, SSTAR, RANK, EPOCHS, BS, LR = [0, 1, 2, 3, 4], 0.60, 12, 4, 256, 1e-3
CACHE_PER_CLASS = 2
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
CACHE = "./feat_cache"

def task_basis(X, r=RANK):
    Xc = torch.as_tensor(X, dtype=torch.float32, device=DEV)
    return torch.linalg.svd(Xc, full_matrices=False)[2][:r].T.contiguous()

def run_stream(feats, seed, method, superclass=False):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    Xtr, Xte = feats["Xtr"], feats["Xte"]
    if superclass:
        yc_tr, yc_te = feats["ctr"], feats["cte"]
        fine_tr, fine_te = feats["ytr"], feats["yte"]
        assign = {}
        for sc in range(20):
            fines = rng.permutation(np.unique(fine_tr[yc_tr == sc]))
            for t, fc in enumerate(fines): assign[fc] = t
        tr_tasks = [np.where([assign[f] == t for f in fine_tr])[0] for t in range(5)]
        te_tasks = [np.where([assign[f] == t for f in fine_te])[0] for t in range(5)]
        ytr, yte, C, masks = yc_tr, yc_te, 20, [None] * 5
    else:
        C = int(feats["C"]); ytr, yte = feats["ytr"], feats["yte"]
        order = rng.permutation(C); per = C // 10
        cls_tasks = [order[t * per:(t + 1) * per] for t in range(10)]
        tr_tasks = [np.where(np.isin(ytr, cs))[0] for cs in cls_tasks]
        te_tasks = [np.where(np.isin(yte, cs))[0] for cs in cls_tasks]
        masks = [torch.as_tensor(np.isin(np.arange(C), cs), device=DEV) for cs in cls_tasks]
    T = len(tr_tasks)
    W = torch.zeros(C, Xtr.shape[1], device=DEV, requires_grad=True)
    opt = torch.optim.Adam([W], lr=LR)
    bases, Q, caches = [], None, []
    acc_after = np.zeros(T)

    def task_acc(t):
        X = torch.as_tensor(Xte[te_tasks[t]], dtype=torch.float32, device=DEV)
        logit = X @ W.T
        if masks[t] is not None: logit = logit.masked_fill(~masks[t], -1e9)
        return float((logit.argmax(1).cpu().numpy() == yte[te_tasks[t]]).mean())

    for t in range(T):
        Bn = task_basis(Xtr[tr_tasks[t]])
        if method in ("ogd", "igfa") and bases:
            ovl = [float((B.T @ Bn).square().sum().item()) / RANK for B in bases]
            prot = bases if method == "ogd" else [B for B, o in zip(bases, ovl) if o < SSTAR]
            Q = torch.linalg.qr(torch.cat(prot, 1))[0] if prot else None
        X = torch.as_tensor(Xtr[tr_tasks[t]], dtype=torch.float32, device=DEV)
        y = torch.as_tensor(ytr[tr_tasks[t]], dtype=torch.long, device=DEV)
        for _ in range(EPOCHS):
            for i in torch.randperm(len(X), device=DEV).split(BS):
                logit = X[i] @ W.T
                if masks[t] is not None: logit = logit.masked_fill(~masks[t], -1e9)
                loss = F.cross_entropy(logit, y[i])
                opt.zero_grad(); loss.backward()
                if method in ("ogd", "igfa") and Q is not None:
                    W.grad -= (W.grad @ Q) @ Q.T
                elif method == "func-sign" and caches:
                    g = W.grad.detach().clone()
                    u = int(rng.integers(len(caches)))
                    Xu, yu, mu = caches[u]
                    logit_u = Xu @ W.T
                    if mu is not None: logit_u = logit_u.masked_fill(~mu, -1e9)
                    W.grad = None
                    F.cross_entropy(logit_u, yu).backward()
                    nA = W.grad.detach().clone()
                    dot = float((g * nA).sum()); nrm2 = float((nA * nA).sum())
                    if nrm2 > 1e-12 and dot < 0:
                        g = g - dot / nrm2 * nA
                    W.grad = g
                opt.step()
        bases.append(Bn)
        idxc = []                                          # micro-cache: 2 features per class
        for c in np.unique(ytr[tr_tasks[t]]):
            idxc += list(np.where(ytr[tr_tasks[t]] == c)[0][:CACHE_PER_CLASS])
        caches.append((X[idxc].detach(), y[idxc].detach(),
                       masks[t] if masks[t] is not None else None))
        acc_after[t] = task_acc(t)
    final = np.array([task_acc(t) for t in range(T)])
    return final.mean(), float((acc_after - final)[:-1].mean())


if __name__ == "__main__":
    results = {}
    streams = [("SplitCIFAR100", "cifar100", False), ("ImageNetR", "imagenet_r", False),
               ("CUB", "cub", False), ("C100-Superclass", "cifar100", True)]
    for tag, cache_name, sup in streams:
        fn = os.path.join(CACHE, f"{cache_name}.npz")
        if not os.path.exists(fn):
            print(f"{tag}: SKIP ({fn} missing -- run kaggle_vit_multiseed.py first)"); continue
        feats = dict(np.load(fn))
        per = {m: [] for m in ["naive", "ogd", "igfa", "func-sign"]}
        for seed in SEEDS:
            for m in per:
                per[m].append(run_stream(feats, seed, m, superclass=sup))
        print(f"\n===== {tag} (5 seeds) =====")
        for m, v in per.items():
            print(f"  {m:10s}: acc {np.mean([x[0] for x in v]):.3f}+/-{np.std([x[0] for x in v]):.3f}"
                  f"   forget {np.mean([x[1] for x in v]):+.3f}+/-{np.std([x[1] for x in v]):.3f}")
        results[tag] = {m: dict(acc=[x[0] for x in v], forget=[x[1] for x in v])
                        for m, v in per.items()}
        for a, b in [("func-sign", "igfa"), ("func-sign", "ogd"), ("func-sign", "naive")]:
            xa = [x[0] for x in per[a]]; xb = [x[0] for x in per[b]]
            p = 1.0 if np.allclose(xa, xb) else stats.ttest_rel(xa, xb).pvalue
            print(f"  paired acc {a} vs {b}: {np.mean(xa):.3f} vs {np.mean(xb):.3f} (p={p:.4f})")
        with open("sign_gate_vit_results.json", "w") as f: json.dump(results, f, indent=1)
    print("\nALL DONE -> sign_gate_vit_results.json")
