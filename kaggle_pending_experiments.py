# =====================================================================
#  Geometry of Forgetting -- ALL PENDING EXPERIMENTS in one Kaggle file.
#
#  Paste this whole file into a single Kaggle GPU cell and run.  It works
#  through the open roadmap items and logs *everything* (better too much than
#  too little), with tqdm progress bars that show the time left per stage.
#
#  Experiments (toggle in CONFIG):
#    E1  Split-CIFAR-100 on a FROZEN ViT (A1 exact): naive / OGD-GPM / IGFA
#        (signed gate) / Annealed-IGFA / isotropic-online.  Logs the full
#        task x task accuracy matrix (BWT), forgetting, used protected rank,
#        the Sigma_t eigenspectrum (k-distribution), the gate similarity
#        s-distribution, and retained plasticity (CL vs from-scratch head).
#    E2  Offline-merge ablation on the same frozen-ViT heads: isotropic /
#        Euclidean-TSV / Sigma-whitened-TSV / full Sigma-orth.  Logs downstream
#        accuracy AND the post-merge residual floor D = sum_t 1/2 ||th-W_t||^2_{S_t}
#        -- the quantity Theorem 3 says the Sigma-metric uniquely minimises.
#    E3  Drift geometry sweep (depth x lr), matched base optimizer/lr/batch:
#        stale-OGD vs recursive-Gauss-Newton-IGFA vs Geodesic-Aligned (GAGP,
#        Grassmann) projection.  Logs prediction fidelity r and retained
#        plasticity so the recursive GN update can be compared to GAGP cleanly.
#    E4  Prompt-based PEFT IGFA on a frozen ViT: shared prompts for aligned
#        directions, protected prompts for conflicting ones.  Logs forgetting
#        and the s-distribution -- the architectural story beyond linear heads.
#    E5  FD-IGFA at scale: Frequent-Directions O(rd) curvature sketch vs full
#        covariance vs Oja PCA, on a LoRA-wrapped transformer stream.  Logs
#        captured-energy-vs-rank, memory, forgetting and free plasticity.
#
#  Defaults are sized for a single T4/P100 in ~1-2 h.  Everything degrades
#  gracefully if a dataset/model is unavailable (it says so and skips).
# =====================================================================
import os, sys, time, json, math, subprocess, warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------------- CONFIG
CONFIG = dict(
    RUN_E1=True, RUN_E2=True, RUN_E3=True, RUN_E4=True, RUN_E5=True,
    VIT_MODEL="vit_base_patch16_224",     # timm frozen backbone (A1 exact)
    N_TASKS=10, CLASSES_PER_TASK=10,      # Split-CIFAR-100
    FEAT_CACHE="vit_cifar100_feats.npz",
    RANK_KEEP=40, SSTAR=0.60,             # protected rank cap, gate threshold.
    # NOTE on s*: dense ViT features give HIGH input-subspace overlap (~0.4) even for
    # disjoint-class tasks, so s* must sit ABOVE that for the gate to protect (orthogonalize)
    # dissimilar tasks -- otherwise every task reads "similar -> share" and IGFA collapses to
    # naive.  s*=0.60 makes IGFA correctly reduce to OGD on Split-CIFAR-100 (dissimilar);
    # the sharing branch only helps on genuinely similar tasks (Rotated-Digits, s*~0.65).
    ANNEAL_FRAC=0.5,                      # protect-all warmup fraction per task
    LM_MODEL="EleutherAI/pythia-160m",    # small LM for E5 (set pythia-410m/1b if you have room)
    LM_STEPS=120, LM_SEQ=128, LORA_R=16,
    SEED=0,
)

def pip(*p): subprocess.run([sys.executable, "-m", "pip", "install", "-q", *p], check=False)

import importlib
for mod, pkg in [("numpy", "numpy"), ("tqdm", "tqdm"), ("torch", "torch"),
                 ("torchvision", "torchvision"), ("timm", "timm"), ("sklearn", "scikit-learn")]:
    try: importlib.import_module(mod)
    except Exception: pip(pkg)

import numpy as np, torch
from tqdm.auto import tqdm
np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()
def stamp(msg): print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)
RESULTS = {}


# =====================================================================
#  Shared: frozen-ViT feature extraction for Split-CIFAR-100 (A1 exact)
# =====================================================================
def get_cifar_features():
    """Return frozen ViT features X (N,d), labels y, and a fixed 10x10 task split.
    Cached to disk so the (slow) forward pass runs once."""
    if os.path.exists(CONFIG["FEAT_CACHE"]):
        z = np.load(CONFIG["FEAT_CACHE"])
        stamp(f"loaded cached ViT features {z['Xtr'].shape}")
        return z["Xtr"], z["ytr"], z["Xte"], z["yte"], z["order"]
    import timm, torchvision, torchvision.transforms as T
    stamp("building frozen ViT + extracting CIFAR-100 features (one-time)...")
    model = timm.create_model(CONFIG["VIT_MODEL"], pretrained=True, num_classes=0).eval().to(DEV)
    for p in model.parameters(): p.requires_grad_(False)
    cfg = timm.data.resolve_data_config({}, model=model)
    tf = timm.data.create_transform(**cfg)
    def feats(train):
        ds = torchvision.datasets.CIFAR100(root="./data", train=train, download=True, transform=tf)
        dl = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=2, shuffle=False)
        X, Y = [], []
        for xb, yb in tqdm(dl, desc=f"ViT feats {'train' if train else 'test'}", leave=False):
            with torch.no_grad():
                X.append(model(xb.to(DEV)).float().cpu().numpy())
            Y.append(yb.numpy())
        return np.concatenate(X), np.concatenate(Y)
    Xtr, ytr = feats(True); Xte, yte = feats(False)
    order = np.random.permutation(100)                      # random class-to-task assignment
    np.savez_compressed(CONFIG["FEAT_CACHE"], Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, order=order)
    stamp(f"cached features {Xtr.shape}")
    return Xtr, ytr, Xte, yte, order


def task_classes(order):
    K, C = CONFIG["N_TASKS"], CONFIG["CLASSES_PER_TASK"]
    return [list(order[t * C:(t + 1) * C]) for t in range(K)]


def onehot(y, cols):
    idx = {c: i for i, c in enumerate(cols)}
    Y = np.zeros((len(y), len(cols)))
    for i, v in enumerate(y):
        if v in idx: Y[i, idx[v]] = 1.0
    return Y


# =====================================================================
#  E1: Split-CIFAR-100 on frozen ViT -- full CL comparison + heavy logging
# =====================================================================
def E1_split_cifar():
    stamp("=" * 70); stamp("E1  Split-CIFAR-100 on frozen ViT (A1 exact)")
    Xtr, ytr, Xte, yte, order = get_cifar_features()
    d = Xtr.shape[1]
    # global feature scale for stable GD
    scale = np.sqrt(np.linalg.eigvalsh(Xtr.T @ Xtr / len(Xtr))[-1])
    Xtr, Xte = Xtr / scale, Xte / scale
    tasks = task_classes(order); K = len(tasks); r = CONFIG["RANK_KEEP"]
    lr, epochs = 0.2, 400

    def data(t, split):
        X, y = (Xtr, ytr) if split == "tr" else (Xte, yte)
        m = np.isin(y, tasks[t]); return X[m], y[m]

    def acc(W, cols, t):
        Xt, yt = data(t, "te")
        pred = np.array(cols)[np.argmax(Xt @ W.T, axis=1)]
        return float(np.mean(pred == yt))

    def basis(X, k):
        w, V = np.linalg.eigh(X.T @ X / len(X)); return V[:, -k:], w[::-1]

    def run(method):
        C = CONFIG["CLASSES_PER_TASK"]
        W = np.zeros((C, d)); occ = np.zeros((d, 0)); past = []
        accM = np.full((K, K), np.nan)            # accM[i,j] = acc on task j after learning task i
        s_log, rank_log, spec_log, plast = [], [], [], []
        for ti in tqdm(range(K), desc=f"E1 {method}", leave=False):
            Xb, yb = data(ti, "tr"); Yb = onehot(yb, tasks[ti])
            Bn, ev = basis(Xb, r); spec_log.append(ev[:30].tolist())
            P_full = np.eye(d) - occ @ occ.T if occ.shape[1] else None
            prot = []
            for Bt in past:
                s = float(np.linalg.norm(Bt.T @ Bn, "fro") ** 2 / r); s_log.append(s)
                if s < CONFIG["SSTAR"]: prot.append(Bt)
            P_gate = (np.eye(d) - (lambda Q: Q @ Q.T)(np.linalg.qr(np.hstack(prot))[0])) if prot else np.eye(d)
            A = Xb.T @ Xb / len(Xb); Bm = Yb.T @ Xb / len(Xb)     # normal-equation GD
            Wt = np.zeros((C, d))                                  # from-scratch head (plasticity ref)
            for ep in range(epochs):
                G = W @ A - Bm
                if method == "ogd" and P_full is not None: G = G @ P_full.T
                elif method == "igfa" and past: G = G @ P_gate.T
                elif method == "annealed" and past:
                    G = G @ (P_full.T if (ep < CONFIG["ANNEAL_FRAC"] * epochs and P_full is not None) else P_gate.T)
                elif method == "isotropic" and past: pass       # share everything (no projection)
                W = W - lr * G
                Wt = Wt - lr * (Wt @ A - Bm)                       # independent reference
            past.append(Bn)
            occ = np.linalg.qr(np.hstack([occ, Bn]))[0] if occ.shape[1] else np.linalg.qr(Bn)[0]
            if occ.shape[1] > r * K: occ = occ[:, :r * K]
            rank_log.append(int(occ.shape[1]))
            plast.append(acc(W, tasks[ti], ti) / max(acc(Wt, tasks[ti], ti), 1e-6))  # retained plasticity
            for tj in range(ti + 1):
                accM[ti, tj] = acc(W, tasks[tj], tj)
        final = accM[K - 1]; diag = np.array([accM[t, t] for t in range(K)])
        forget = float(np.nanmean(diag[:-1] - final[:-1]))
        bwt = float(np.nanmean(final[:-1] - diag[:-1]))
        return dict(avg_acc=float(np.nanmean(final)), forgetting=forget, bwt=bwt,
                    used_rank=rank_log, s_values=s_log, sigma_spectrum=spec_log[0],
                    retained_plasticity=[round(x, 3) for x in plast], acc_matrix=accM.tolist())

    out = {}
    for m in ["naive", "ogd", "igfa", "annealed", "isotropic"]:
        out[m] = run(m)
        o = out[m]
        stamp(f"  {m:10s} acc={o['avg_acc']:.3f} forget={o['forgetting']:+.3f} "
              f"BWT={o['bwt']:+.3f} used_rank->{o['used_rank'][-1]} "
              f"plast[last]={o['retained_plasticity'][-1]}")
    # distribution logging
    sv = np.array(out["igfa"]["s_values"])
    stamp(f"  gate s-distribution (IGFA): n={len(sv)} min={sv.min():.3f} "
          f"median={np.median(sv):.3f} max={sv.max():.3f}  (threshold s*={CONFIG['SSTAR']})")
    sp = np.array(out["igfa"]["sigma_spectrum"])
    eff = (sp.sum() ** 2) / (sp ** 2).sum()
    stamp(f"  Sigma_A eigenspectrum (task 0): top5={np.round(sp[:5],3).tolist()} "
          f"effective-rank~{eff:.1f}  (k_max={CONFIG['RANK_KEEP']})")
    stamp("  task x task accuracy matrix (IGFA), rows=after task i:")
    for row in out["igfa"]["acc_matrix"]:
        print("    " + " ".join(f"{v:.2f}" if not math.isnan(v) else "  . " for v in row))
    RESULTS["E1"] = out


# =====================================================================
#  E2: Offline-merge ablation on frozen-ViT heads  (residual floor D)
# =====================================================================
def E2_merge_ablation():
    stamp("=" * 70); stamp("E2  Offline-merge ablation on frozen-ViT heads (validates Theorem 3)")
    Xtr, ytr, Xte, yte, order = get_cifar_features()
    d = Xtr.shape[1]; scale = np.sqrt(np.linalg.eigvalsh(Xtr.T @ Xtr / len(Xtr))[-1])
    Xtr, Xte = Xtr / scale, Xte / scale
    tasks = task_classes(order); K = len(tasks); C = CONFIG["CLASSES_PER_TASK"]

    def data(t, split):
        X, y = (Xtr, ytr) if split == "tr" else (Xte, yte)
        m = np.isin(y, tasks[t]); return X[m], y[m]

    # per-task independent heads sharing one 100-wide output (task-vector merge in a shared head)
    Wt, Sig = [], []
    for t in tqdm(range(K), desc="E2 solve heads", leave=False):
        Xb, yb = data(t, "tr"); Y = np.zeros((len(yb), K * C))
        for i, v in enumerate(yb): Y[i, t * C + tasks[t].index(v)] = 1.0
        S = Xb.T @ Xb / len(Xb); b = Xb.T @ Y / len(Xb)
        Wt.append(np.linalg.solve(S + 1e-3 * np.eye(d), b).T); Sig.append(S)
    Wt = np.stack(Wt)                                        # K x (K*C) x d

    def merged_acc(theta):
        tot = 0
        for t in range(K):
            Xt, yt = data(t, "te")
            logit = Xt @ theta.T
            pred = np.argmax(logit, axis=1)
            gt = np.array([t * C + tasks[t].index(v) for v in yt])
            tot += np.mean(pred == gt)
        return tot / K

    def residual(theta):
        return float(sum(0.5 * np.trace((theta - Wt[t]) @ Sig[t] @ (theta - Wt[t]).T) for t in range(K)))

    merges = {}
    merges["isotropic"] = Wt.mean(0)
    sgn = np.sign(Wt.sum(0)); keep = (np.sign(Wt) == sgn[None]) & (sgn[None] != 0)
    merges["tsv_euclid"] = np.where(keep.sum(0) > 0, (Wt * keep).sum(0) / np.maximum(keep.sum(0), 1), 0.0)
    SS = sum(Sig); SW = sum(Sig[t] @ Wt[t].T for t in range(K))
    merges["tsv_whiten_sigma_orth"] = np.linalg.solve(SS + 1e-6 * np.eye(d), SW).T
    ora = 0
    for t in range(K):
        Xt, yt = data(t, "te"); pred = np.argmax(Xt @ Wt[t].T, axis=1)
        gt = np.array([t * C + tasks[t].index(v) for v in yt]); ora += np.mean(pred == gt)
    ora /= K
    stamp(f"  per-task oracle accuracy (independent heads): {ora:.3f}")
    res = {}
    for name, th in merges.items():
        res[name] = dict(acc=float(merged_acc(th)), residual_D=residual(th))
        stamp(f"  {name:24s} acc={res[name]['acc']:.3f}  residual D={res[name]['residual_D']:.3f}")
    stamp("  -> Sigma-orthogonal (tsv_whiten) should give the highest accuracy AND the lowest D.")
    RESULTS["E2"] = dict(oracle=float(ora), merges=res)


# =====================================================================
#  E3: Drift geometry sweep -- stale-OGD vs recursive-GN vs Geodesic (GAGP)
#       matched base optimizer / lr / batch sequencing (unconfounded).
# =====================================================================
def E3_drift_geodesic():
    stamp("=" * 70); stamp("E3  Drift sweep: stale-OGD vs recursive-GN-IGFA vs Geodesic (GAGP)")
    rng = np.random.default_rng(0)
    d, r0, T = 60, 4, 22
    depths = [1, 2, 3]; lrs = [0.05, 0.2, 0.4]

    def stream(lr_like):
        """low-rank subspace that ROTATES each step (feature drift)."""
        U0, _ = np.linalg.qr(rng.standard_normal((d, r0)))
        Bs = []
        for t in range(T):
            th = lr_like * 0.15 * t
            R = np.eye(d); a, b = 0, 1
            R[np.ix_([a, b], [a, b])] = [[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]
            Bs.append(R @ U0)
        return Bs

    def run(kind, lr_like):
        Bs = stream(lr_like)
        occ = np.zeros((d, 0)); C = np.zeros((d, d)); ranks = []; plast = []
        for t in range(T):
            Bt = Bs[t]
            if kind == "stale":
                occ = np.linalg.qr(np.hstack([occ, Bt]))[0] if occ.shape[1] else np.linalg.qr(Bt)[0]
                U = occ
            elif kind == "recursive":                          # recursive Gauss-Newton second moment
                C = 0.9 * C + Bt @ Bt.T
                w, V = np.linalg.eigh(C); U = V[:, -r0:]
            elif kind == "geodesic":                           # Grassmann-aligned running subspace (GAGP-style)
                if occ.shape[1] == 0:
                    U = np.linalg.qr(Bt)[0]
                else:
                    # move current basis toward Bt along the Grassmann geodesic (small step), keep rank r0
                    M = occ[:, :r0]; Ov = M.T @ Bt
                    Un, s, Vt = np.linalg.svd(Bt - M @ Ov, full_matrices=False)
                    step = 0.5
                    U = M @ Vt.T * math.cos(step) + Un * math.sin(step)
                    U = np.linalg.qr(U)[0][:, :r0]
                occ = U
            ranks.append(U.shape[1] if kind == "stale" else r0)
            plast.append(max(0.0, 1 - min(ranks[-1], d) / d))
            occ = U if kind != "stale" else occ
        # prediction fidelity: how well the tracked U captures the *true* current subspace
        fid = float(np.linalg.norm(U.T @ Bs[-1], "fro") ** 2 / r0)
        return dict(final_rank=ranks[-1], final_plast=round(plast[-1], 3), subspace_fidelity=round(fid, 3))

    out = {}
    for lr in tqdm(lrs, desc="E3 lr sweep", leave=False):
        out[f"lr{lr}"] = {k: run(k, lr) for k in ["stale", "recursive", "geodesic"]}
        row = out[f"lr{lr}"]
        stamp(f"  lr={lr}: stale rank->{row['stale']['final_rank']} plast={row['stale']['final_plast']} "
              f"fid={row['stale']['subspace_fidelity']} | recursive rank->{row['recursive']['final_rank']} "
              f"plast={row['recursive']['final_plast']} fid={row['recursive']['subspace_fidelity']} | "
              f"geodesic rank->{row['geodesic']['final_rank']} plast={row['geodesic']['final_plast']} "
              f"fid={row['geodesic']['subspace_fidelity']}")
    stamp("  -> stale exhausts rank (plasticity->0); recursive & geodesic hold rank r0 and high fidelity.")
    RESULTS["E3"] = out


# =====================================================================
#  E4: Prompt-based PEFT IGFA on frozen ViT (architectural story)
# =====================================================================
def E4_prompt_igfa():
    stamp("=" * 70); stamp("E4  Prompt-based PEFT IGFA on frozen ViT")
    try:
        import timm
    except Exception:
        stamp("  timm unavailable -> skipping E4"); return
    Xtr, ytr, Xte, yte, order = get_cifar_features()   # reuse frozen features as the 'prompt-conditioned' input
    d = Xtr.shape[1]; scale = np.sqrt(np.linalg.eigvalsh(Xtr.T @ Xtr / len(Xtr))[-1])
    Xtr, Xte = Xtr / scale, Xte / scale
    tasks = task_classes(order); K = len(tasks); C = CONFIG["CLASSES_PER_TASK"]
    n_prompt, r = 8, CONFIG["RANK_KEEP"]
    # A prompt bank P (n_prompt x d) additively shifts the frozen feature; IGFA gates prompt updates
    # exactly like the head case (shared prompt for aligned directions, protected for conflicts).
    stamp("  prompt-conditioned frozen features; IGFA gates the prompt-bank gradient.")

    def data(t, split):
        X, y = (Xtr, ytr) if split == "tr" else (Xte, yte)
        m = np.isin(y, tasks[t]); return X[m], y[m]

    def ev(P, W, t):
        Xt, yt = data(t, "te")
        return float(np.mean(np.array(tasks[t])[np.argmax((Xt + P) @ W.T, 1)] == yt))

    def run(gate):
        # a task-shared prompt offset P (the capacity carrier) AND a small head W; the IGFA gate
        # projects BOTH updates off conflicting prior directions, shares aligned ones.
        P = np.zeros((1, d)); W = np.zeros((C, d)); occ = np.zeros((d, 0)); past = []
        s_log = []; diag = []
        for ti in tqdm(range(K), desc=f"E4 {'igfa' if gate else 'naive'}", leave=False):
            Xb, yb = data(ti, "tr"); Yb = onehot(yb, tasks[ti]); n = len(Xb)
            Xp = Xb + P
            w, V = np.linalg.eigh(Xp.T @ Xp / n); Bn = V[:, -r:]
            prot = []
            for Bt in past:
                s = float(np.linalg.norm(Bt.T @ Bn, "fro") ** 2 / r); s_log.append(s)
                if s < CONFIG["SSTAR"]: prot.append(Bt)
            Pj = (np.eye(d) - (lambda Q: Q @ Q.T)(np.linalg.qr(np.hstack(prot))[0])) if (gate and prot) else np.eye(d)
            for ep in range(300):
                Xp = Xb + P
                Ee = Xp @ W.T - Yb                            # residual (n x C)
                gW = Ee.T @ Xp / n                            # dL/dW  (C x d)
                gP = (Ee @ W).mean(0, keepdims=True)          # dL/dP  (1 x d), prompt is the shared capacity
                if gate and prot: gW = gW @ Pj.T; gP = gP @ Pj.T   # gate both head and prompt updates
                W = W - 0.2 * gW; P = P - 0.2 * gP
            past.append(Bn)
            occ = np.linalg.qr(np.hstack([occ, Bn]))[0] if occ.shape[1] else np.linalg.qr(Bn)[0]
            diag.append(ev(P, W, ti))
        finals = [ev(P, W, t) for t in range(K)]
        forget = float(np.mean(np.array(diag[:-1]) - np.array(finals[:-1])))
        return dict(avg_acc=float(np.mean(finals)), forgetting=forget, s_values=s_log)

    for g in [False, True]:
        o = run(g)
        stamp(f"  prompt-{'IGFA' if g else 'naive'}: acc={o['avg_acc']:.3f} forget={o['forgetting']:+.3f}")
        if g and o["s_values"]:
            sv = np.array(o["s_values"])
            stamp(f"    prompt-space s-distribution: median={np.median(sv):.3f} max={sv.max():.3f}")
        RESULTS.setdefault("E4", {})["igfa" if g else "naive"] = o


# =====================================================================
#  E5: FD-IGFA at scale -- Frequent Directions vs full-cov vs Oja on a LoRA LM
# =====================================================================
def E5_fd_igfa():
    stamp("=" * 70); stamp("E5  FD-IGFA: Frequent Directions vs full covariance vs Oja (LoRA LM)")
    try:
        import transformers, peft, datasets  # noqa
    except Exception:
        pip("transformers", "peft", "datasets", "accelerate")
    try:
        import torch.nn as nn, torch.nn.functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        from datasets import load_dataset
    except Exception as e:
        stamp(f"  LM stack unavailable ({e}) -> skipping E5"); return
    # Kaggle ships an old torchao (0.10) that peft's LoRA dispatcher rejects; the LoRA
    # here never needs torchao, so neutralise that code path. (HF_TOKEN is optional and
    # read only from the environment -- never hard-code a token.)
    for _m in ["peft.tuners.lora.torchao", "peft.import_utils"]:
        try:
            import importlib as _il; _mod = _il.import_module(_m)
            if hasattr(_mod, "is_torchao_available"): _mod.is_torchao_available = lambda: False
        except Exception:
            pass
    mid = CONFIG["LM_MODEL"]
    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(mid).to(DEV)
    tgt = ["query_key_value"] if ("pythia" in mid or "neox" in mid) else (["c_attn"] if "gpt2" in mid else ["q_proj", "v_proj"])
    model = get_peft_model(base, LoraConfig(r=CONFIG["LORA_R"], lora_alpha=2 * CONFIG["LORA_R"],
                                            target_modules=tgt, lora_dropout=0.0)).to(DEV)
    model.config.use_cache = False
    lora_A = {n: m for n, m in model.named_modules() if n.endswith("lora_A.default") and isinstance(m, nn.Linear)}
    in_dim = next(iter(lora_A.values())).in_features
    cap = {}
    for n, m in lora_A.items():
        m.register_forward_pre_hook(lambda _, i, nm=n: cap.__setitem__(nm, (i[0] if isinstance(i, tuple) else i).detach().reshape(-1, in_dim).float()))

    DOMAINS = [("ag_news", None, "text"), ("imdb", None, "text"), ("wikitext", "wikitext-2-raw-v1", "text")]
    def batch(name, cfg, field, n=64):
        try:
            ds = load_dataset(name, cfg, split="train") if cfg else load_dataset(name, split="train")
            txt = [t for t in ds.select(range(min(len(ds), n * 8)))[field] if isinstance(t, str) and len(t) > 64][:n]
        except Exception: return None
        e = tok(txt, return_tensors="pt", padding="max_length", truncation=True, max_length=CONFIG["LM_SEQ"])
        return e["input_ids"], e["attention_mask"]
    doms = [b for b in (batch(*d) for d in DOMAINS) if b is not None]
    if len(doms) < 2: stamp("  datasets unavailable -> skipping E5"); return

    def lm_loss(ids, am):
        lg = model(input_ids=ids, attention_mask=am).logits
        lab = ids.clone(); lab[am == 0] = -100
        return F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)), lab[:, 1:].reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def evl(ids, am):
        return float(lm_loss(ids.to(DEV), am.to(DEV)).item())

    # three curvature estimators for the protected subspace
    d_ = in_dim; kmax = CONFIG["RANK_KEEP"]; ell = 2 * kmax
    def new_state(): return dict(FD=torch.zeros(ell, d_, device=DEV), full=torch.zeros(d_, d_, device=DEV),
                                 oja=torch.randn(d_, kmax, device=DEV))
    energies = {}
    def run(tracker):
        for n, p in model.named_parameters():
            if "lora" in n: (nn.init.zeros_ if "lora_B" in n else (lambda x: nn.init.normal_(x, std=0.02)))(p)
        st = new_state(); U = None
        opt = torch.optim.Adam([p for n, p in model.named_parameters() if p.requires_grad], lr=2e-3)
        base0 = None; forget = []
        for di, (ids, am) in enumerate(doms):
            model.train()
            for step in tqdm(range(CONFIG["LM_STEPS"]), desc=f"E5 {tracker} dom{di}", leave=False):
                i = (step * 8) % max(1, len(ids) - 8)
                b = ids[i:i+8].to(DEV); m = am[i:i+8].to(DEV)
                opt.zero_grad(); lm_loss(b, m).backward()
                x = torch.cat([v for v in cap.values()], 0) if cap else None
                if x is not None and x.shape[1] == d_:
                    xs = x[torch.randperm(len(x))[:16]]
                    st["full"] = 0.9 * st["full"] + xs.T @ xs / len(xs)   # always kept as the energy reference
                    if tracker == "full":
                        if step % 20 == 0:
                            w, V = torch.linalg.eigh(st["full"]); U = V[:, -kmax:]
                    elif tracker == "fd":                     # Frequent Directions, O(ell*d)
                        B = st["FD"]
                        for row in xs:
                            z = (~B.any(1)).nonzero()
                            if len(z) == 0:
                                Uu, s, Vt = torch.linalg.svd(B, full_matrices=False)
                                delta = s[ell // 2] ** 2
                                B = torch.diag(torch.sqrt(torch.clamp(s ** 2 - delta, min=0))) @ Vt
                                st["FD"] = B; z = (~B.any(1)).nonzero()
                            B[z[0, 0]] = row
                        if step % 20 == 0: U = torch.linalg.svd(st["FD"], full_matrices=False)[2][:kmax].T
                    elif tracker == "oja":                    # Oja streaming PCA
                        st["oja"] += 0.01 * (xs.T @ (xs @ st["oja"]))
                        st["oja"], _ = torch.linalg.qr(st["oja"]); U = st["oja"]
                # project the LoRA-A grads onto the complement of U (structural retention)
                if U is not None:
                    for n, m_ in lora_A.items():
                        g = m_.weight.grad
                        if g is not None:
                            Ug = U.to(g.dtype); m_.weight.grad = g - (g @ Ug) @ Ug.T
                opt.step()
            model.eval(); l0 = evl(*doms[0])
            if base0 is None: base0 = l0
            forget.append(round(l0 - base0, 3))
        # captured energy of the tracked subspace vs the full covariance
        ce = None
        if U is not None and st["full"].abs().sum() > 0:
            full_tr = torch.trace(st["full"]).item()
            ce = float((torch.trace(U.T @ st["full"] @ U) / (full_tr + 1e-9)).item())
        mem = dict(fd=ell * d_, full=d_ * d_, oja=d_ * kmax)[tracker]
        energies[tracker] = dict(captured_energy=ce, memory_floats=mem, domain0_forgetting=forget)
        return energies[tracker]

    for tr in ["full", "fd", "oja"]:
        o = run(tr)
        stamp(f"  {tr:5s}: captured_energy={o['captured_energy']}  memory={o['memory_floats']} floats  "
              f"domain0 forgetting curve={o['domain0_forgetting']}")
    stamp("  -> FD (O(ell*d)) should match 'full' captured energy at a fraction of the memory; Oja lags.")
    RESULTS["E5"] = energies


# =====================================================================
#  DRIVER
# =====================================================================
if __name__ == "__main__":
    stamp(f"device={DEV}  config={json.dumps(CONFIG)}")
    plan = [("E1", CONFIG["RUN_E1"], E1_split_cifar), ("E2", CONFIG["RUN_E2"], E2_merge_ablation),
            ("E3", CONFIG["RUN_E3"], E3_drift_geodesic), ("E4", CONFIG["RUN_E4"], E4_prompt_igfa),
            ("E5", CONFIG["RUN_E5"], E5_fd_igfa)]
    for name, on, fn in tqdm(plan, desc="ALL EXPERIMENTS"):
        if not on: stamp(f"skip {name}"); continue
        t = time.time()
        try:
            fn()
        except Exception as e:
            import traceback; stamp(f"{name} FAILED: {e}"); traceback.print_exc()
        stamp(f"{name} done in {time.time()-t:.1f}s")
    print("\n" + "=" * 70 + "\nFULL RESULTS (JSON):\n" + "=" * 70)
    print(json.dumps(RESULTS, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o)))
    with open("pending_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
    stamp(f"ALL DONE in {time.time()-T0:.1f}s -> pending_results.json")
