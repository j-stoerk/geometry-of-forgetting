# =====================================================================
#  kaggle_vit_multiseed.py  --  ALL pending frozen-ViT experiments, multi-seed.
#
#  Designed for a Kaggle notebook with 2x T4 GPUs (falls back to 1 GPU / CPU).
#  Feature extraction is parallelized across both GPUs; the continual-learning
#  runs then operate on cached features and are fast.
#
#  Experiments (each over SEEDS>=5, mean +/- 95% CI, paired t-tests):
#    Y1  Split-CIFAR-100  (10 tasks x 10 classes)   naive/EWC/replay/OGD/igfa
#    Y2  ImageNet-R       (10 tasks x 20 classes)   same methods
#    Y3  CUB-200-2011     (10 tasks x 20 classes)   same methods
#    Y4  CIFAR-100 SUPERCLASS STREAM (5 tasks, 20-way coarse labels, disjoint
#        fine classes per task): a realistic SIMILAR-task benchmark -- the
#        regime where the signed gate should beat OGD (cf. Rotated-Digits).
#    Y5  Offline-merge ablation on ViT features (isotropic / TSV sign-elect /
#        Sigma-orthogonal) with residual floor D, multi-seed.
#    Y6  prompt-igfa vs prompt-naive on Split-CIFAR-100 (multi-seed; heavy --
#        full ViT forwards; seeds are split across the two GPUs).
#
#  Prints extensive results (better too much than too little) with tqdm ETAs,
#  and dumps everything to vit_multiseed_results.json.
# =====================================================================
import os, io, json, time, tarfile, pickle, threading, urllib.request
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import torch, torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm
from scipy import stats

# ------------------------------- CONFIG -------------------------------
SEEDS       = [0, 1, 2, 3, 4]          # >=5 seeds everywhere
SSTAR       = 0.60                     # gate threshold (validated on frozen-ViT E1)
RANK        = 12                       # per-task protected-subspace rank
EPOCHS      = 4                        # linear-head epochs per task
BS          = 256
LR          = 1e-3
EWC_LAM     = 5.0                      # with unit-mean-normalized Fisher (tuned once, Y1 seed 0)
REPLAY_PER_CLASS = 10
RUN = dict(Y1=True, Y2=True, Y3=True, Y4=True, Y5=True, Y6=True)
PROMPT_TOKENS, PROMPT_EPOCHS, PROMPT_BS = 8, 2, 64
DATA, CACHE = "./data", "./feat_cache"
os.makedirs(DATA, exist_ok=True); os.makedirs(CACHE, exist_ok=True)

DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
DEV0 = DEVICES[0]
T0 = time.time()
def log(msg): print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)
log(f"devices: {DEVICES}")

# ------------------------- dataset acquisition ------------------------
def download(url, dst):
    if os.path.exists(dst): log(f"cached: {dst}"); return dst
    log(f"downloading {url}")
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(dst)) as p:
            while True:
                chunk = r.read(1 << 20)
                if not chunk: break
                f.write(chunk); p.update(len(chunk))
    return dst

def get_cifar100():
    import torchvision
    tr = torchvision.datasets.CIFAR100(DATA, train=True,  download=True)
    te = torchvision.datasets.CIFAR100(DATA, train=False, download=True)
    def coarse(split):  # torchvision only exposes fine labels; read coarse from the pickle
        with open(os.path.join(DATA, "cifar-100-python", split), "rb") as f:
            d = pickle.load(f, encoding="bytes")
        return np.array(d[b"coarse_labels"])
    return (tr.data, np.array(tr.targets), coarse("train"),
            te.data, np.array(te.targets), coarse("test"))

def get_imagenet_r():
    root = os.path.join(DATA, "imagenet-r")
    if not os.path.isdir(root):
        tar = download("https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar",
                       os.path.join(DATA, "imagenet-r.tar"))
        log("extracting imagenet-r.tar")
        with tarfile.open(tar) as t:
            try: t.extractall(DATA, filter="data")
            except TypeError: t.extractall(DATA)
    wnids = sorted(w for w in os.listdir(root)              # skip README.txt etc.
                   if os.path.isdir(os.path.join(root, w)))
    files, labels = [], []
    for c, w in enumerate(wnids):
        for fn in sorted(os.listdir(os.path.join(root, w))):
            files.append(os.path.join(root, w, fn)); labels.append(c)
    files, labels = np.array(files), np.array(labels)
    rng = np.random.default_rng(0)                    # fixed 80/20 split per class
    tr_m = np.zeros(len(files), bool)
    for c in range(len(wnids)):
        idx = np.where(labels == c)[0]; rng.shuffle(idx)
        tr_m[idx[: int(0.8 * len(idx))]] = True
    return files[tr_m], labels[tr_m], files[~tr_m], labels[~tr_m], len(wnids)

def get_cub():
    root = os.path.join(DATA, "CUB_200_2011")
    if not os.path.isdir(root):
        tgz = download("https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz",
                       os.path.join(DATA, "cub.tgz"))
        log("extracting cub.tgz")
        with tarfile.open(tgz) as t:
            try: t.extractall(DATA, filter="data")
            except TypeError: t.extractall(DATA)
    rd = lambda fn: [l.split() for l in open(os.path.join(root, fn))]
    paths = {i: p for i, p in rd("images.txt")}
    lab   = {i: int(c) - 1 for i, c in rd("image_class_labels.txt")}
    is_tr = {i: s == "1" for i, s in rd("train_test_split.txt")}
    ids = sorted(paths)
    f = np.array([os.path.join(root, "images", paths[i]) for i in ids])
    y = np.array([lab[i] for i in ids]); m = np.array([is_tr[i] for i in ids])
    return f[m], y[m], f[~m], y[~m], 200

# ------------------------- feature extraction --------------------------
def make_vit(device):
    import timm
    m = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    return m.eval().to(device)

def make_transform():
    import timm
    from timm.data import resolve_data_config, create_transform
    cfg = resolve_data_config({}, model="vit_base_patch16_224")
    return create_transform(**cfg)

class ImgSet(torch.utils.data.Dataset):
    def __init__(self, items, tf):  # items: file paths or HWC uint8 arrays
        self.items, self.tf = items, tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        from PIL import Image
        it = self.items[i]
        img = Image.open(it).convert("RGB") if isinstance(it, str) else Image.fromarray(it)
        return self.tf(img)

@torch.no_grad()
def extract(items, device, tag):
    tf = make_transform(); model = make_vit(device)
    dl = torch.utils.data.DataLoader(ImgSet(items, tf), batch_size=128,
                                     num_workers=2, pin_memory=True)
    out = []
    for xb in tqdm(dl, desc=f"{tag}@{device}", leave=False):
        with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            out.append(model(xb.to(device)).float().cpu())
    del model; torch.cuda.empty_cache()
    return torch.cat(out).numpy()

def cached_features(name, builder):
    """builder() -> (train_items, ytr, test_items, yte, C [, coarse_tr, coarse_te])"""
    fn = os.path.join(CACHE, f"{name}.npz")
    if os.path.exists(fn):
        z = np.load(fn); log(f"features cached: {name}")
        return dict(z)
    parts = builder()
    (tr_items, ytr, te_items, yte, C), extra = parts[:5], parts[5:]
    # split the (larger) train extraction across both GPUs
    if len(DEVICES) >= 2 and len(tr_items) > 2000:
        h = len(tr_items) // 2
        with ThreadPoolExecutor(2) as ex:
            a = ex.submit(extract, tr_items[:h], DEVICES[0], f"{name}-trA")
            b = ex.submit(extract, tr_items[h:], DEVICES[1], f"{name}-trB")
            Xtr = np.concatenate([a.result(), b.result()])
        Xte = extract(te_items, DEVICES[0], f"{name}-te")
    else:
        Xtr = extract(tr_items, DEVICES[0], f"{name}-tr")
        Xte = extract(te_items, DEVICES[0], f"{name}-te")
    d = dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, C=np.array(C))
    if extra: d["ctr"], d["cte"] = extra
    np.savez_compressed(fn, **d); log(f"features saved: {name} {Xtr.shape}")
    return d

# ---------------------- continual learning on features -----------------
def task_basis(X, r=RANK):
    Xc = torch.as_tensor(X, dtype=torch.float32, device=DEV0)
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    return Vh[:r].T.contiguous()                              # d x r

def overlap(Ba, Bb, r=RANK):
    return float((Ba.T @ Bb).square().sum().item()) / r

def run_stream(feats, seed, method, superclass=False, verbose=False):
    """Task-incremental linear head on frozen features. Returns (avg_acc, forgetting, sims)."""
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    Xtr, Xte = feats["Xtr"], feats["Xte"]
    if superclass:                                            # Y4: 20-way coarse, 5 tasks of
        yc_tr, yc_te = feats["ctr"], feats["cte"]             # disjoint fine classes
        fine_tr, fine_te = feats["ytr"], feats["yte"]
        assign = {}                                           # fine class -> task
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
        masks = [torch.as_tensor(np.isin(np.arange(C), cs), device=DEV0) for cs in cls_tasks]
    T = len(tr_tasks)
    W = torch.zeros(C, Xtr.shape[1], device=DEV0, requires_grad=True)
    opt = torch.optim.Adam([W], lr=LR)
    bases, Q, fisher, Wstar, buf_x, buf_y = [], None, None, None, [], []
    acc_after, sims = np.zeros(T), []

    def task_acc(t):
        X = torch.as_tensor(Xte[te_tasks[t]], dtype=torch.float32, device=DEV0)
        logit = X @ W.T
        if masks[t] is not None: logit = logit.masked_fill(~masks[t], -1e9)
        return float((logit.argmax(1).cpu().numpy() == yte[te_tasks[t]]).mean())

    for t in range(T):
        Bn = task_basis(Xtr[tr_tasks[t]])
        if method in ("ogd", "igfa") and bases:
            ovl = [overlap(B, Bn) for B in bases]
            sims.extend(ovl)
            prot = bases if method == "ogd" else [B for B, o in zip(bases, ovl) if o < SSTAR]
            Q = torch.linalg.qr(torch.cat(prot, 1))[0] if prot else None
            if verbose: log(f"    t{t} overlaps={np.round(ovl,2)} protecting {len(prot)}/{len(bases)}")
        X = torch.as_tensor(Xtr[tr_tasks[t]], dtype=torch.float32, device=DEV0)
        y = torch.as_tensor(ytr[tr_tasks[t]], dtype=torch.long, device=DEV0)
        for _ in range(EPOCHS):
            for i in torch.randperm(len(X), device=DEV0).split(BS):
                xb, yb = X[i], y[i]
                if method == "replay" and buf_x:
                    bx = torch.cat(buf_x); by = torch.cat(buf_y)
                    j = torch.randint(0, len(bx), (min(len(i), len(bx)),), device=DEV0)
                    xb, yb = torch.cat([xb, bx[j]]), torch.cat([yb, by[j]])
                logit = xb @ W.T
                if masks[t] is not None and method != "replay":
                    logit = logit.masked_fill(~masks[t], -1e9)
                loss = F.cross_entropy(logit, yb)
                if method == "ewc" and fisher is not None:
                    loss = loss + EWC_LAM * (fisher * (W - Wstar) ** 2).sum()
                opt.zero_grad(); loss.backward()
                if method in ("ogd", "igfa") and Q is not None:
                    W.grad -= (W.grad @ Q) @ Q.T              # remove protected feature directions
                opt.step()
        bases.append(Bn)
        if method == "ewc":
            g2 = torch.zeros_like(W)
            for i in torch.arange(len(X), device=DEV0).split(BS):
                logit = X[i] @ W.T
                if masks[t] is not None: logit = logit.masked_fill(~masks[t], -1e9)
                l = F.cross_entropy(logit, y[i]); W.grad = None; l.backward()
                g2 += W.grad ** 2
            g2 = g2 / g2.mean().clamp(min=1e-12)          # unit-mean Fisher so EWC_LAM is meaningful
            fisher = (fisher + g2) if fisher is not None else g2
            Wstar = W.detach().clone()
        if method == "replay":
            for c in np.unique(ytr[tr_tasks[t]]):
                k = np.where(ytr[tr_tasks[t]] == c)[0][:REPLAY_PER_CLASS]
                buf_x.append(X[k].detach()); buf_y.append(y[k].detach())
        acc_after[t] = task_acc(t)
    final = np.array([task_acc(t) for t in range(T)])
    return final.mean(), float((acc_after - final)[:-1].mean()), sims

# ----------------------------- statistics ------------------------------
def ci95(a):
    a = np.asarray(a, float)
    return a.mean(), stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))

def report(tag, per_method, results):
    print(f"\n===== {tag}: mean +/- 95% CI over {len(SEEDS)} seeds =====")
    for m, vals in per_method.items():
        am, ah = ci95([v[0] for v in vals]); fm, fh = ci95([v[1] for v in vals])
        print(f"  {m:8s}  acc {am:.3f}+/-{ah:.3f}   forget {fm:.3f}+/-{fh:.3f}")
        results[f"{tag}/{m}"] = dict(acc=[v[0] for v in vals], forget=[v[1] for v in vals])
    for a, b in [("igfa", "ogd"), ("igfa", "naive"), ("igfa", "replay"),
                 ("prompt-igfa", "prompt-naive")]:
        if a in per_method and b in per_method:
            xa = [v[0] for v in per_method[a]]; xb = [v[0] for v in per_method[b]]
            p = 1.0 if np.allclose(xa, xb) else stats.ttest_rel(xa, xb).pvalue
            print(f"  paired t-test acc {a} vs {b}: p={p:.4f}"
                  + ("  (identical runs)" if np.allclose(xa, xb) else ""))
    with open("vit_multiseed_results.json", "w") as f:      # incremental dump: results
        json.dump(results, f, indent=1)                     # survive a lost kernel

# ------------------------- Y5: merge ablation ---------------------------
def merge_ablation(feats, seed):
    rng = np.random.default_rng(seed)
    Xtr, ytr, Xte, yte, C = feats["Xtr"], feats["ytr"], feats["Xte"], feats["yte"], int(feats["C"])
    order = rng.permutation(C); per = C // 10
    cls_tasks = [order[t * per:(t + 1) * per] for t in range(10)]
    d = Xtr.shape[1]; Ws, Ss = [], []
    for cs in cls_tasks:
        idx = np.where(np.isin(ytr, cs))[0]
        X = torch.as_tensor(Xtr[idx], dtype=torch.float32, device=DEV0)
        Y = F.one_hot(torch.as_tensor(ytr[idx], device=DEV0).long(), C).float()
        A = X.T @ X / len(X) + 1e-3 * torch.eye(d, device=DEV0)
        Ws.append(torch.linalg.solve(A, X.T @ Y / len(X)))    # d x C ridge head
        Ss.append(X.T @ X / len(X))
    def evaluate(Wm):
        acc = []
        for cs in cls_tasks:
            idx = np.where(np.isin(yte, cs))[0]
            X = torch.as_tensor(Xte[idx], dtype=torch.float32, device=DEV0)
            logit = (X @ Wm).masked_fill(
                ~torch.as_tensor(np.isin(np.arange(C), cs), device=DEV0), -1e9)
            acc.append(float((logit.argmax(1).cpu().numpy() == yte[idx]).mean()))
        return float(np.mean(acc))
    def floor(Wm):
        return float(sum(0.5 * torch.trace((Wm - Wt).T @ St @ (Wm - Wt)).item()
                         for Wt, St in zip(Ws, Ss)))
    iso = torch.stack(Ws).mean(0)
    sgn = torch.sign(torch.stack(Ws).sum(0))                  # TSV/TIES-style sign election
    stackd = torch.stack(Ws); keep = (torch.sign(stackd) == sgn) & (sgn != 0)
    ties = torch.where(keep.any(0), (stackd * keep).sum(0) / keep.sum(0).clamp(min=1), iso)
    Ssum = sum(Ss); rhs = sum(St @ Wt for St, Wt in zip(Ss, Ws))
    sig = torch.linalg.pinv(Ssum) @ rhs                       # Sigma-orthogonal merge (Thm 3)
    return {m: (evaluate(Wm), floor(Wm)) for m, Wm in
            [("isotropic", iso), ("tsv", ties), ("sigma_orth", sig)]}

# --------------------- Y6: prompt-igfa (full ViT) -----------------------
def prompt_forward(model, P, img):
    x = model.patch_embed(img)
    x = model._pos_embed(x)                                   # adds cls token + pos embed
    x = torch.cat([x[:, :1], P.unsqueeze(0).expand(len(x), -1, -1), x[:, 1:]], 1)
    for blk in model.blocks: x = blk(x)
    return model.norm(x)[:, 0]

def tlog(msg):
    """Thread-safe progress log for Y6: worker threads must NOT print to the notebook
    stream (ipykernel can deadlock on multi-threaded prints) -- they append to a file."""
    with open("y6_progress.log", "a") as f:
        f.write(f"[{time.time()-T0:7.1f}s] {msg}\n")

def prompt_stream(seed, method, device, cifar):
    from PIL import Image
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Xtr_img, ytr, Xte_img, yte = cifar
    model = make_vit(device); tf = make_transform()
    C = 100; order = rng.permutation(C)
    cls_tasks = [order[t * 10:(t + 1) * 10] for t in range(10)]
    P = torch.zeros(PROMPT_TOKENS, 768, device=device, requires_grad=True)
    W = torch.zeros(C, 768, device=device, requires_grad=True)
    opt = torch.optim.Adam([P, W], lr=1e-2)
    bases, Q, acc_after = [], None, np.zeros(10)
    # NO DataLoader here: multiprocessing workers inside threads leak + respawn each
    # epoch (slow, noisy destructors). Transform each task once, then slice tensors.
    tensorize = lambda items: torch.stack([tf(Image.fromarray(x)) for x in items])
    def batches(X, y, shuffle):
        idx = torch.randperm(len(X)) if shuffle else torch.arange(len(X))
        for i in idx.split(PROMPT_BS):
            yield X[i].to(device, non_blocking=True), y[i].to(device)
    @torch.no_grad()
    def task_acc(t):
        idx = np.where(np.isin(yte, cls_tasks[t]))[0][:500]   # 500-image eval subsample
        Xe = tensorize(Xte_img[idx]); ye = torch.as_tensor(yte[idx]).long()
        mask = ~torch.as_tensor(np.isin(np.arange(C), cls_tasks[t]), device=device)
        good = 0
        for xb, yb in batches(Xe, ye, False):
            with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
                f = prompt_forward(model, P, xb).float()
            good += int(((f @ W.T).masked_fill(mask, -1e9).argmax(1) == yb).sum())
        return good / len(idx)
    for t in range(10):
        idx = np.where(np.isin(ytr, cls_tasks[t]))[0][:2000]  # 2000-image train subsample
        Xt = tensorize(Xtr_img[idx]); yt = torch.as_tensor(ytr[idx]).long()
        mask = ~torch.as_tensor(np.isin(np.arange(C), cls_tasks[t]), device=device)
        grads = []
        for _ in range(PROMPT_EPOCHS):
            for xb, yb in batches(Xt, yt, True):
                with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
                    f = prompt_forward(model, P, xb).float()
                loss = F.cross_entropy((f @ W.T).masked_fill(mask, -1e9), yb)
                opt.zero_grad(); loss.backward()
                if method == "igfa" and Q is not None:
                    P.grad -= (P.grad @ Q) @ Q.T              # gate in prompt-embedding space
                grads.append(P.grad.detach().cpu())
                opt.step()
        G = torch.cat(grads[-50:]).to(device)                 # last-50-step gradient rows
        _, _, Vh = torch.linalg.svd(G, full_matrices=False)
        Bn = Vh[:RANK].T
        if method == "igfa":
            prot = [B for B in bases
                    if (B.T @ Bn).square().sum().item() / RANK < SSTAR]
            Q = torch.linalg.qr(torch.cat(prot, 1))[0] if prot else None
        bases.append(Bn)
        acc_after[t] = task_acc(t)
        tlog(f"prompt-{method} seed{seed} task{t} done @{device} (acc {acc_after[t]:.3f})")
    tlog(f"prompt-{method} seed{seed}: final re-evaluation of all 10 tasks @{device} (~2 min)")
    final = np.array([task_acc(t) for t in range(10)])
    tlog(f"prompt-{method} seed{seed} FINISHED @{device} "
         f"(final acc {final.mean():.3f}, forget {float((acc_after-final)[:-1].mean()):.3f})")
    del model; torch.cuda.empty_cache()
    return final.mean(), float((acc_after - final)[:-1].mean())

# ================================ MAIN =================================
if __name__ == "__main__":
    results, METHODS = {}, ["naive", "ewc", "replay", "ogd", "igfa"]

    log("stage 1/3: acquiring datasets + extracting ViT features (2-GPU parallel)")
    cif = get_cifar100()
    feats_c100 = cached_features("cifar100", lambda: (
        list(cif[0]), cif[1], list(cif[3]), cif[4], 100, cif[2], cif[5]))
    feats_inr = cached_features("imagenet_r", lambda: get_imagenet_r()) if RUN["Y2"] else None
    feats_cub = cached_features("cub", lambda: get_cub()) if RUN["Y3"] else None

    log("stage 2/3: continual-learning runs on cached features")
    for tag, feats, sup, on in [("Y1-SplitCIFAR100", feats_c100, False, RUN["Y1"]),
                                ("Y2-ImageNetR",     feats_inr,  False, RUN["Y2"]),
                                ("Y3-CUB",           feats_cub,  False, RUN["Y3"]),
                                ("Y4-C100-Superclass(similar-task)", feats_c100, True, RUN["Y4"])]:
        if not on: continue
        per = {m: [] for m in METHODS}
        for seed in tqdm(SEEDS, desc=tag):
            for m in METHODS:
                a, f, sims = run_stream(feats, seed, m, superclass=sup,
                                        verbose=(seed == 0 and m == "igfa"))
                per[m].append((a, f))
            if seed == 0 and sims:
                print(f"  {tag} pairwise subspace overlaps (igfa view): "
                      f"mean={np.mean(sims):.2f} min={np.min(sims):.2f} max={np.max(sims):.2f} "
                      f"(gate threshold {SSTAR})")
        report(tag, per, results)

    if RUN["Y5"]:
        log("Y5: offline-merge ablation on ViT features (Split-CIFAR-100 heads)")
        per = {m: [] for m in ["isotropic", "tsv", "sigma_orth"]}
        for seed in tqdm(SEEDS, desc="Y5-merge"):
            for m, (acc, D) in merge_ablation(feats_c100, seed).items():
                per[m].append((acc, D))
        print(f"\n===== Y5 merge ablation: mean +/- 95% CI over {len(SEEDS)} seeds =====")
        for m, vals in per.items():
            am, ah = ci95([v[0] for v in vals]); dm, dh = ci95([v[1] for v in vals])
            print(f"  {m:10s}  acc {am:.3f}+/-{ah:.3f}   residual D {dm:.2f}+/-{dh:.2f}")
            results[f"Y5/{m}"] = dict(acc=[v[0] for v in vals], D=[v[1] for v in vals])

    if RUN["Y6"]:
        log("stage 3/3: Y6 prompt-igfa vs prompt-naive (full ViT forwards, seeds split on 2 GPUs)")
        cifar_imgs = (cif[0], cif[1], cif[3], cif[4])   # keep uint8 (N,32,32,3) arrays
        per = {"prompt-naive": [], "prompt-igfa": []}
        from concurrent.futures import wait as fwait
        for m, key in [("naive", "prompt-naive"), ("igfa", "prompt-igfa")]:
            with ThreadPoolExecutor(len(DEVICES)) as ex:
                futs = {ex.submit(prompt_stream, s, m, DEVICES[i % len(DEVICES)], cifar_imgs): s
                        for i, s in enumerate(SEEDS)}
                pending = set(futs)
                while pending:                              # heartbeat from the MAIN thread only
                    done, pending = fwait(pending, timeout=120)
                    for f in done: f.result()               # surface worker exceptions
                    log(f"Y6 {key}: {len(SEEDS)-len(pending)}/{len(SEEDS)} seeds done "
                        f"(details: y6_progress.log)")
                per[key] = [f.result() for f in sorted(futs, key=lambda f: futs[f])]
        report("Y6-PromptCIFAR100", per, results)

    with open("vit_multiseed_results.json", "w") as f:
        json.dump(results, f, indent=1)
    log("ALL DONE -> vit_multiseed_results.json")
