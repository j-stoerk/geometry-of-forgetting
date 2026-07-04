# =====================================================================
#  kaggle_sign_gate_vit.py -- can the threshold-free functional sign
#  gate REPLACE the s*-threshold gate on the vision streams?
#
#  STAND-ALONE: no prior script needed.  If ./feat_cache/*.npz are
#  missing it downloads the datasets and extracts frozen ViT-B/16
#  features itself (CIFAR-100 via torchvision; ImageNet-R ~2 GB from
#  Berkeley; CUB ~1.1 GB from Caltech -- enable Internet in the Kaggle
#  notebook).  Caches are format-compatible with kaggle_vit_multiseed.py
#  so either script reuses the other's features.
#
#  Streams: Split-CIFAR-100, ImageNet-R, CUB (task-incremental, 10
#  tasks) and the CIFAR-100 superclass similar-task stream.  Methods:
#    naive       : plain Adam on the shared linear head
#    ogd         : protect-all (null-space of per-task feature bases)
#    igfa (s*)   : the paper's threshold gate (s* = 0.60)
#    func-sign   : threshold-free -- per step, sample one past task,
#                  compute grad L_u from a MICRO-CACHE (2 features per
#                  class), and remove the update's component only if it
#                  would increase L_u.  No threshold, no subspaces.
#  Prediction: func-sign inherits the ImageNet-R win without s*, ties
#  the exact-tie streams, and needs no per-stream calibration.
#  5 seeds, mean +/- std, paired t-tests.
#  ~15 min with cached features; +30-45 min first-run extraction (2xT4).
# =====================================================================
import os, json, time, pickle, tarfile, urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch, torch.nn.functional as F
from scipy import stats
from tqdm.auto import tqdm

SCRIPT_VERSION = "v2.0-standalone"

SEEDS, SSTAR, RANK, EPOCHS, BS, LR = [0, 1, 2, 3, 4], 0.60, 12, 4, 256, 1e-3
CACHE_PER_CLASS = 2
DATA, CACHE = "./data", "./feat_cache"
os.makedirs(DATA, exist_ok=True); os.makedirs(CACHE, exist_ok=True)

DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
DEV = DEVICES[0]
T0 = time.time()
def log(msg): print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


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
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
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


# ------------------------- the gate comparison -------------------------
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
    log(f"SCRIPT VERSION {SCRIPT_VERSION}")
    log(f"devices: {DEVICES}")

    log("stage 1/2: features (reuse cache or build)")
    cif = get_cifar100()
    feats_c100 = cached_features("cifar100", lambda: (
        list(cif[0]), cif[1], list(cif[3]), cif[4], 100, cif[2], cif[5]))
    feats_inr = cached_features("imagenet_r", get_imagenet_r)
    feats_cub = cached_features("cub", get_cub)

    log("stage 2/2: gate comparison, 5 seeds x 4 methods x 4 streams")
    results = {}
    streams = [("SplitCIFAR100", feats_c100, False), ("ImageNetR", feats_inr, False),
               ("CUB", feats_cub, False), ("C100-Superclass", feats_c100, True)]
    for tag, feats, sup in streams:
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
