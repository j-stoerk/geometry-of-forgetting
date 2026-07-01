# %% [markdown]
# # IGFA on Split-CIFAR-100 with a frozen Vision Transformer
#
# Companion notebook to *"Forgetting as Interference"*. Demonstrates Interference-
# Gated Functional Allocation (IGFA) in the **frozen-feature / PEFT regime** — the
# setting where the interference functional is *exact* (assumption A1).
#
# A frozen ViT-B/16 backbone provides features; a **single shared linear head** is
# trained sequentially over 10 tasks of Split-CIFAR-100 (10 classes each). We
# compare Naive, EWC, OGD/GPM, Experience Replay, and IGFA (signed gate, with an
# optional recursive Gauss-Newton subspace that tracks feature drift).
#
# **Run on Colab/Kaggle with a GPU.**  Runtime ~5-10 min (feature extraction
# dominates). The CL math is pure NumPy/Torch on cached features.
#
# To use LoRA instead of a frozen backbone, unfreeze LoRA adapters in the ViT and
# extract features per task after adapting; the head-level IGFA code below is
# unchanged. The frozen-feature version is the cleanest exact-A1 demonstration.

# %%
# !pip install -q timm torch torchvision

import numpy as np, torch, torch.nn as nn, timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
N_TASKS, CLS_PER_TASK = 10, 10
C = N_TASKS * CLS_PER_TASK

# %% [markdown]
# ## 1. Frozen ViT backbone and cached features

# %%
backbone = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
backbone.eval().to(DEVICE)
for p in backbone.parameters():
    p.requires_grad_(False)
cfg = timm.data.resolve_data_config({}, model=backbone)
tf = transforms.Compose([
    transforms.Resize(248), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(cfg["mean"], cfg["std"]),
])
train_set = datasets.CIFAR100("./data", train=True, download=True, transform=tf)
test_set = datasets.CIFAR100("./data", train=False, download=True, transform=tf)

@torch.no_grad()
def extract(ds):
    feats, labels = [], []
    for xb, yb in DataLoader(ds, batch_size=256, num_workers=2):
        feats.append(backbone(xb.to(DEVICE)).cpu()); labels.append(yb)
    F = torch.cat(feats).numpy(); y = torch.cat(labels).numpy()
    return F, y

print("Extracting frozen features ...")
Ftr, ytr = extract(train_set); Fte, yte = extract(test_set)
d = Ftr.shape[1]
# normalize so GD is stable (global top-eigenvalue ~1)
scale = np.sqrt(np.linalg.eigvalsh(Ftr.T @ Ftr / len(Ftr))[-1])
Ftr, Fte = Ftr / scale, Fte / scale
task_classes = [list(range(t * CLS_PER_TASK, (t + 1) * CLS_PER_TASK)) for t in range(N_TASKS)]

# %% [markdown]
# ## 2. Continual-learning methods on the shared head
# A1 holds exactly (backbone frozen, head linear), so forgetting of task A equals
# ½ Δ_c^T Σ_A Δ_c summed over A's class rows — the basis of OGD/GPM and IGFA.

# %%
def task_data(F, y, classes):
    idx = np.isin(y, classes)
    Yp = np.zeros((idx.sum(), C), np.float32)
    Yp[np.arange(idx.sum()), y[idx]] = 1.0
    return F[idx], Yp, y[idx]

def basis_of(X, r):
    S = X.T @ X / len(X)
    w, V = np.linalg.eigh(S)
    return V[:, -r:]

def evaluate(W, classes):
    Xt, _, yt = task_data(Fte, yte, classes)       # renamed locals: don't shadow global yte
    mask = np.full(C, -1e9); mask[classes] = 0.0
    pred = np.argmax(Xt @ W.T + mask, 1)           # task-IL: restrict to task's classes
    return float(np.mean(pred == yt))

def run(method, epochs=300, lr=0.5, r=40, ewc_lam=1.0, mem=20, sstar=0.55):
    W = np.zeros((C, d), np.float32)
    occupied = np.zeros((d, 0), np.float32); past_bases = []
    fisher = np.zeros(d, np.float32); W_star = np.zeros((C, d), np.float32)
    bufX, bufY = [], []
    acc_matrix = []
    for ti, classes in enumerate(task_classes):
        Xb, Yb, _ = task_data(Ftr, ytr, classes)
        Bn = basis_of(Xb, r)
        # ---- choose protected subspace ----
        if method == "ogd" and occupied.shape[1]:
            P = np.eye(d, dtype=np.float32) - occupied @ occupied.T
        elif method == "igfa":
            # Signed gate: protect (orthogonalize against) only the PAST tasks that
            # CONFLICT, i.e. whose feature-subspace overlap with the new task is below
            # s*; let aligned tasks share. With a frozen backbone there is no drift,
            # so the per-task bases are the correct geometry. (If you adapt the backbone
            # with LoRA, replace `past_bases` protection by the significant eigenvectors
            # of a decayed running Gauss-Newton estimate C <- gamma*C + X^T X / n, per
            # Sec. 4 of the paper -- the self-correcting update under drift.)
            protect = [Bt for Bt in past_bases
                       if np.linalg.norm(Bt.T @ Bn, "fro") ** 2 / r < sstar]
            if protect:
                Q = np.linalg.qr(np.hstack(protect))[0]
                P = np.eye(d, dtype=np.float32) - Q @ Q.T
            else:
                P = None
        else:
            P = None
        # ---- train shared head on this task ----
        for _ in range(epochs):
            if method == "replay" and bufX:
                Xt = np.vstack([Xb] + bufX); Yt = np.vstack([Yb] + bufY)
            else:
                Xt, Yt = Xb, Yb
            G = (Xt @ W.T - Yt).T @ Xt / len(Xt)
            if method == "ewc" and ti:
                # normalized Fisher keeps the penalty scale stable as it accumulates
                G = G + ewc_lam * (fisher / (fisher.max() + 1e-8))[None, :] * (W - W_star)
            if P is not None:
                G = G @ P.T
            W = W - lr * G
        # ---- bookkeeping ----
        past_bases.append(Bn)
        occupied = np.linalg.qr(np.hstack([occupied, Bn]))[0] if occupied.shape[1] else np.linalg.qr(Bn)[0]
        if method == "ewc":
            fisher += np.diag(Xb.T @ Xb / len(Xb)).astype(np.float32); W_star = W.copy()
        if method == "replay":
            sel = np.random.choice(len(Xb), min(mem * CLS_PER_TASK, len(Xb)), replace=False)
            bufX.append(Xb[sel]); bufY.append(Yb[sel])
        acc_matrix.append([evaluate(W, task_classes[j]) for j in range(N_TASKS)])
    A = np.array(acc_matrix)
    final = A[-1]
    diag = np.array([A[t, t] for t in range(N_TASKS)])
    return float(final.mean()), float(np.mean(diag[:-1] - final[:-1]))

# %% [markdown]
# ## 3. Run all methods

# %%
results = {}
for m in ["naive", "ewc", "ogd", "replay", "igfa"]:
    acc, forget = run(m)
    results[m] = (acc, forget)
    print(f"{m:8s}  avg-acc {acc:.3f}   forgetting {forget:.3f}")

print("\n=== Split-CIFAR-100 (frozen ViT-B/16, task-incremental) ===")
print(f"{'method':10s}{'avg acc':>10s}{'forgetting':>12s}")
for m, (a, f) in results.items():
    print(f"{m:10s}{a:>10.3f}{f:>12.3f}")
# Expected pattern: IGFA matches/beats OGD and EWC with no buffer and no Fisher;
# replay is a strong buffer-based reference. The s* gate shares aligned tasks and
# orthogonalizes conflicting ones; tune s* in the broad 0.4-0.8 range (see the
# ablation in the paper). For an ADAPTING backbone (LoRA), swap the per-task-basis
# protection for the running Gauss-Newton estimate noted in the igfa branch above.
