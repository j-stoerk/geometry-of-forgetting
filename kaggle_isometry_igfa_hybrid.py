"""
Interference control x plasticity preservation: IGFA + dynamical isometry.

One Kaggle-runnable file (2x T4). Tests the hybrid proposed on top of the
interference-geometry paper: keep IGFA as the interference controller (it decides
WHERE the update may go, protecting old-task geometry) and add a Rosseau-style
dynamical-isometry regularizer that keeps the ALLOWED directions learnable by driving
each weight matrix's singular values toward one.

The gate uses the correct A-GEM sign: it fires ONLY on genuine conflict (<g, g_anchor> < 0,
the update would raise past loss) and keeps beneficial transfer (<g, g_anchor> > 0), removing
an alpha-fraction of the conflicting component (alpha=1 hard, alpha=0 naive).

v3 -- confirmation + two mechanism refinements (this file):
* 3 seeds by default, with per-seed std reported on avg_acc / learn / forget.
* Sharper gate: per-task GEM projection ('gem' methods) protects each conflicting past
  task individually (project onto the intersection of halfspaces, solved in the reduced
  k-dim dual by projected Gauss-Seidel) vs the mixed-aggregate A-GEM gate ('agem').
* Synergy probe: logs gate COST = ||g - gated|| / ||g|| per task, to test whether
  isometry lowers the price of gating (cheaper gate in a higher-rank representation).
* Default run is the mechanism comparison at the winning alpha=1, hybrid iso_lambda=0.1,
  on the PER-TASK PIXEL-PERMUTED stream (the harder A1-breaks / visible-drift regime; on
  by default -- disable with --no-permute-inputs):
    naive, iso, igfa(agem), igfa-gem, igfa+iso(agem), igfa+iso-gem   x seeds [0,1,2]
  Re-trace the soft-gate curve with e.g. --alpha-sweep 1 0.75 0.5 0.25 0.
* Results are written to output/isometry_igfa_hybrid_alpha_results.json
"""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_VERSION = "isometry-igfa-hybrid-v3-gem-3seed"
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- logging
def to_python(x: Any):
    if isinstance(x, dict):
        return {k: to_python(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_python(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return x


class RunLogger:
    def __init__(self):
        self.lines: List[str] = []
        self.events: List[Dict[str, Any]] = []
        self.t0 = time.time()

    def log(self, msg: str, **fields) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        self.lines.append(line)
        ev = {"t_sec": round(time.time() - self.t0, 3), "message": msg}
        ev.update({k: to_python(v) for k, v in fields.items()})
        self.events.append(ev)


LOGGER = RunLogger()


def log(msg: str, **fields) -> None:
    LOGGER.log(msg, **fields)


# ----------------------------------------------------------------------------- model
class Net(nn.Module):
    """
    Deeper plain ReLU CNN, no normalization, orthogonal init.
    No norm is deliberate: weight-conditioning drift and dead ReLUs
    only appear without a normalizer papering over them.
    """

    def __init__(self, ncls: int = 100, width: int = 64):
        super().__init__()
        w = width
        self.f = nn.Sequential(
            nn.Conv2d(3, w, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(w, 2 * w, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(2 * w, 2 * w, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(2 * w, 4 * w, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.head = nn.Linear(4 * w * 4 * 4, ncls)
        self._orthogonal_init()

    def _orthogonal_init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight.reshape(m.weight.shape[0], -1))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def weight_matrices(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            out.append((name, mod.weight))
    return out


# ------------------------------------------------------------------ isometry (Rosseau)
def _as_matrix(w: torch.Tensor) -> torch.Tensor:
    return w if w.dim() == 2 else w.reshape(w.shape[0], -1)


@torch.no_grad()
def iso_grad(w: torch.Tensor) -> torch.Tensor:
    """
    Closed-form gradient of:
      ||W^T W - I||_F^2 for tall/square
      ||W W^T - I||_F^2 for wide
    """
    M = _as_matrix(w)
    out_dim, in_dim = M.shape
    if out_dim >= in_dim:
        G = M.transpose(0, 1) @ M
        G.diagonal().sub_(1.0)
        grad = 4.0 * (M @ G)
    else:
        G = M @ M.transpose(0, 1)
        G.diagonal().sub_(1.0)
        grad = 4.0 * (G @ M)
    return grad.reshape(w.shape)


@torch.no_grad()
def iso_penalty(model: nn.Module) -> float:
    tot = 0.0
    for _, w in weight_matrices(model):
        M = _as_matrix(w)
        out_dim, in_dim = M.shape
        if out_dim >= in_dim:
            G = M.transpose(0, 1) @ M
            G.diagonal().sub_(1.0)
        else:
            G = M @ M.transpose(0, 1)
            G.diagonal().sub_(1.0)
        tot += float((G * G).sum().item())
    return tot


# ------------------------------------------------------------------------- IGFA gate
def flat_grad(params: List[nn.Parameter]) -> torch.Tensor:
    return torch.cat([
        p.grad.reshape(-1) if p.grad is not None else torch.zeros(p.numel(), device=p.device)
        for p in params
    ])


def write_grad(params: List[nn.Parameter], vec: torch.Tensor) -> None:
    i = 0
    for p in params:
        n = p.numel()
        if p.grad is not None:
            p.grad.copy_(vec[i:i + n].reshape(p.shape))
        i += n


def sign_gate(g: torch.Tensor, g_anchor: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, float]:
    """
    IGFA gate (A-GEM/GPM convention).  The parameter update is -g, so the first-order
    change in past loss is  dL_past = <g_anchor, -g> = -ip.  Hence:
        ip = <g, g_anchor> > 0  ->  past loss DECREASES  ->  beneficial transfer  -> KEEP g
        ip < 0                  ->  past loss INCREASES   ->  forgetting           -> GATE
    When ip < 0 remove an alpha-fraction of the conflicting component:
        gated = g - alpha * (ip/||g_anchor||^2) * g_anchor
    which drives <gated, g_anchor> toward 0 (alpha=1 -> exactly non-increasing on the
    anchors; alpha=0 -> naive).  Firing only on genuine conflict preserves the shared
    learning signal that dominates a shared-feature net.
    """
    denom = float(g_anchor.dot(g_anchor).item())
    if denom <= 1e-20:
        return g, 0.0
    ip = float(g.dot(g_anchor).item())
    if ip >= 0.0:                              # aligned: transfer, keep the update
        return g, 0.0
    gated = g - alpha * (ip / denom) * g_anchor
    rate = -ip / (float(g.norm().item()) * float(g_anchor.norm().item()) + 1e-20)
    return gated, rate


# ------------------------------------------------------------------------------ AdamO
class AdamO:
    """
    Adam moments are built from the task gradient only.
    Isometry is applied as a decoupled post-step.
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.t = 0

    @torch.no_grad()
    def step(
        self,
        model: nn.Module,
        iso_lambda: float = 0.0,
        iso_lr: Optional[float] = None,
        iso_clip: float = 0.1,
    ) -> None:
        self.t += 1
        b1c = 1.0 - self.b1 ** self.t
        b2c = 1.0 - self.b2 ** self.t

        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            p.addcdiv_(m / b1c, (v / b2c).sqrt_().add_(self.eps), value=-self.lr)

        if iso_lambda > 0.0:
            ilr = self.lr if iso_lr is None else iso_lr
            for _, w in weight_matrices(model):
                s = iso_grad(w).mul_(ilr * iso_lambda)
                if iso_clip > 0.0:
                    max_norm = iso_clip * float(w.norm().item())
                    sn = float(s.norm().item())
                    if sn > max_norm > 0.0:
                        s.mul_(max_norm / sn)
                w.sub_(s)


# -------------------------------------------------------------------------------- EWC
class EWC:
    def __init__(self):
        self.means: List[List[torch.Tensor]] = []
        self.fisher: List[List[torch.Tensor]] = []

    def penalty(self, model: nn.Module) -> torch.Tensor:
        p = torch.tensor(0.0, device=DEV)
        for mean, fish in zip(self.means, self.fisher):
            for (_, w), m0, f in zip(model.named_parameters(), mean, fish):
                p = p + (f * (w - m0) ** 2).sum()
        return p

    @torch.no_grad()
    def consolidate(self, model: nn.Module) -> None:
        self.means.append([w.detach().clone() for _, w in model.named_parameters()])
        self.fisher.append([torch.zeros_like(w) for _, w in model.named_parameters()])

    def estimate_fisher(self, model: nn.Module, task_items, batch_size: int, budget: int = 1000) -> None:
        fish = [torch.zeros_like(w) for _, w in model.named_parameters()]
        n = 0
        sample = task_items[:budget] if len(task_items) > budget else task_items
        for x, y in make_loader(sample, batch_size, True):
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y).backward()
            for f, (_, w) in zip(fish, model.named_parameters()):
                if w.grad is not None:
                    f += (w.grad.detach() ** 2) * len(y)
            n += len(y)
        self.fisher[-1] = [f / max(1, n) for f in fish]


# ---------------------------------------------------------------- IGFA anchor memory
class AnchorMemory:
    """
    Bounded per-task exemplar pools of past tasks.
    IGFA samples one mixed past batch per step.
    """

    def __init__(self, mem_per_task: int = 200):
        self.mem_per_task = mem_per_task
        self.pools: List[List[Tuple[torch.Tensor, int]]] = []

    def add_task(self, items: List[Tuple[torch.Tensor, int]]) -> None:
        idx = torch.randperm(len(items))[: self.mem_per_task]
        self.pools.append([items[i] for i in idx.tolist()])

    def has_past(self) -> bool:
        return len(self.pools) > 0

    def num_past(self) -> int:
        return len(self.pools)

    def sample(self, n: int) -> List[Tuple[torch.Tensor, int]]:
        flat = [it for pool in self.pools for it in pool]
        if len(flat) == 0:
            return []
        idx = torch.randperm(len(flat))[: min(n, len(flat))]
        return [flat[i] for i in idx.tolist()]

    def sample_task(self, ti: int, n: int) -> List[Tuple[torch.Tensor, int]]:
        pool = self.pools[ti]
        idx = torch.randperm(len(pool))[: min(n, len(pool))]
        return [pool[i] for i in idx.tolist()]


def anchor_gradient(model: nn.Module, mem: AnchorMemory, n: int, params: List[nn.Parameter]) -> Optional[torch.Tensor]:
    if not mem.has_past():
        return None
    batch = mem.sample(n)
    if len(batch) == 0:
        return None
    X = torch.stack([x for x, _ in batch]).to(DEV)
    Y = torch.tensor([y for _, y in batch], device=DEV)
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(X), Y).backward()
    g = flat_grad(params).detach().clone()
    model.zero_grad(set_to_none=True)
    return g


def anchor_gradients_per_task(model: nn.Module, mem: AnchorMemory, k: int, n: int,
                              params: List[nn.Parameter]) -> Optional[List[torch.Tensor]]:
    """Sharper than the mixed aggregate: sample up to k past tasks and return a SEPARATE
    reference gradient per task, so the gate can protect each conflicting task individually
    (GEM) rather than an averaged 'past' direction (A-GEM).  k backwards/step."""
    if not mem.has_past():
        return None
    m = mem.num_past()
    task_ids = torch.randperm(m)[: min(k, m)].tolist()
    grads = []
    for ti in task_ids:
        batch = mem.sample_task(ti, n)
        if not batch:
            continue
        X = torch.stack([x for x, _ in batch]).to(DEV)
        Y = torch.tensor([y for _, y in batch], device=DEV)
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(X), Y).backward()
        grads.append(flat_grad(params).detach().clone())
    model.zero_grad(set_to_none=True)
    return grads or None


def gem_gate(g: torch.Tensor, anchors: List[torch.Tensor], alpha: float = 1.0) -> Tuple[torch.Tensor, float]:
    """GEM multi-constraint gate.  Find g_tilde = g + A v (v>=0) closest to g such that
    <g_i, g_tilde> >= 0 for every conflicting past task i, so the update -g_tilde is
    non-increasing on ALL of them simultaneously -- the exact projection onto the
    intersection of the halfspaces, not their average.  Solved in the reduced k-dim dual
    (k = #conflicting tasks) by projected Gauss-Seidel on the small Gram system; no
    million-dim CPU transfer.  alpha scales the correction (1=full projection, 0=naive)."""
    Gmat = torch.stack(anchors)                    # (m, d)
    c = Gmat @ g                                   # (m,) = <g_i, g>
    viol = c < 0.0                                 # tasks the raw step would forget
    if not bool(viol.any()):
        return g, 0.0
    A = Gmat[viol]                                 # (k, d) conflicting refs
    cc = c[viol]                                   # (k,)
    Q = A @ A.transpose(0, 1)                       # (k, k) Gram
    diag = Q.diagonal().clamp_min(1e-12)
    v = torch.zeros(A.shape[0], device=g.device)
    for _ in range(20):                            # projected Gauss-Seidel for the LCP
        for i in range(A.shape[0]):
            v[i] = torch.clamp(v[i] - (cc[i] + Q[i].dot(v)) / diag[i], min=0.0)
    gated = g + alpha * (A.transpose(0, 1) @ v)     # g + A v : pushes each <g_i,.> up to >=0
    rate = float(viol.float().mean().item())        # fraction of past tasks in conflict
    return gated, rate


# ------------------------------------------------------------------------ diagnostics
@torch.no_grad()
def isometry_diag(model: nn.Module) -> Dict[str, Any]:
    layers = {}
    drifts, kappas, msv = [], [], []
    for name, w in weight_matrices(model):
        M = _as_matrix(w).float()
        out_dim, in_dim = M.shape
        d = min(out_dim, in_dim)
        if out_dim >= in_dim:
            G = M.transpose(0, 1) @ M
        else:
            G = M @ M.transpose(0, 1)
        drift = float((G - torch.eye(G.shape[0], device=G.device)).norm().item()) / (d ** 0.5)
        sv = torch.linalg.svdvals(M)
        smax = float(sv[0].item())
        smin = float(sv[min(d - 1, sv.numel() - 1)].item())
        kappa = smax / (smin + 1e-12)
        mean_sq_sv = float((sv * sv).mean().item())
        layers[name] = {
            "drift": drift,
            "kappa": kappa,
            "smax": smax,
            "smin": smin,
            "mean_sq_sv": mean_sq_sv,
        }
        drifts.append(drift)
        kappas.append(kappa)
        msv.append(mean_sq_sv)
    return {
        "per_layer": layers,
        "drift_mean": float(np.mean(drifts)),
        "drift_max": float(np.max(drifts)),
        "kappa_mean": float(np.mean(kappas)),
        "kappa_max": float(np.max(kappas)),
        "mean_sq_sv_mean": float(np.mean(msv)),
    }


@torch.no_grad()
def plasticity_probes(model: nn.Module, probe_items, batch_size: int) -> Dict[str, Any]:
    was_training = model.training
    model.eval()
    acts: Dict[str, torch.Tensor] = {}
    handles = []
    relu_id = [0]

    def mk_hook(key):
        def hook(_m, _i, out):
            pos = (out > 0).float().flatten(1).mean(0)
            acts[key] = pos.detach()
        return hook

    for mod in model.f:
        if isinstance(mod, nn.ReLU):
            handles.append(mod.register_forward_hook(mk_hook(f"relu{relu_id[0]}")))
            relu_id[0] += 1

    batch = probe_items[: min(batch_size, len(probe_items))]
    X = torch.stack([x for x, _ in batch]).to(DEV)
    feats = model.features(X)

    for h in handles:
        h.remove()

    dead = {k: float((v == 0).float().mean().item()) for k, v in acts.items()}
    dead_overall = float(np.mean(list(dead.values()))) if dead else 0.0

    f = feats - feats.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(f)
    p = (sv * sv)
    p = p / (p.sum() + 1e-12)
    eff_rank = float(torch.exp(-(p * (p + 1e-12).log()).sum()).item())

    if was_training:
        model.train()
    return {
        "dead_relu_frac": dead_overall,
        "dead_relu_per_layer": dead,
        "feature_eff_rank": eff_rank,
        "feature_dim": int(feats.shape[1]),
    }


# ------------------------------------------------------------------------- data (port)
def find_cifar_root():
    import glob

    def has(p):
        return all(os.path.isfile(os.path.join(p, f)) for f in ["train", "test", "meta"])

    cand = (
        ["./data"]
        + glob.glob("/kaggle/input/*")
        + glob.glob("/kaggle/input/*/*")
        + glob.glob("/kaggle/input/*/*/*")
    )
    seen, ordered = set(), []
    for c in cand:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    for root in ordered:
        if not os.path.isdir(root):
            continue
        if has(root):
            return root, "flat"
        nested = os.path.join(root, "cifar-100-python")
        if os.path.isdir(nested) and has(nested):
            return nested, "nested"
    return None, None


def load_cifar100_pickles(base: str):
    import pickle

    def unpickle(fp):
        with open(fp, "rb") as fo:
            return pickle.load(fo, encoding="bytes")

    trd = unpickle(os.path.join(base, "train"))
    ted = unpickle(os.path.join(base, "test"))

    class DS:
        pass

    tr, te = DS(), DS()
    tr.data = np.asarray(trd[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    te.data = np.asarray(ted[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    tr.targets = [int(y) for y in trd[b"fine_labels"]]
    te.targets = [int(y) for y in ted[b"fine_labels"]]
    return tr, te


def get_split_cifar100(tasks: int, cpt: int):
    import socket
    import time as _t

    socket.setdefaulttimeout(30)
    root, mode = find_cifar_root()

    if root is not None:
        log(f"CIFAR-100 found locally at {root} (mode={mode}); skipping download")
        tr, te = load_cifar100_pickles(root)
    else:
        import torchvision

        log("CIFAR-100 not found locally; downloading (needs Kaggle Internet ON)")
        last = None
        for attempt in range(4):
            try:
                trv = torchvision.datasets.CIFAR100("./data", train=True, download=True, transform=None)
                tev = torchvision.datasets.CIFAR100("./data", train=False, download=True, transform=None)

                class DS:
                    pass

                tr, te = DS(), DS()
                tr.data, tr.targets = trv.data, trv.targets
                te.data, te.targets = tev.data, tev.targets
                break
            except Exception as e:
                last = e
                log(f"download attempt {attempt + 1}/4 failed: {type(e).__name__}: {e}; retrying")
                _t.sleep(5 * (attempt + 1))
        else:
            raise SystemExit(f"Could not load CIFAR-100 after 4 tries. Last error: {last}")

    mean = torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1)
    std = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1)

    def prep(ds):
        X = (
            torch.from_numpy(np.asarray(ds.data))
            .float()
            .permute(0, 3, 1, 2)
            .div_(255.0)
            .sub_(mean)
            .div_(std)
        )
        y = torch.as_tensor(ds.targets)
        out = [[] for _ in range(tasks)]
        for i in range(len(y)):
            out[int(y[i]) // cpt].append((X[i], int(y[i])))
        return out

    log("transforming + splitting CIFAR-100 (vectorized)")
    return prep(tr), prep(te)


def make_loader(items, batch_size: int, shuffle: bool):
    Xs = torch.stack([x for x, _ in items])
    ys = torch.tensor([y for _, y in items])
    idx = torch.randperm(len(items)) if shuffle else torch.arange(len(items))
    for i in range(0, len(items), batch_size):
        j = idx[i:i + batch_size]
        yield Xs[j].to(DEV), ys[j].to(DEV)


def apply_task_permutations(tr, te) -> None:
    for t in range(len(tr)):
        perm = torch.from_numpy(np.random.RandomState(1000 + t).permutation(32 * 32)).long()
        for pool in (tr[t], te[t]):
            for k in range(len(pool)):
                x, y = pool[k]
                pool[k] = (
                    x.reshape(3, 32 * 32)[:, perm].reshape(3, 32, 32).contiguous(),
                    y,
                )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    te_task,
    batch_size: int,
    task_id: Optional[int] = None,
    cpt: Optional[int] = None,
) -> float:
    was_training = model.training
    model.eval()
    c = n = 0
    lo = task_id * cpt if (task_id is not None and cpt is not None) else None

    for x, y in make_loader(te_task, batch_size, False):
        logits = model(x)
        if lo is not None:
            masked = torch.full_like(logits, float("-inf"))
            masked[:, lo:lo + cpt] = logits[:, lo:lo + cpt]
            logits = masked
        c += (logits.argmax(1) == y).sum().item()
        n += len(y)

    if was_training:
        model.train()
    return c / max(1, n)


def summarize_metrics(acc: np.ndarray, cpt: int) -> Dict[str, float]:
    T = acc.shape[0]
    chance = 1.0 / cpt
    learn = float(np.mean([acc[j, j] for j in range(T)]))
    final = float(np.mean([acc[T - 1, j] for j in range(T)]))
    forget = (
        float(np.mean([max(acc[i, j] for i in range(j, T)) - acc[T - 1, j] for j in range(T - 1)]))
        if T > 1 else 0.0
    )
    bwt = float(np.mean([acc[T - 1, j] - acc[j, j] for j in range(T - 1)])) if T > 1 else 0.0
    L = learn - chance
    omega = float(1.0 - forget / L) if L > 1e-6 else float("nan")
    return {
        "avg_acc": final,
        "learning_acc": learn,
        "forgetting": forget,
        "bwt": bwt,
        "omega": omega,
    }


# -------------------------------------------------------------------- method registry
METHODS = {
    "naive": dict(gate=False, iso=0.0, ewc=False, gate_mode=None),
    "ewc": dict(gate=False, iso=0.0, ewc=True, gate_mode=None),
    "iso": dict(gate=False, iso=1.0, ewc=False, gate_mode=None),
    "igfa": dict(gate=True, iso=0.0, ewc=False, gate_mode="agem"),        # mixed-aggregate gate
    "igfa-gem": dict(gate=True, iso=0.0, ewc=False, gate_mode="gem"),     # per-task (sharper) gate
    "igfa+iso": dict(gate=True, iso=1.0, ewc=False, gate_mode="agem"),
    "igfa+iso-gem": dict(gate=True, iso=1.0, ewc=False, gate_mode="gem"),
}


def run_one(
    method: str,
    tr,
    te,
    args,
    seed: int,
    run_alpha: Optional[float] = None,
    run_iso_lambda: Optional[float] = None,
) -> Dict[str, Any]:
    run_t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = METHODS[method]
    gate_alpha = float(args.alpha if run_alpha is None else run_alpha)

    default_iso_lambda = args.iso_lambda * cfg["iso"]
    if cfg["gate"] and cfg["iso"] > 0.0:                 # any hybrid (agem or gem) uses hybrid_iso_lambda
        default_iso_lambda = args.hybrid_iso_lambda
    iso_lambda = float(default_iso_lambda if run_iso_lambda is None else run_iso_lambda)

    model = Net(ncls=args.tasks * args.cpt, width=args.width).to(DEV)
    model.train()
    params = [p for p in model.parameters()]
    opt = AdamO(params, lr=args.lr)
    ewc = EWC() if cfg["ewc"] else None
    mem = AnchorMemory(args.mem_per_task) if cfg["gate"] else None
    gate_mode = cfg["gate_mode"]

    acc = np.zeros((args.tasks, args.tasks), dtype=np.float64)
    gate_rates: List[float] = []
    gate_costs: List[float] = []                    # ||g - gated|| / ||g||: how much the gate removed
    task_logs: List[Dict[str, Any]] = []

    for t in range(args.tasks):
        task_t0 = time.time()
        epoch_losses = []

        for ep in range(args.epochs):
            loss_sum, nb = 0.0, 0
            for x, y in make_loader(tr[t], args.batch_size, True):
                # protected direction(s), computed BEFORE the task backward overwrites .grad
                anchors = None
                if mem is not None and mem.has_past():
                    if gate_mode == "gem":
                        anchors = anchor_gradients_per_task(model, mem, args.gem_k, args.anchor_batch, params)
                    else:
                        ga = anchor_gradient(model, mem, args.anchor_batch, params)
                        anchors = [ga] if ga is not None else None

                model.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits, y)

                if ewc is not None and ewc.means:
                    loss = loss + args.ewc_lambda * ewc.penalty(model)

                loss.backward()

                if anchors is not None:
                    g = flat_grad(params)
                    if gate_mode == "gem":
                        gated, rate = gem_gate(g, anchors, alpha=gate_alpha)
                    else:
                        gated, rate = sign_gate(g, anchors[0], alpha=gate_alpha)
                    cost = float((g - gated).norm().item() / (g.norm().item() + 1e-20))
                    if rate > 0.0:
                        write_grad(params, gated)
                    gate_rates.append(rate)
                    gate_costs.append(cost)

                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                opt.step(model, iso_lambda=iso_lambda, iso_lr=args.iso_lr, iso_clip=args.iso_clip)

                loss_sum += float(loss.item())
                nb += 1

            epoch_losses.append(loss_sum / max(1, nb))

        if ewc is not None:
            ewc.consolidate(model)
            ewc.estimate_fisher(model, tr[t], args.batch_size)

        if mem is not None:
            mem.add_task(tr[t])

        task_accs = {}
        for j in range(t + 1):
            task_accs[f"task_{j}"] = float(evaluate(model, te[j], args.batch_size, task_id=j, cpt=args.cpt))
            acc[t, j] = task_accs[f"task_{j}"]

        iso_d = isometry_diag(model)
        plast = plasticity_probes(model, tr[t], args.batch_size)
        metrics_now = summarize_metrics(acc[: t + 1, : t + 1], args.cpt)

        task_log = {
            "task_index": t,
            "epoch_losses": epoch_losses,
            "task_eval": task_accs,
            "mean_acc_so_far": float(np.mean(acc[t, : t + 1])),
            "metrics_so_far": metrics_now,
            "isometry": {k: v for k, v in iso_d.items() if k != "per_layer"},
            "isometry_per_layer": iso_d["per_layer"],
            "plasticity": plast,
            "iso_penalty": iso_penalty(model),
            "gate_alpha": gate_alpha,
            "gate_mode": gate_mode,
            "gate_rate_mean": float(np.mean(gate_rates[-nb:])) if (mem is not None and gate_rates) else 0.0,
            "gate_cost_mean": float(np.mean(gate_costs[-nb:])) if (mem is not None and gate_costs) else 0.0,
            "task_wallclock_sec": round(time.time() - task_t0, 4),
        }
        task_logs.append(task_log)

        log(
            f"[{method}] task {t}: acc-so-far={task_log['mean_acc_so_far']:.3f} "
            f"learn={metrics_now['learning_acc']:.3f} forget={metrics_now['forgetting']:.3f} "
            f"drift={iso_d['drift_mean']:.3f} kappa={iso_d['kappa_mean']:.1f} "
            f"dead={plast['dead_relu_frac']:.2f} effrank={plast['feature_eff_rank']:.1f} "
            f"alpha={gate_alpha:.2f} iso={iso_lambda:g}",
            method=method, seed=seed, task=t, alpha=gate_alpha, iso_lambda=iso_lambda,
        )

    final_metrics = summarize_metrics(acc, args.cpt)
    return {
        "method": method,
        "seed": seed,
        "device": DEV,
        "run_wallclock_sec": round(time.time() - run_t0, 4),
        "final_metrics": final_metrics,
        "accuracy_matrix": acc.tolist(),
        "task_logs": task_logs,
        "gate_rate_overall": float(np.mean(gate_rates)) if gate_rates else 0.0,
        "gate_cost_overall": float(np.mean(gate_costs)) if gate_costs else 0.0,
        "gate_alpha": gate_alpha,
        "gate_mode": gate_mode,
        "iso_lambda": iso_lambda,
    }


def aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    fm = [r["final_metrics"] for r in runs]
    s = {
        k: float(np.mean([m[k] for m in fm]))
        for k in ["avg_acc", "learning_acc", "forgetting", "bwt", "omega"]
    }
    last = [r["task_logs"][-1] for r in runs]
    s["drift_mean_final"] = float(np.mean([l["isometry"]["drift_mean"] for l in last]))
    s["kappa_mean_final"] = float(np.mean([l["isometry"]["kappa_mean"] for l in last]))
    s["dead_relu_final"] = float(np.mean([l["plasticity"]["dead_relu_frac"] for l in last]))
    s["eff_rank_final"] = float(np.mean([l["plasticity"]["feature_eff_rank"] for l in last]))
    s["gate_rate"] = float(np.mean([r["gate_rate_overall"] for r in runs]))
    s["gate_cost"] = float(np.mean([r["gate_cost_overall"] for r in runs]))
    # per-seed std on the headline so 3-seed error bars are in the summary
    s["avg_acc_std"] = float(np.std([m["avg_acc"] for m in fm]))
    s["learning_acc_std"] = float(np.std([m["learning_acc"] for m in fm]))
    s["forgetting_std"] = float(np.std([m["forgetting"] for m in fm]))
    s["gate_alpha"] = float(np.mean([r["gate_alpha"] for r in runs]))
    s["iso_lambda"] = float(np.mean([r["iso_lambda"] for r in runs]))
    return s


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, default=20)
    p.add_argument("--cpt", type=int, default=5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument(
        "--iso-lambda",
        type=float,
        default=0.1,
        help="isometry strength for the standalone iso method (matches hybrid_iso_lambda)",
    )
    p.add_argument("--iso-lr", type=float, default=1e-3, help="decoupled isometry step eta_iso (=lr)")
    p.add_argument(
        "--iso-clip",
        type=float,
        default=0.1,
        help="cap the decoupled iso step at iso_clip*||W|| (stability; 0 disables)",
    )

    p.add_argument("--ewc-lambda", type=float, default=40.0)
    p.add_argument("--anchor-batch", type=int, default=128, help="IGFA past-batch size for g_anchor")
    p.add_argument("--mem-per-task", type=int, default=200, help="IGFA exemplars stored per past task")
    p.add_argument("--gem-k", type=int, default=4, help="GEM: #past tasks sampled per step for per-task gating")
    p.add_argument("--grad-clip", type=float, default=5.0)

    # DEFAULT stream is the per-task pixel-permuted one (the harder "A1 breaks / visible
    # drift" regime): each task must relearn low-level features, stressing plasticity.
    # Running the file in Kaggle with no args uses it.  Disable with --no-permute-inputs.
    p.add_argument(
        "--permute-inputs",
        dest="permute_inputs",
        action="store_true",
        default=True,
        help="apply a fixed per-task pixel permutation (DEFAULT ON)",
    )
    p.add_argument(
        "--no-permute-inputs",
        dest="permute_inputs",
        action="store_false",
        help="use the standard (unpermuted) Split-CIFAR-100 stream instead",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    p.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["naive", "iso", "igfa", "igfa-gem", "igfa+iso", "igfa+iso-gem"],
        help="default 3-seed mechanism comparison: baselines + aggregate(agem) vs per-task(gem) gates",
    )

    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="soft-gate projection fraction alpha: 1=hard gate, 0=naive",
    )
    p.add_argument(
        "--alpha-sweep",
        type=float,
        nargs="*",
        default=[1.0],
        help="alpha values for gated methods (default just 1.0, the winner). "
             "Pass e.g. --alpha-sweep 1 0.75 0.5 0.25 0 to re-trace the soft-gate curve",
    )
    p.add_argument(
        "--hybrid-iso-lambda",
        type=float,
        default=0.1,
        help="fixed iso-lambda for igfa+iso in the default soften-and-compare run",
    )
    p.add_argument(
        "--iso-sweep",
        type=float,
        nargs="*",
        default=[],
        help="optional iso-lambda sweep for explicit iso baselines or non-default experiments",
    )

    args, _ = p.parse_known_args()

    for m in args.methods:
        if m not in METHODS:
            raise ValueError(f"Unknown method: {m}")

    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0, 1]")

    if args.alpha_sweep is not None:
        for a in args.alpha_sweep:
            if not (0.0 <= a <= 1.0):
                raise ValueError("all --alpha-sweep values must be in [0, 1]")

    return args


def build_run_list(args):
    """
    Default:
      naive once
      igfa sweep over alpha
      igfa+iso sweep over alpha with fixed hybrid iso lambda = 0.1
    If --alpha-sweep is disabled (pass with no values), run selected methods once
    at --alpha unless an explicit --iso-sweep is requested for iso methods.
    """
    run_list = []

    if args.alpha_sweep:
        for m in args.methods:
            cfg = METHODS[m]
            if cfg["gate"]:
                for alpha in args.alpha_sweep:
                    lam = args.hybrid_iso_lambda if cfg["iso"] > 0.0 else 0.0
                    run_list.append((m, lam, alpha))
            elif cfg["iso"] > 0.0 and args.iso_sweep:
                for lam in args.iso_sweep:
                    run_list.append((m, lam, args.alpha))
            else:
                run_list.append((m, args.iso_lambda * cfg["iso"], args.alpha))
    else:
        for m in args.methods:
            cfg = METHODS[m]
            if cfg["iso"] > 0.0 and args.iso_sweep:
                for lam in args.iso_sweep:
                    run_list.append((m, lam, args.alpha))
            else:
                lam = args.hybrid_iso_lambda if (cfg["gate"] and cfg["iso"] > 0.0) else args.iso_lambda * cfg["iso"]
                run_list.append((m, lam, args.alpha))

    return run_list


def make_tag(method: str, iso_lambda: float, alpha: float) -> str:
    parts = [method]
    if METHODS[method]["gate"]:
        parts.append(f"alpha{alpha:g}")
    if METHODS[method]["iso"] > 0.0:
        parts.append(f"lam{iso_lambda:g}")
    return "@".join(parts)


if __name__ == "__main__":
    started = time.time()
    args = parse_args()

    log(f"{SCRIPT_VERSION}; device {DEV}", script_version=SCRIPT_VERSION, device=DEV)

    tr, te = get_split_cifar100(args.tasks, args.cpt)
    if args.permute_inputs:
        apply_task_permutations(tr, te)
        log("applied fixed per-task pixel permutations (plasticity-stress / visible-drift stream)")

    log(
        f"Split-CIFAR-100: {args.tasks} tasks x {args.cpt} classes "
        f"({'permuted-input' if args.permute_inputs else 'standard'} stream)",
        tasks=args.tasks, cpt=args.cpt, permute_inputs=args.permute_inputs,
    )

    run_list = build_run_list(args)
    method_runs: Dict[str, List[Dict[str, Any]]] = {}
    method_summaries: Dict[str, Any] = {}

    for base, lam, alpha in run_list:
        tag = make_tag(base, lam, alpha)
        runs = []
        for seed in args.seeds:
            log(
                f"START {tag} seed={seed} (iso_lambda={lam}, alpha={alpha})",
                method=tag, seed=seed, iso_lambda=lam, alpha=alpha,
            )
            rec = run_one(base, tr, te, args, seed, run_alpha=alpha, run_iso_lambda=lam)
            runs.append(rec)
            fm = rec["final_metrics"]
            log(
                f"END {tag} seed={seed}: avg_acc={fm['avg_acc']:.3f} "
                f"learn={fm['learning_acc']:.3f} forget={fm['forgetting']:.3f} "
                f"Omega={fm['omega']:+.3f}",
                method=tag, seed=seed, final_metrics=fm, iso_lambda=lam, alpha=alpha,
            )

        method_runs[tag] = runs
        method_summaries[tag] = aggregate(runs)
        s = method_summaries[tag]
        log(
            f"== {tag:22s} avg_acc {s['avg_acc']:.3f} learn {s['learning_acc']:.3f} "
            f"forget {s['forgetting']:.3f} Omega {s['omega']:+.3f} | "
            f"drift {s['drift_mean_final']:.3f} kappa {s['kappa_mean_final']:.1f} "
            f"dead {s['dead_relu_final']:.2f} effrank {s['eff_rank_final']:.1f} "
            f"gate {s['gate_rate']:.3f}",
            method=tag, summary=s,
        )

    payload = {
        "script_version": SCRIPT_VERSION,
        "device": DEV,
        "started_unix": started,
        "finished_unix": time.time(),
        "total_wallclock_sec": round(time.time() - started, 4),
        "args": vars(args),
        "default_behavior": {
            "methods": ["naive", "iso", "igfa", "igfa-gem", "igfa+iso", "igfa+iso-gem"],
            "seeds": [0, 1, 2],
            "alpha": 1.0,
            "gem_k": args.gem_k,
            "hybrid_iso_lambda": args.hybrid_iso_lambda,
            "permute_inputs": args.permute_inputs,
            "stream": "permuted-input" if args.permute_inputs else "standard",
            "output_json": "output/isometry_igfa_hybrid_alpha_results.json",
        },
        "method_summaries": method_summaries,
        "method_runs": method_runs,
        "stdout_lines": LOGGER.lines,
        "event_log": LOGGER.events,
    }

    os.makedirs("output", exist_ok=True)
    with open("output/isometry_igfa_hybrid_alpha_results.json", "w") as f:
        json.dump(to_python(payload), f, indent=1)

    print("\n" + "=" * 118)
    print("IGFA (interference control) x dynamical isometry (plasticity preservation) -- 3-seed confirm + gem/cost")
    print("agem = mixed-aggregate gate; gem = per-task gate; cost = ||g-gated||/||g|| (price of gating)")
    print("=" * 118)

    hdr = (
        f"{'method':<24}"
        f"{'avg_acc':>10}{'learn':>8}{'forget':>9}{'Omega':>8}"
        f"{'drift':>8}{'kappa':>8}{'dead':>7}{'effrank':>9}{'gcost':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for tag, s in method_summaries.items():
        avg = f"{s['avg_acc']:.3f}+-{s.get('avg_acc_std', 0.0):.2f}"
        print(
            f"{tag:<24}{avg:>10}"
            f"{s['learning_acc']:>8.3f}{s['forgetting']:>9.3f}"
            f"{s['omega']:>8.3f}{s['drift_mean_final']:>8.3f}{s['kappa_mean_final']:>8.1f}"
            f"{s['dead_relu_final']:>7.2f}{s['eff_rank_final']:>9.1f}{s.get('gate_cost', 0.0):>8.3f}"
        )

    log("ALL DONE -> output/isometry_igfa_hybrid_alpha_results.json")