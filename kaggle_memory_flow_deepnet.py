# =====================================================================
# kaggle_memory_flow_deep_exact_logged.py
#
# Deep, K-FAC-style exactization of the representation-drift note with
# structured logging and complete JSON output.
#
# Output JSON includes:
#   - config / args / device / timing
#   - per-method aggregated summaries
#   - per-seed run records
#   - full accuracy matrices
#   - per-task evaluation logs
#   - transport diagnostics
#   - refresh / anchor-backward accounting
#   - mirrored textual log lines and structured events
# =====================================================================

import argparse
import json
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_VERSION = "memflow-deep-exact-logged-v1"
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
if DEV.startswith("cuda") and torch.cuda.get_device_capability(0) < (7, 0):
    raise SystemExit("Unsupported GPU (sm<70): set Accelerator to 'GPU T4 x2'.")

T0 = time.time()


def to_python(x: Any):
    if isinstance(x, torch.Tensor):
        if x.ndim == 0:
            return x.item()
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: to_python(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_python(v) for v in x]
    return x


class RunLogger:
    def __init__(self):
        self.lines: List[str] = []
        self.events: List[Dict[str, Any]] = []

    def log(self, msg: str, **fields) -> None:
        dt = time.time() - T0
        line = f"[{dt:7.1f}s] {msg}"
        print(line, flush=True)
        self.lines.append(line)
        event = {"t_sec": round(dt, 4), "message": msg}
        if fields:
            event.update(to_python(fields))
        self.events.append(event)


LOGGER = RunLogger()


def log(msg: str, **fields) -> None:
    LOGGER.log(msg, **fields)


class ConvGN(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1, bias=True)
        self.norm = nn.GroupNorm(num_groups=min(8, cout), num_channels=cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Net(nn.Module):
    def __init__(self, ncls: int = 100):
        super().__init__()
        self.f = nn.Sequential(
            ConvGN(3, 32), ConvGN(32, 32), nn.MaxPool2d(2),
            ConvGN(32, 64), ConvGN(64, 64), nn.MaxPool2d(2),
            ConvGN(64, 128), ConvGN(128, 128), nn.MaxPool2d(2),
        )
        self.head = nn.Linear(128 * 4 * 4, ncls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.f(x).flatten(1))


@dataclass
class ReservoirEntry:
    uid: int
    x: torch.Tensor
    target_logits: torch.Tensor
    step_added: int


class AnchorReservoir:
    def __init__(self, cap: int, protect_delay: int):
        self.cap = cap
        self.protect_delay = protect_delay
        self.items: List[ReservoirEntry] = []
        self.n_seen = 0
        self.next_uid = 0

    def add_batch(self, x: torch.Tensor, logits: torch.Tensor, step: int) -> None:
        x_cpu = x.detach().cpu()
        z_cpu = logits.detach().cpu()
        for xi, zi in zip(x_cpu, z_cpu):
            self.n_seen += 1
            entry = ReservoirEntry(uid=self.next_uid, x=xi.clone(), target_logits=zi.clone(), step_added=step)
            self.next_uid += 1
            if len(self.items) < self.cap:
                self.items.append(entry)
            else:
                j = random.randrange(self.n_seen)
                if j < self.cap:
                    self.items[j] = entry

    def eligible(self, step: int) -> List[ReservoirEntry]:
        return [e for e in self.items if step - e.step_added >= self.protect_delay]

    def sample(self, step: int, n: int) -> List[ReservoirEntry]:
        pool = self.eligible(step)
        if not pool:
            return []
        k = min(n, len(pool))
        idx = random.sample(range(len(pool)), k)
        return [pool[i] for i in idx]

    def stats(self, step: int) -> Dict[str, int]:
        return {
            "total": len(self.items),
            "eligible": len(self.eligible(step)),
            "seen": self.n_seen,
        }

    def __len__(self) -> int:
        return len(self.items)


@dataclass
class FactorState:
    A: Dict[str, torch.Tensor]
    G: Dict[str, torch.Tensor]
    A_eig: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    G_eig: Dict[str, Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class ProbeState:
    entries: List[ReservoirEntry]
    act_sketch: Dict[str, torch.Tensor]
    grad_sketch: Dict[str, torch.Tensor]


class KFACAnchorMetric:
    def __init__(
        self,
        model: nn.Module,
        eps: float,
        rank_a: int,
        rank_g: int,
        ridge: float,
        transport_blend: float,
        sketch_rows: int,
    ):
        self.model = model
        self.eps = eps
        self.rank_a = rank_a
        self.rank_g = rank_g
        self.ridge = ridge
        self.transport_blend = transport_blend
        self.sketch_rows = sketch_rows

        self.layers: List[Tuple[str, nn.Module]] = []
        self.factor: Optional[FactorState] = None
        self.initialized = False
        self.prev_probe: Optional[ProbeState] = None
        self.refresh_count = 0
        self.anchor_backward_calls = 0
        self.transport_diag: List[Dict[str, float]] = []
        self.refresh_log: List[Dict[str, Any]] = []

        self._fwd_cache: Dict[str, torch.Tensor] = {}
        self._bwd_cache: Dict[str, torch.Tensor] = {}
        self.hooks = []

        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                self.layers.append((name, m))
                self.hooks.append(m.register_forward_hook(self._save_input(name, m)))
                self.hooks.append(m.register_full_backward_hook(self._save_grad_output(name, m)))

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()

    def _save_input(self, name: str, mod: nn.Module):
        def hook(module, inp, out):
            self._fwd_cache[name] = self._expand_input(inp[0].detach(), mod)
        return hook

    def _save_grad_output(self, name: str, mod: nn.Module):
        def hook(module, grad_input, grad_output):
            if grad_output and grad_output[0] is not None:
                self._bwd_cache[name] = self._expand_grad(grad_output[0].detach(), mod)
        return hook

    @staticmethod
    def _expand_input(x: torch.Tensor, mod: nn.Module) -> torch.Tensor:
        if isinstance(mod, nn.Conv2d):
            patches = F.unfold(x, mod.kernel_size, padding=mod.padding, stride=mod.stride)
            return patches.transpose(1, 2).reshape(-1, patches.shape[1])
        return x.reshape(-1, x.shape[-1])

    @staticmethod
    def _expand_grad(g: torch.Tensor, mod: nn.Module) -> torch.Tensor:
        if isinstance(mod, nn.Conv2d):
            return g.permute(0, 2, 3, 1).reshape(-1, g.shape[1])
        return g.reshape(-1, g.shape[-1])

    @staticmethod
    def _subsample_rows(x: torch.Tensor, max_rows: int) -> torch.Tensor:
        if x.shape[0] <= max_rows:
            return x
        idx = torch.randperm(x.shape[0], device=x.device)[:max_rows]
        return x[idx]

    @staticmethod
    def _cov(rows: torch.Tensor) -> torch.Tensor:
        rows = rows.float()
        return (rows.T @ rows) / max(1, rows.shape[0])

    def _pack_entries(self, entries: List[ReservoirEntry]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.stack([e.x for e in entries]).to(DEV)
        y = torch.stack([e.target_logits for e in entries]).to(DEV)
        return x, y

    def _compute_anchor_batch(self, entries: List[ReservoirEntry]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], float]:
        if not entries:
            return {}, {}, {}, {}, 0.0
        x, y_target = self._pack_entries(entries)
        self.model.zero_grad(set_to_none=True)
        self._fwd_cache = {}
        self._bwd_cache = {}
        was_training = self.model.training
        self.model.train()
        logits = self.model(x)
        loss = 0.5 * F.mse_loss(logits, y_target, reduction="mean")
        loss.backward()
        self.anchor_backward_calls += 1

        A, G, a_sketch, g_sketch = {}, {}, {}, {}
        for name, _ in self.layers:
            if name not in self._fwd_cache or name not in self._bwd_cache:
                continue
            a = self._subsample_rows(self._fwd_cache[name].float(), self.sketch_rows)
            g = self._subsample_rows(self._bwd_cache[name].float(), self.sketch_rows)
            A[name] = self._cov(a)
            G[name] = self._cov(g)
            a_sketch[name] = a.detach().cpu()
            g_sketch[name] = g.detach().cpu()

        self.model.zero_grad(set_to_none=True)
        if not was_training:
            self.model.eval()
        return A, G, a_sketch, g_sketch, float(loss.item())

    @staticmethod
    def _top_eig(cov: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
        w, V = torch.linalg.eigh(cov.float())
        r = min(rank, cov.shape[0])
        return V[:, -r:], w[-r:].clamp(min=0)

    def _make_factor_state(self, A: Dict[str, torch.Tensor], G: Dict[str, torch.Tensor]) -> FactorState:
        return FactorState(
            A=A,
            G=G,
            A_eig={name: self._top_eig(cov, self.rank_a) for name, cov in A.items()},
            G_eig={name: self._top_eig(cov, self.rank_g) for name, cov in G.items()},
        )

    def _factor_layer_summary(self, A: Dict[str, torch.Tensor], G: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        out = {}
        for name in sorted(set(A.keys()) & set(G.keys())):
            wa = torch.linalg.eigvalsh(A[name].float()).clamp(min=0)
            wg = torch.linalg.eigvalsh(G[name].float()).clamp(min=0)
            ra = min(self.rank_a, A[name].shape[0])
            rg = min(self.rank_g, G[name].shape[0])
            tr_a = float(wa.sum().item())
            tr_g = float(wg.sum().item())
            top_a = float(wa[-ra:].sum().item()) if ra > 0 else 0.0
            top_g = float(wg[-rg:].sum().item()) if rg > 0 else 0.0
            out[name] = {
                "A_dim": float(A[name].shape[0]),
                "G_dim": float(G[name].shape[0]),
                "A_trace": tr_a,
                "G_trace": tr_g,
                "A_rank_used": float(ra),
                "G_rank_used": float(rg),
                "A_top_mass_ratio": float(top_a / tr_a) if tr_a > 1e-12 else 0.0,
                "G_top_mass_ratio": float(top_g / tr_g) if tr_g > 1e-12 else 0.0,
                "A_top_eig": float(wa[-1].item()) if wa.numel() else 0.0,
                "G_top_eig": float(wg[-1].item()) if wg.numel() else 0.0,
            }
        return out

    def _estimate_transport(self, old_x: Optional[torch.Tensor], new_x: Optional[torch.Tensor], dim: int) -> torch.Tensor:
        if old_x is None or new_x is None:
            return torch.eye(dim, device=DEV)
        n = min(old_x.shape[0], new_x.shape[0])
        if n < 2 or old_x.shape[1] != new_x.shape[1]:
            return torch.eye(dim, device=DEV)
        Xo = old_x[:n].to(DEV, dtype=torch.float32)
        Xn = new_x[:n].to(DEV, dtype=torch.float32)
        Coo = (Xo.T @ Xo) / n
        Cno = (Xn.T @ Xo) / n
        I = torch.eye(Coo.shape[0], device=DEV, dtype=torch.float32)
        try:
            return Cno @ torch.linalg.inv(Coo + self.ridge * I)
        except RuntimeError:
            return Cno @ torch.linalg.pinv(Coo + self.ridge * I)

    @staticmethod
    def _transport_cov(T: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        return T @ S @ T.T

    @staticmethod
    def _rel_err(A: torch.Tensor, B: torch.Tensor) -> float:
        denom = B.norm().item()
        if denom < 1e-12:
            return 0.0
        return float((A - B).norm().item() / denom)

    @staticmethod
    def _subspace_overlap(U: torch.Tensor, V: torch.Tensor) -> float:
        q = min(U.shape[1], V.shape[1])
        if q == 0:
            return 0.0
        M = U[:, :q].T @ V[:, :q]
        return float((torch.linalg.norm(M, ord="fro") ** 2 / q).item())

    def maybe_refresh(self, reservoir: AnchorReservoir, step: int, mode: str, anchor_batch: int) -> None:
        fresh_entries = reservoir.sample(step, anchor_batch)
        if not fresh_entries:
            return

        if not self.initialized:
            A, G, a_sk, g_sk, anchor_loss = self._compute_anchor_batch(fresh_entries)
            if not A or not G:
                return
            self.factor = self._make_factor_state(A, G)
            self.prev_probe = ProbeState(entries=fresh_entries, act_sketch=a_sk, grad_sketch=g_sk)
            self.initialized = True
            self.refresh_count += 1
            self.refresh_log.append({
                "step": step,
                "mode": "init",
                "anchor_loss": anchor_loss,
                "n_entries": len(fresh_entries),
                "layer_summary": self._factor_layer_summary(A, G),
            })
            return

        if mode == "stale":
            return

        if mode == "functional":
            A, G, a_sk, g_sk, anchor_loss = self._compute_anchor_batch(fresh_entries)
            self.factor = self._make_factor_state(A, G)
            self.prev_probe = ProbeState(entries=fresh_entries, act_sketch=a_sk, grad_sketch=g_sk)
            self.refresh_count += 1
            self.refresh_log.append({
                "step": step,
                "mode": "functional",
                "anchor_loss": anchor_loss,
                "n_entries": len(fresh_entries),
                "layer_summary": self._factor_layer_summary(A, G),
            })
            return

        if mode != "transport":
            raise ValueError(f"unknown mode: {mode}")

        probe_now_A, probe_now_G, probe_now_a, probe_now_g, probe_loss, = self._compute_anchor_batch(self.prev_probe.entries)
        fresh_A, fresh_G, fresh_a, fresh_g, fresh_loss = self._compute_anchor_batch(fresh_entries)
        new_A, new_G = {}, {}
        layer_diag = {}
        diag = {"step": step, "A_rel_err": 0.0, "G_rel_err": 0.0, "A_overlap": 0.0, "G_overlap": 0.0, "layers": 0.0, "probe_anchor_loss": probe_loss, "fresh_anchor_loss": fresh_loss}

        for name, _ in self.layers:
            if name not in self.factor.A or name not in probe_now_A or name not in fresh_A:
                continue
            Ta = self._estimate_transport(self.prev_probe.act_sketch.get(name), probe_now_a.get(name), self.factor.A[name].shape[0])
            Tg = self._estimate_transport(self.prev_probe.grad_sketch.get(name), probe_now_g.get(name), self.factor.G[name].shape[0])
            A_trans = self._transport_cov(Ta, self.factor.A[name])
            G_trans = self._transport_cov(Tg, self.factor.G[name])
            A_new = (1.0 - self.transport_blend) * A_trans + self.transport_blend * fresh_A[name]
            G_new = (1.0 - self.transport_blend) * G_trans + self.transport_blend * fresh_G[name]
            new_A[name] = A_new
            new_G[name] = G_new

            U_A_trans, _ = self._top_eig(A_trans, self.rank_a)
            U_A_fresh, _ = self._top_eig(fresh_A[name], self.rank_a)
            U_G_trans, _ = self._top_eig(G_trans, self.rank_g)
            U_G_fresh, _ = self._top_eig(fresh_G[name], self.rank_g)
            A_rel = self._rel_err(A_trans, fresh_A[name])
            G_rel = self._rel_err(G_trans, fresh_G[name])
            A_ov = self._subspace_overlap(U_A_trans, U_A_fresh)
            G_ov = self._subspace_overlap(U_G_trans, U_G_fresh)
            diag["A_rel_err"] += A_rel
            diag["G_rel_err"] += G_rel
            diag["A_overlap"] += A_ov
            diag["G_overlap"] += G_ov
            diag["layers"] += 1.0
            layer_diag[name] = {
                "A_rel_err": float(A_rel),
                "G_rel_err": float(G_rel),
                "A_overlap": float(A_ov),
                "G_overlap": float(G_ov),
                "A_trace_transported": float(torch.trace(A_trans).item()),
                "A_trace_fresh": float(torch.trace(fresh_A[name]).item()),
                "G_trace_transported": float(torch.trace(G_trans).item()),
                "G_trace_fresh": float(torch.trace(fresh_G[name]).item()),
            }

        if diag["layers"] > 0:
            for k in ["A_rel_err", "G_rel_err", "A_overlap", "G_overlap"]:
                diag[k] /= diag["layers"]
        diag["layer_diag"] = layer_diag
        self.transport_diag.append(diag)
        self.factor = self._make_factor_state(new_A, new_G)
        self.prev_probe = ProbeState(entries=fresh_entries, act_sketch=fresh_a, grad_sketch=fresh_g)
        self.refresh_count += 1
        self.refresh_log.append({
            "step": step,
            "mode": "transport",
            "n_entries": len(fresh_entries),
            "probe_anchor_loss": probe_loss,
            "fresh_anchor_loss": fresh_loss,
            "diag": diag,
            "layer_summary": self._factor_layer_summary(new_A, new_G),
        })

    def _apply_inv_right(self, X: torch.Tensor, eig: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        U, s = eig
        coef = s / (self.eps + s)
        return (X - ((X @ U) * coef) @ U.T) / self.eps

    def _apply_inv_left(self, X: torch.Tensor, eig: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        U, s = eig
        coef = s / (self.eps + s)
        return (X - U @ ((U.T @ X) * coef[:, None])) / self.eps

    @torch.no_grad()
    def precondition(self) -> None:
        if not self.initialized or self.factor is None:
            return
        for name, module in self.layers:
            if name not in self.factor.A_eig or name not in self.factor.G_eig:
                continue
            if module.weight.grad is None:
                continue
            Gm = module.weight.grad.reshape(module.weight.shape[0], -1).float()
            Gm = self._apply_inv_left(Gm, self.factor.G_eig[name])
            Gm = self._apply_inv_right(Gm, self.factor.A_eig[name])
            module.weight.grad.copy_(Gm.reshape_as(module.weight.grad).to(module.weight.grad.dtype))
            if module.bias is not None and module.bias.grad is not None:
                b = module.bias.grad.float().unsqueeze(1)
                b = self._apply_inv_left(b, self.factor.G_eig[name])
                module.bias.grad.copy_(b.squeeze(1).to(module.bias.grad.dtype))


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


DEFAULT_TASKS = 10
DEFAULT_CPT = 10


def _find_cifar_root():
    """Locate a pre-extracted CIFAR-100 (cifar-100-python/) without downloading:
    checks ./data and any attached Kaggle dataset under /kaggle/input/*.  Attach
    a CIFAR-100 dataset to the notebook to avoid the flaky torchvision download."""
    import glob, os
    cands = ["./data"] + [os.path.dirname(p) for p in
             glob.glob("/kaggle/input/*/cifar-100-python") +
             glob.glob("/kaggle/input/*/*/cifar-100-python")]
    for root in cands:
        if os.path.isdir(os.path.join(root, "cifar-100-python")):
            return root
    return None


def get_split_cifar100(tasks: int, cpt: int):
    import os, time as _t, torchvision
    import torchvision.transforms as T
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    def build(download):
        root = _find_cifar_root() or "./data"
        tr = torchvision.datasets.CIFAR100(root, train=True, download=download, transform=tf)
        te = torchvision.datasets.CIFAR100(root, train=False, download=download, transform=tf)
        return tr, te

    root = _find_cifar_root()
    if root is not None:                                   # already present -> no network
        log(f"CIFAR-100 found locally at {root}; skipping download")
        tr, te = build(download=False)
    else:                                                  # download with retries (Kaggle nets flake)
        last = None
        for attempt in range(4):
            try:
                tr, te = build(download=True); break
            except Exception as e:                         # ConnectionReset / URLError / timeout
                last = e
                log(f"CIFAR-100 download attempt {attempt+1}/4 failed: {type(e).__name__}; retrying")
                _t.sleep(5 * (attempt + 1))
        else:
            raise SystemExit(
                "Could not download CIFAR-100 after 4 tries. Two fixes: (1) enable "
                "Internet in the Kaggle notebook (Settings -> Internet -> On); or (2) "
                "attach a CIFAR-100 dataset (with cifar-100-python/) as an input -- this "
                f"script finds it automatically under /kaggle/input/*.  Last error: {last}")

    def split(ds):
        out = [[] for _ in range(tasks)]
        for x, y in ds:
            out[y // cpt].append((x, y))
        return out

    return split(tr), split(te)


def make_loader(items, batch_size: int, shuffle: bool):
    Xs = torch.stack([x for x, _ in items])
    ys = torch.tensor([y for _, y in items])
    idx = torch.randperm(len(items)) if shuffle else torch.arange(len(items))
    for i in range(0, len(items), batch_size):
        j = idx[i:i + batch_size]
        yield Xs[j].to(DEV), ys[j].to(DEV)


@torch.no_grad()
def evaluate(model: nn.Module, te_task, batch_size: int) -> float:
    was_training = model.training
    model.eval()
    c = 0
    n = 0
    for x, y in make_loader(te_task, batch_size, False):
        c += (model(x).argmax(1) == y).sum().item()
        n += len(y)
    if was_training:
        model.train()
    return c / max(1, n)


def summarize_metrics(acc: np.ndarray, cpt: int) -> Dict[str, float]:
    T = acc.shape[0]
    chance = 1.0 / cpt
    learn = float(np.mean([acc[j, j] for j in range(T)]))
    final = float(np.mean([acc[T - 1, j] for j in range(T)]))
    forget = float(np.mean([max(acc[i, j] for i in range(j, T)) - acc[T - 1, j] for j in range(T - 1)]))
    bwt = float(np.mean([acc[T - 1, j] - acc[j, j] for j in range(T - 1)]))
    L = learn - chance
    omega = float(1.0 - forget / L) if L > 1e-6 else float("nan")
    return {"avg_acc": final, "learning_acc": learn, "forgetting": forget, "bwt": bwt, "omega": omega}


MEM_METHODS = {
    "exact-stale": {"mode": "stale"},
    "exact-transport": {"mode": "transport"},
    "exact-functional": {"mode": "functional"},
}


def run_one(method: str, tr, te, args, seed: int) -> Dict[str, Any]:
    run_t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = Net(ncls=args.tasks * args.cpt).to(DEV)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    ewc = EWC() if method == "ewc" else None
    metric = None
    reservoir = AnchorReservoir(cap=args.res_cap, protect_delay=args.protect_delay)

    if method in MEM_METHODS:
        metric = KFACAnchorMetric(
            model=model,
            eps=args.eps,
            rank_a=args.rank_a,
            rank_g=args.rank_g,
            ridge=args.ridge,
            transport_blend=args.transport_blend,
            sketch_rows=args.sketch_rows,
        )

    acc = np.zeros((args.tasks, args.tasks), dtype=np.float64)
    step = 0
    task_logs: List[Dict[str, Any]] = []

    refresh_period = {
        "naive": None,
        "ewc": None,
        "exact-stale": None,
        "exact-transport": args.refresh,
        "exact-functional": 1,
    }[method]

    for t in range(args.tasks):
        task_t0 = time.time()
        epoch_losses = []
        for ep in range(args.epochs):
            epoch_loss_sum = 0.0
            epoch_batches = 0
            for x, y in make_loader(tr[t], args.batch_size, True):
                if metric is not None:
                    need_init = not metric.initialized
                    need_refresh = refresh_period is not None and step % refresh_period == 0
                    if need_init or need_refresh:
                        metric.maybe_refresh(reservoir, step, MEM_METHODS[method]["mode"], args.anchor_batch)

                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                if ewc is not None and ewc.means:
                    loss = loss + args.ewc_lambda * ewc.penalty(model)
                loss.backward()
                if metric is not None and metric.initialized:
                    metric.precondition()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                reservoir.add_batch(x, logits, step)
                step += 1
                epoch_loss_sum += float(loss.item())
                epoch_batches += 1
            epoch_losses.append(epoch_loss_sum / max(1, epoch_batches))

        if ewc is not None:
            ewc.consolidate(model)
            ewc.estimate_fisher(model, tr[t], args.batch_size)

        task_accs = {}
        for j in range(t + 1):
            task_accs[f"task_{j}"] = float(evaluate(model, te[j], args.batch_size))
            acc[t, j] = task_accs[f"task_{j}"]

        metrics_now = summarize_metrics(acc[: t + 1, : t + 1], args.cpt)
        task_log = {
            "task_index": t,
            "epoch_losses": epoch_losses,
            "task_eval": task_accs,
            "mean_acc_so_far": float(np.mean(acc[t, : t + 1])),
            "metrics_so_far": metrics_now,
            "reservoir": reservoir.stats(step),
            "refresh_count": metric.refresh_count if metric is not None else 0,
            "anchor_backward_calls": metric.anchor_backward_calls if metric is not None else 0,
            "task_wallclock_sec": round(time.time() - task_t0, 4),
        }
        task_logs.append(task_log)
        log(f"[{method}] task {t}: acc-so-far = {task_log['mean_acc_so_far']:.3f}", method=method, seed=seed, task=t, task_log=task_log)

    final_metrics = summarize_metrics(acc, args.cpt)
    run_record = {
        "method": method,
        "seed": seed,
        "device": DEV,
        "run_wallclock_sec": round(time.time() - run_t0, 4),
        "final_metrics": final_metrics,
        "accuracy_matrix": acc.tolist(),
        "task_logs": task_logs,
        "refresh_count": metric.refresh_count if metric is not None else 0,
        "anchor_backward_calls": metric.anchor_backward_calls if metric is not None else 0,
        "transport_diag": metric.transport_diag if metric is not None else [],
        "refresh_log": metric.refresh_log if metric is not None else [],
        "layer_factor_summary_final": metric._factor_layer_summary(metric.factor.A, metric.factor.G) if (metric is not None and metric.factor is not None) else {},
        "reservoir_final": reservoir.stats(step),
    }

    if metric is not None:
        metric.remove()
    return run_record


def aggregate_method_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    final_metrics = [r["final_metrics"] for r in runs]
    summary = {k: float(np.mean([m[k] for m in final_metrics])) for k in ["avg_acc", "learning_acc", "forgetting", "bwt", "omega"]}
    summary["refresh_count"] = float(np.mean([r["refresh_count"] for r in runs]))
    summary["anchor_backward_calls"] = float(np.mean([r["anchor_backward_calls"] for r in runs]))
    if runs and runs[0]["transport_diag"]:
        diags = [d for r in runs for d in r["transport_diag"]]
        for key in ["A_rel_err", "G_rel_err", "A_overlap", "G_overlap"]:
            summary[key] = float(np.mean([d[key] for d in diags]))
    layer_names = sorted({lname for r in runs for lname in r.get("layer_factor_summary_final", {}).keys()})
    if layer_names:
        layer_summary = {}
        keys = ["A_dim", "G_dim", "A_trace", "G_trace", "A_rank_used", "G_rank_used", "A_top_mass_ratio", "G_top_mass_ratio", "A_top_eig", "G_top_eig"]
        for lname in layer_names:
            vals = [r["layer_factor_summary_final"][lname] for r in runs if lname in r.get("layer_factor_summary_final", {})]
            if vals:
                layer_summary[lname] = {k: float(np.mean([v[k] for v in vals])) for k in keys}
        summary["layer_factor_summary_final"] = layer_summary
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, default=10)
    p.add_argument("--cpt", type=int, default=10)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--eps", type=float, default=0.10)
    p.add_argument("--rank-a", type=int, default=64)
    p.add_argument("--rank-g", type=int, default=32)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--refresh", type=int, default=50)
    p.add_argument("--res-cap", type=int, default=400)
    p.add_argument("--protect-delay", type=int, default=100)
    p.add_argument("--anchor-batch", type=int, default=128)
    p.add_argument("--sketch-rows", type=int, default=1024)
    p.add_argument("--transport-blend", type=float, default=0.25)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--ewc-lambda", type=float, default=40.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    args, unknown = p.parse_known_args()
    return args


if __name__ == "__main__":
    started_at = time.time()
    args = parse_args()
    log(f"{SCRIPT_VERSION}; device {DEV}", script_version=SCRIPT_VERSION, device=DEV)
    tr, te = get_split_cifar100(args.tasks, args.cpt)
    log(f"Split-CIFAR-100: {args.tasks} tasks x {args.cpt} classes", tasks=args.tasks, cpt=args.cpt)

    methods = ["naive", "ewc", "exact-stale", "exact-transport", "exact-functional"]
    method_runs = {}
    method_summaries = {}

    for method in methods:
        runs = []
        for seed in args.seeds:
            log(f"START {method} seed={seed}", method=method, seed=seed)
            run_record = run_one(method, tr, te, args, seed)
            runs.append(run_record)
            log(
                f"END {method} seed={seed}: avg_acc={run_record['final_metrics']['avg_acc']:.3f} Omega={run_record['final_metrics']['omega']:+.3f}",
                method=method,
                seed=seed,
                final_metrics=run_record["final_metrics"],
                refresh_count=run_record["refresh_count"],
                anchor_backward_calls=run_record["anchor_backward_calls"],
            )
        method_runs[method] = runs
        method_summaries[method] = aggregate_method_runs(runs)
        s = method_summaries[method]
        log(
            f"== {method:18s} avg_acc {s['avg_acc']:.3f} learn {s['learning_acc']:.3f} forget {s['forgetting']:.3f} BWT {s['bwt']:+.3f} Omega {s['omega']:+.3f} refresh {s['refresh_count']:.1f} anchor_bw {s['anchor_backward_calls']:.1f}",
            method=method,
            summary=s,
        )

    payload = {
        "script_version": SCRIPT_VERSION,
        "device": DEV,
        "started_unix": started_at,
        "finished_unix": time.time(),
        "total_wallclock_sec": round(time.time() - started_at, 4),
        "args": vars(args),
        "method_summaries": method_summaries,
        "method_runs": method_runs,
        "stdout_lines": LOGGER.lines,
        "event_log": LOGGER.events,
    }

    with open("output/memflow_deep_exact_logged_results.json", "w") as f:
        json.dump(to_python(payload), f, indent=1)

    print("\n" + "=" * 86)
    print("Omega = 1 - forgetting / learning-above-chance; full structured logs are in JSON")
    print("=" * 86)
    for method, s in method_summaries.items():
        print(f" {method:18s} avg_acc={s['avg_acc']:.3f} Omega={s['omega']:+.3f} refresh={s['refresh_count']:.1f} anchor_bw={s['anchor_backward_calls']:.1f}")
    log("ALL DONE -> output/memflow_deep_exact_logged_results.json")
