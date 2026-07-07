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

SCRIPT_VERSION = "memflow-deep-exact-logged-v2-maskedTIL-tracenorm"
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
        if not torch.isfinite(rows).all():
            rows = torch.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)
        C = (rows.T @ rows) / max(1, rows.shape[0])
        C = 0.5 * (C + C.T)
        return C

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
        r = min(rank, cov.shape[0])
        if r == 0:
            return cov.new_zeros((cov.shape[0], 0)), cov.new_zeros((0,))

        M = cov.detach().float()
        M = 0.5 * (M + M.T)

        if not torch.isfinite(M).all():
            raise RuntimeError("Non-finite entries in covariance matrix before eigh")

        eye = torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
        jitter_scales = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
        last_err = None

        for eps in jitter_scales:
            try:
                A = M if eps == 0.0 else (M + eps * eye)
                w, V = torch.linalg.eigh(A)
                return V[:, -r:], w[-r:].clamp(min=0)
            except Exception as e:
                last_err = e

        try:
            A_cpu = M.cpu().double()
            A_cpu = 0.5 * (A_cpu + A_cpu.T)
            eye_cpu = torch.eye(A_cpu.shape[0], device=A_cpu.device, dtype=A_cpu.dtype)
            for eps in [1e-8, 1e-6, 1e-4]:
                try:
                    w, V = torch.linalg.eigh(A_cpu + eps * eye_cpu)
                    V = V[:, -r:].to(cov.device, dtype=cov.dtype)
                    w = w[-r:].clamp(min=0).to(cov.device, dtype=cov.dtype)
                    return V, w
                except Exception as e:
                    last_err = e
        except Exception as e:
            last_err = e

        raise RuntimeError(f"_top_eig failed after GPU and CPU fallback: {last_err}")

    @staticmethod
    def _trace_normalize(cov: torch.Tensor) -> torch.Tensor:
        """Scale a factor to unit MEAN eigenvalue (trace/dim = 1).  This makes eps
        a scale-invariant relative-damping knob for BOTH factors: without it the
        output-gradient factor G -- whose raw scale depends on the loss reduction
        (mse-mean divides grads by batch*classes) -- has eigenvalues ~1e-6 << eps,
        so (G+eps I)^{-1} degenerates to pure 1/eps scaling and contributes no
        directional protection.  The already-saturated A factor is essentially
        unchanged (its top eigenvalues stay >> eps after normalization)."""
        d = cov.shape[0]
        tr = torch.trace(cov)
        return cov * (d / (tr + 1e-12)) if tr > 0 else cov

    def _make_factor_state(self, A: Dict[str, torch.Tensor], G: Dict[str, torch.Tensor]) -> FactorState:
        return FactorState(
            A=A,
            G=G,
            A_eig={name: self._top_eig(self._trace_normalize(cov), self.rank_a)
                   for name, cov in A.items()},
            G_eig={name: self._top_eig(self._trace_normalize(cov), self.rank_g)
                   for name, cov in G.items()},
        )

    @staticmethod
    def _safe_eigvalsh(cov: torch.Tensor) -> torch.Tensor:
        M = cov.detach().float()
        M = 0.5 * (M + M.T)
        if not torch.isfinite(M).all():
            M = torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

        eye = torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
        last_err = None
        for eps in [0.0, 1e-6, 1e-5, 1e-4, 1e-3]:
            try:
                A = M if eps == 0.0 else (M + eps * eye)
                return torch.linalg.eigvalsh(A).clamp(min=0)
            except Exception as e:
                last_err = e

        A_cpu = M.cpu().double()
        A_cpu = 0.5 * (A_cpu + A_cpu.T)
        eye_cpu = torch.eye(A_cpu.shape[0], device=A_cpu.device, dtype=A_cpu.dtype)
        for eps in [1e-8, 1e-6, 1e-4]:
            try:
                w = torch.linalg.eigvalsh(A_cpu + eps * eye_cpu)
                return w.clamp(min=0).to(cov.device, dtype=cov.dtype)
            except Exception as e:
                last_err = e

        raise RuntimeError(f"_safe_eigvalsh failed after GPU and CPU fallback: {last_err}")


    def _factor_layer_summary(self, A: Dict[str, torch.Tensor], G: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        out = {}
        for name in sorted(set(A.keys()) & set(G.keys())):
            wa = self._safe_eigvalsh(A[name])
            wg = self._safe_eigvalsh(G[name])
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


def find_cifar_root():
    """Locate CIFAR-100 without relying on one exact Kaggle folder layout.

    Supported layouts:
    - ./data/cifar-100-python/
    - /kaggle/input/<slug>/train,test,meta
    - /kaggle/input/<slug>/cifar-100-python/train,test,meta
    - deeper nested variants under /kaggle/input
    """
    import glob
    import os

    def has_cifar_files(p: str) -> bool:
        return all(os.path.isfile(os.path.join(p, f)) for f in ["train", "test", "meta"])

    candidates = ["./data"]
    candidates += glob.glob("/kaggle/input/*")
    candidates += glob.glob("/kaggle/input/*/*")
    candidates += glob.glob("/kaggle/input/*/*/*")

    seen = set()
    ordered = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    for root in ordered:
        if not os.path.isdir(root):
            continue
        if has_cifar_files(root):
            return root, "flat"
        nested = os.path.join(root, "cifar-100-python")
        if os.path.isdir(nested) and has_cifar_files(nested):
            return nested, "nested"

    return None, None


def load_cifar100_pickles(base: str):
    import os
    import pickle
    import numpy as np

    def unpickle(fp):
        with open(fp, "rb") as fo:
            obj = pickle.load(fo, encoding="bytes")
        return obj

    if not os.path.isdir(base):
        raise FileNotFoundError(f"CIFAR root does not exist: {base}")

    required = [os.path.join(base, f) for f in ["train", "test", "meta"]]
    missing = [p for p in required if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Missing CIFAR-100 files under {base}: {missing}")

    trd = unpickle(os.path.join(base, "train"))
    ted = unpickle(os.path.join(base, "test"))

    class DS:
        pass

    tr = DS()
    te = DS()

    tr.data = np.asarray(trd[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    te.data = np.asarray(ted[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    tr.targets = [int(y) for y in trd[b"fine_labels"]]
    te.targets = [int(y) for y in ted[b"fine_labels"]]
    return tr, te


def get_split_cifar100(tasks: int, cpt: int):
    import socket
    import time as _t
    import torchvision

    socket.setdefaulttimeout(30)

    def raw(root, download: bool):
        tr = torchvision.datasets.CIFAR100(root, train=True, download=download, transform=None)
        te = torchvision.datasets.CIFAR100(root, train=False, download=download, transform=None)
        return tr, te

    root, mode = find_cifar_root()
    if root is not None:
        log(f"CIFAR-100 found locally at {root} (mode={mode}); skipping download")
        tr, te = load_cifar100_pickles(root)
    else:
        log("CIFAR-100 not found locally; downloading (needs Kaggle Internet ON)")
        last = None
        for attempt in range(4):
            try:
                tr, te = raw("./data", True)
                break
            except Exception as e:
                last = e
                log(f"download attempt {attempt+1}/4 failed: {type(e).__name__}: {e}; retrying")
                _t.sleep(5 * (attempt + 1))
        else:
            raise SystemExit(
                "Could not load CIFAR-100 after 4 tries. Supported local layouts are: "
                "(1) /kaggle/input/<dataset>/train,test,meta, "
                "(2) /kaggle/input/<dataset>/cifar-100-python/train,test,meta, "
                "or (3) enable Internet so torchvision can download. "
                f"Last error: {last}"
            )

    mean = torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1)
    std = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1)
    def prep(ds):
        X = torch.from_numpy(ds.data).float().permute(0, 3, 1, 2).div_(255.0).sub_(mean).div_(std)
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


@torch.no_grad()
def evaluate(model: nn.Module, te_task, batch_size: int,
             task_id: Optional[int] = None, cpt: Optional[int] = None) -> float:
    """Task-incremental (masked) accuracy when task_id/cpt are given: the argmax
    is restricted to task_id's own class block.  Without masking, a single shared
    100-way head under class-incremental training collapses to EXACTLY 0 on every
    past task (cross-entropy on the current task drives all old-class logits down),
    which measures head collapse -- unfixable by any gradient-geometry method,
    needs replay -- not the FEATURE forgetting this method protects.  Task-IL is
    the correct protocol for a feature-protection method."""
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
            task_accs[f"task_{j}"] = float(
                evaluate(model, te[j], args.batch_size, task_id=j, cpt=args.cpt))
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

    import os
    os.makedirs("output", exist_ok=True)
    with open("output/memflow_deep_exact_logged_results.json", "w") as f:
        json.dump(to_python(payload), f, indent=1)

    print("\n" + "=" * 86)
    print("Omega = 1 - forgetting / learning-above-chance; full structured logs are in JSON")
    print("=" * 86)
    for method, s in method_summaries.items():
        print(f" {method:18s} avg_acc={s['avg_acc']:.3f} Omega={s['omega']:+.3f} refresh={s['refresh_count']:.1f} anchor_bw={s['anchor_backward_calls']:.1f}")
    log("ALL DONE -> output/memflow_deep_exact_logged_results.json")