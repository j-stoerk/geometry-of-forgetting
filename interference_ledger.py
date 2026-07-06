# =====================================================================
#  interference_ledger.py -- a standard "interference ledger" for
#  adaptive models, and a geometry-driven controller that turns the
#  measured geometry into online decisions (share / separate / replay /
#  expand / defer / merge / route).
#
#  This is the reference implementation of the design calculus in
#  "Forgetting as Interference in Continual Learning".  Everything here
#  is exact in the frozen-feature / function-space regime the paper
#  validates; the same estimators are the online surrogates under drift.
#  Dependency: numpy only (portable; torch adapters are trivial).
#
#  Core objects
#    GeometryTracker   -- recursive Gauss-Newton / Frequent-Directions
#                         estimate of the occupied functional geometry.
#    InterferenceLedger-- the metric panel logged during training.
#    InterferenceController -- detect shift -> reject OOD -> estimate
#                         geometry -> pick action -> update, with
#                         uncertainty-aware conservative defaults.
#
#  See LEDGER.md for the metric definitions, units, regime matrix
#  (exact / bounded / empirical / heuristic), and failure cases.
# =====================================================================
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np


# --------------------------------------------------------------------- utils
def _orthonormal(M: np.ndarray) -> np.ndarray:
    if M.shape[1] == 0:
        return M
    Q, _ = np.linalg.qr(M)
    return Q

def principal_overlap(U: np.ndarray, V: np.ndarray) -> float:
    """Normalized subspace overlap ||U^T V||_F^2 / min(rank).  In [0, 1].
    1 = identical span, 0 = orthogonal.  (Exact geometric quantity.)"""
    if U.shape[1] == 0 or V.shape[1] == 0:
        return 0.0
    r = min(U.shape[1], V.shape[1])
    return float(np.linalg.norm(U.T @ V, "fro") ** 2 / r)

def commutator_velocity(U: np.ndarray, V: np.ndarray) -> float:
    """Frobenius commutator ||P_U P_V - P_V P_U||_F of the projectors.
    A boundary/drift-rate signal: 0 = no change, large = subspace slew."""
    if U.shape[1] == 0 or V.shape[1] == 0:
        return 0.0
    Pu, Pv = U @ U.T, V @ V.T
    return float(np.linalg.norm(Pu @ Pv - Pv @ Pu, "fro"))

def effective_rank(evals: np.ndarray) -> float:
    """Entropy (participation) effective rank of a spectrum: exp(H(p)),
    p = lambda / sum(lambda).  Continuous 'how many directions are live'."""
    lam = np.clip(np.asarray(evals, float), 0, None)
    s = lam.sum()
    if s <= 0:
        return 0.0
    p = lam / s
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


# --------------------------------------------------------------- GeometryTracker
class GeometryTracker:
    """Online estimate of the occupied functional geometry Sigma = E[phi phi^T]
    (frozen features) or E[J J^T] (Gauss-Newton, drifting features).

    method='recursive' : decayed covariance  C <- gamma C + mean(phi phi^T),
                         O(d^2) memory, exact top-r under stationarity.
    method='fd'        : Frequent-Directions sketch, O(ell d) memory,
                         the ell>=2r cheap surrogate (paper Sec. S8/S11).
    """
    def __init__(self, dim: int, rank: int = 8, decay: float = 0.6,
                 method: str = "recursive", ell: Optional[int] = None):
        self.d, self.r, self.gamma, self.method = dim, rank, decay, method
        self.ell = ell or 2 * rank
        self.C = np.zeros((dim, dim)) if method == "recursive" else None
        self.sketch = np.zeros((self.ell, dim)) if method == "fd" else None
        self.U_prev: Optional[np.ndarray] = None
        self.U: Optional[np.ndarray] = None
        self.n_updates = 0

    def update(self, Phi: np.ndarray) -> None:
        """Fold a batch of feature/Jacobian rows Phi (n x d) into the estimate."""
        Phi = np.atleast_2d(np.asarray(Phi, float))
        if self.method == "recursive":
            self.C = self.gamma * self.C + Phi.T @ Phi / max(len(Phi), 1)
        else:                                              # Frequent Directions (decayed)
            B = np.vstack([np.sqrt(self.gamma) * self.sketch, Phi / np.sqrt(max(len(Phi), 1))])
            _, s, Vt = np.linalg.svd(B, full_matrices=False)
            delta = s[self.ell - 1] ** 2 if len(s) >= self.ell else 0.0
            self.sketch = np.sqrt(np.maximum(s[:self.ell] ** 2 - delta, 0))[:, None] * Vt[:self.ell]
        self.U_prev = self.U
        self.U = self.subspace(self.r)
        self.n_updates += 1

    def spectrum(self) -> np.ndarray:
        if self.method == "recursive":
            return np.clip(np.linalg.eigvalsh(self.C)[::-1], 0, None)
        s = np.linalg.svd(self.sketch, compute_uv=False)
        return s ** 2

    def subspace(self, r: Optional[int] = None) -> np.ndarray:
        r = r or self.r
        if self.method == "recursive":
            w, V = np.linalg.eigh(self.C)
            return V[:, -r:] if r > 0 else V[:, :0]
        Vt = np.linalg.svd(self.sketch, full_matrices=False)[2]
        return Vt[:r].T

    def occupied_rank(self) -> float:
        return effective_rank(self.spectrum())

    def drift_velocity(self) -> float:
        if self.U_prev is None or self.U is None:
            return 0.0
        return commutator_velocity(self.U, self.U_prev)


# ------------------------------------------------------------- InterferenceLedger
@dataclass
class LedgerRow:
    step: int
    interference_energy: float          # 1/2 g^T Sigma g  (harm of the raw update)
    interference_rate: float            # <g, grad L_protected>  (signed harm rate)
    shared_density: float               # fraction of past tasks the new one overlaps
    max_overlap: float                  # strongest single-task overlap
    floor_estimate: float               # predicted irreducible distortion floor
    occupied_rank: float                # effective rank of the occupied geometry
    free_capacity: float                # (cap - occupied)/cap in [0,1], or nan
    drift_velocity: float               # subspace slew (boundary/drift signal)
    ood_ratio: float                    # ||UU^T g||/||g|| (1=in-distribution, low=OOD)
    replay_value: float                 # RECOVERABLE excess above the floor (replay
                                        # cannot beat the floor), 0 if no memory
    confidence: float                   # estimator confidence in [0,1]


class InterferenceLedger:
    """The measurable primitive.  Feed it per-step gradients / features / an
    optional protected reference; read a full metric panel.  Every quantity is
    the same object the paper's theorems use, so decisions are geometry-driven,
    not heuristic."""
    def __init__(self, dim: int, rank: int = 8, decay: float = 0.6,
                 capacity: Optional[int] = None, sstar: float = 0.5,
                 tracker_method: str = "recursive", warmup_steps: int = 8):
        self.d, self.r, self.sstar = dim, rank, sstar
        self.capacity = capacity
        self.warmup_steps = warmup_steps                # updates to full confidence
        #   ~20 for per-batch training, ~3-5 for per-task streams
        self.tracker = GeometryTracker(dim, rank, decay, tracker_method)
        self.task_bases: List[np.ndarray] = []          # per stored task/domain top-r basis
        self.task_targets: List[np.ndarray] = []        # optional stored heads (for floor/align)
        self.rows: List[LedgerRow] = []
        self._step = 0

    # -- geometry maintenance --
    def register_task(self, Phi: np.ndarray, target: Optional[np.ndarray] = None) -> None:
        """Store a finished task's occupied subspace (and optional head)."""
        B = self.tracker.subspace(self.r) if Phi is None else \
            _top_r(np.atleast_2d(Phi), self.r)
        self.task_bases.append(B)
        if target is not None:
            self.task_targets.append(np.asarray(target, float))

    # -- individual metrics (all exact in the frozen/function-space regime) --
    def interference_energy(self, g: np.ndarray) -> float:
        if self.tracker.method == "recursive":
            return float(0.5 * g @ self.tracker.C @ g)
        U = self.tracker.subspace(self.r); lam = self.tracker.spectrum()[:U.shape[1]]
        c = U.T @ g
        return float(0.5 * np.sum(lam * c ** 2))

    def interference_rate(self, g: np.ndarray, grad_protected: np.ndarray) -> float:
        """<g, grad L_u>: the sign is the exact continuous-time harm direction
        (paper Sec. S15).  <0 => descending -g would increase L_u (conflict)."""
        return float(g @ grad_protected)

    def shared_density(self, B_new: np.ndarray) -> tuple:
        if not self.task_bases:
            return 0.0, 0.0
        ov = [principal_overlap(Bt, B_new) for Bt in self.task_bases]
        return float(np.mean([o >= self.sstar for o in ov])), float(np.max(ov))

    def floor_estimate(self, B_new: np.ndarray, target_new: Optional[np.ndarray]) -> float:
        """Predicted distortion floor: energy on shared active directions weighted
        by target disagreement (paper Thm 4 / Cor N3).  0 if disjoint or aligned."""
        if not self.task_bases or target_new is None or not self.task_targets:
            return 0.0
        fl = 0.0
        for Bt, wt in zip(self.task_bases, self.task_targets):
            if Bt.shape[1] == 0:
                continue
            shared = _shared_directions(Bt, B_new)
            if shared.shape[1] == 0:
                continue
            gap = shared.T @ (target_new - wt)
            fl += float(gap @ gap)
        return fl / max(len(self.task_bases), 1)

    def ood_ratio(self, g: np.ndarray) -> float:
        U = self.tracker.subspace(self.r)
        if U.shape[1] == 0 or np.linalg.norm(g) < 1e-12:
            return 1.0
        return float(np.linalg.norm(U @ (U.T @ g)) / np.linalg.norm(g))

    def free_capacity(self) -> float:
        if self.capacity is None:
            return float("nan")
        return float(max(self.capacity - self.tracker.occupied_rank(), 0) / self.capacity)

    def confidence(self) -> float:
        """Estimator confidence: rises with observations, falls with drift.
        Conservative below ~0.5 (controller then prefers protection)."""
        warmup = min(self.tracker.n_updates / float(self.warmup_steps), 1.0)
        drift_pen = 1.0 / (1.0 + 3.0 * self.tracker.drift_velocity())
        return float(warmup * drift_pen)

    def log(self, g: np.ndarray, Phi_new: Optional[np.ndarray] = None,
            target_new: Optional[np.ndarray] = None,
            grad_protected: Optional[np.ndarray] = None,
            memory_available: bool = True) -> LedgerRow:
        self._step += 1
        B_new = _top_r(np.atleast_2d(Phi_new), self.r) if Phi_new is not None \
            else self.tracker.subspace(self.r)
        dens, mx = self.shared_density(B_new)
        floor = self.floor_estimate(B_new, target_new)
        energy = self.interference_energy(g)
        # Replay value = RECOVERABLE excess: full rehearsal lands a task AT its
        # floor, never below (evaluate_replay_policy.py, P1), so the floor part
        # of the predicted damage cannot be bought back -- only the rest can.
        recoverable = max(energy - floor, 0.0)
        row = LedgerRow(
            step=self._step,
            interference_energy=energy,
            interference_rate=(self.interference_rate(g, grad_protected)
                               if grad_protected is not None else float("nan")),
            shared_density=dens, max_overlap=mx, floor_estimate=floor,
            occupied_rank=self.tracker.occupied_rank(),
            free_capacity=self.free_capacity(),
            drift_velocity=self.tracker.drift_velocity(),
            ood_ratio=self.ood_ratio(g),
            replay_value=(recoverable if memory_available else 0.0),
            confidence=self.confidence(),
        )
        self.rows.append(row)
        return row

    def to_dict(self) -> List[Dict]:
        return [vars(r) for r in self.rows]


# --------------------------------------------------------- InterferenceController
@dataclass
class Decision:
    action: str                  # share | project | replay | expand | defer | route | skip
    reason: str
    confidence: float
    metrics: LedgerRow


class InterferenceController:
    """Hierarchical geometry-driven controller.

    Order per step:  detect shift -> reject OOD/noise -> estimate geometry ->
    pick action -> update state.  Actions and their triggers (paper-derived):

      skip     : OOD/noise batch (ood_ratio low AND extreme gradient) -> don't adapt.
      project  : measured conflict (interference_rate < 0, or overlap high with
                 disagreeing target) -> null-space route the update.
      share    : measured transfer (interference_rate >= 0, overlap high, floor low)
                 -> keep the update whole.
      replay   : predicted floor positive AND memory budget available.
      expand   : occupied rank near capacity (free_capacity < eps).
      defer    : overlap high but confidence low -> protect now, revisit later.

    Uncertainty handling: confidence < conf_floor forces the safe action (project),
    because a wrong 'share' costs retention while a wrong 'project' costs only
    (recoverable) transfer.  The threshold-free functional gate is the default
    share/project decision -- it reads the SIGN of measured interference, no s*.
    """
    def __init__(self, ledger: InterferenceLedger, memory_budget: int = 0,
                 conf_floor: float = 0.5, ood_extreme: float = 3.0,
                 capacity_eps: float = 0.05, use_threshold_gate: bool = False):
        self.L = ledger
        self.memory_budget = memory_budget
        self.conf_floor = conf_floor
        self.ood_extreme = ood_extreme
        self.capacity_eps = capacity_eps
        self.use_threshold_gate = use_threshold_gate
        self.trace: List[Decision] = []
        self._grad_scale = 1.0        # running gradient-magnitude scale (for OOD)

    def decide(self, g: np.ndarray, Phi_new=None, target_new=None,
               grad_protected=None) -> Decision:
        gn = float(np.linalg.norm(g))
        self._grad_scale = 0.9 * self._grad_scale + 0.1 * gn
        row = self.L.log(g, Phi_new, target_new, grad_protected,
                         memory_available=self.memory_budget > 0)

        def D(a, r): return Decision(a, r, row.confidence, row)

        # 1. OOD / noise rejection (precedence: skip before anything else)
        if row.ood_ratio < 0.5 and gn > self.ood_extreme * (self._grad_scale + 1e-9):
            return self._commit(D("skip", "orthogonal-and-extreme gradient (OOD/noise)"))

        # 2. capacity: expand before forcing conflict into a full space
        if not np.isnan(row.free_capacity) and row.free_capacity < self.capacity_eps:
            return self._commit(D("expand", f"occupied rank near capacity "
                                             f"(free={row.free_capacity:.2f})"))

        # 3. uncertainty: low confidence -> protect (safe default)
        if row.confidence < self.conf_floor and self.L.task_bases:
            return self._commit(D("defer", f"low estimator confidence "
                                            f"({row.confidence:.2f}) -> protect now"))

        # 4. the core share/project decision
        conflict = self._is_conflict(row, grad_protected)
        if self.L.task_bases and conflict:
            # replay is offered when a positive floor can be bought down and memory exists
            if row.replay_value > 1e-6 and self.memory_budget > 0:
                return self._commit(D("replay", f"positive floor {row.floor_estimate:.3f} "
                                                 f"with memory available"))
            return self._commit(D("project", self._conflict_reason(row, grad_protected)))
        if self.L.task_bases:
            return self._commit(D("share", self._share_reason(row, grad_protected)))
        return self._commit(D("share", "first task: nothing to protect"))

    # -- decision internals --
    def _is_conflict(self, row: LedgerRow, grad_protected) -> bool:
        if not self.use_threshold_gate and grad_protected is not None \
                and not np.isnan(row.interference_rate):
            return row.interference_rate < 0.0            # threshold-free functional gate
        # fallback: similarity-threshold gate (overlap high AND floor positive)
        return row.max_overlap >= self.L.sstar and row.floor_estimate > 1e-6

    def _conflict_reason(self, row, gp):
        if gp is not None and not np.isnan(row.interference_rate):
            return f"interference rate {row.interference_rate:+.3f} < 0 (measured conflict)"
        return f"overlap {row.max_overlap:.2f} >= s* with floor {row.floor_estimate:.3f}"

    def _share_reason(self, row, gp):
        if gp is not None and not np.isnan(row.interference_rate):
            return f"interference rate {row.interference_rate:+.3f} >= 0 (measured transfer)"
        return f"overlap {row.max_overlap:.2f} < s* or floor ~ 0"

    def _commit(self, d: Decision) -> Decision:
        self.trace.append(d)
        return d


# ------------------------------------------------------------------- helpers
def _top_r(Phi: np.ndarray, r: int) -> np.ndarray:
    if Phi.shape[0] == 0 or r == 0:
        return np.zeros((Phi.shape[1] if Phi.ndim == 2 else 0, 0))
    S = Phi.T @ Phi / len(Phi)
    w, V = np.linalg.eigh(S)
    return V[:, -r:]

def _shared_directions(Ba: np.ndarray, Bb: np.ndarray, tol: float = 0.3) -> np.ndarray:
    """Directions in span(Ba) that also lie substantially in span(Bb)."""
    if Ba.shape[1] == 0 or Bb.shape[1] == 0:
        return np.zeros((Ba.shape[0], 0))
    M = Ba.T @ (Bb @ Bb.T) @ Ba
    w, V = np.linalg.eigh(M)
    keep = w > tol
    return Ba @ V[:, keep] if keep.any() else np.zeros((Ba.shape[0], 0))


# ---------------------------------------------------------------------
#  Bregman / cross-entropy forms (NOTE_bregman_generalization.md).
#  The quadratic metrics above are the Gaussian case; for categorical
#  outputs (classification, LLMs) the exact forms are these.
# ---------------------------------------------------------------------
def kl_interference(p_old: np.ndarray, p_new: np.ndarray) -> float:
    """Exact CE forgetting on a converged task: E[KL(p_old || p_new)].

    p_old, p_new: (n, K) predictive distributions of the protected model
    and the updated model on the protected data (e.g. a micro-cache).
    """
    p_old = np.asarray(p_old, float); p_new = np.asarray(p_new, float)
    return float(np.mean((p_old * (np.log(p_old + 1e-300)
                                   - np.log(p_new + 1e-300))).sum(-1)))


def jsd_floor(pi_a: np.ndarray, pi_b: np.ndarray,
              dens_a: np.ndarray, dens_b: np.ndarray) -> float:
    """Closed-form irreducible floor for one predictor serving two tasks
    with conditionals pi_a, pi_b (n, K) and input densities dens_a,
    dens_b (n,) on a shared sample: sum (p_A+p_B) JSD_w(pi_A, pi_B),
    minimizer = density-weighted mixture.  Quadratic floor = its
    small-divergence limit."""
    pi_a = np.asarray(pi_a, float); pi_b = np.asarray(pi_b, float)
    da = np.asarray(dens_a, float)[:, None]; db = np.asarray(dens_b, float)[:, None]
    q = (da * pi_a + db * pi_b) / (da + db)
    kl = lambda p: (p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum(-1)
    return float(np.mean(da[:, 0] * kl(pi_a) + db[:, 0] * kl(pi_b)))


# ---------------------------------------------------------------------
#  The unified gate (Corollary "the sign gate is optimal monotone
#  control"): EVERY protection method in this framework is
#
#      v* = argmin ||v - u||^2   s.t.  <g_i, v> <= 0   (inequality rows)
#                                      <h_j, v>  = 0   (equality rows)
#
#  for some choice of constraint rows -- one function, one dual (NNLS;
#  equalities enter as mirrored +/- inequality pairs):
#
#      constraint set                          recovered method
#      -----------------------------------------------------------------
#      none                                    naive sharing
#      one sampled grad L_u                    A-GEM / threshold-free sign gate
#      all stored grad L_u (inequalities)      active-set gate
#      protected-subspace bases (equalities)   OGD / GPM projection
#      bases of tasks with overlap < s* only   igfa's s*-threshold gate
# ---------------------------------------------------------------------
def qp_gate(u: np.ndarray, G: Optional[np.ndarray] = None,
            G_eq: Optional[np.ndarray] = None) -> np.ndarray:
    """One gate for all protection methods.  u: desired update (d,);
    G: (d, m) inequality rows <g_i, v> <= 0;  G_eq: (d, k) equality rows.
    Returns the exact KKT solution v = u - C @ nnls(C, u)."""
    from scipy.optimize import nnls as _nnls
    cols = []
    if G is not None and G.size:
        cols.append(np.atleast_2d(G.T).T if G.ndim > 1 else G[:, None])
    if G_eq is not None and G_eq.size:
        E = G_eq if G_eq.ndim > 1 else G_eq[:, None]
        cols.append(E); cols.append(-E)                   # mirrored pair = equality
    if not cols:
        return u
    C = np.hstack(cols)
    lam, _ = _nnls(C, u)
    return u - C @ lam


def constraints_from_tasks(bases: List[np.ndarray], B_new: Optional[np.ndarray] = None,
                           sstar: Optional[float] = None) -> np.ndarray:
    """Equality rows for subspace protection.  With sstar and B_new given,
    keep only CONFLICTING tasks (overlap < s*): igfa's threshold gate.
    Without them, keep all: OGD/GPM."""
    keep = []
    for B in bases:
        if sstar is not None and B_new is not None:
            if principal_overlap(B, B_new) >= sstar:
                continue                                   # aligned -> shared
        keep.append(B)
    return np.hstack(keep) if keep else np.zeros((bases[0].shape[0], 0))
