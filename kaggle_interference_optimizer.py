# =====================================================================
#  kaggle_interference_optimizer.py -- Q12 at scale: is the practical
#  successor to the igfa gate an OPTIMIZER?
#
#  InterferenceSGD: plain SGD on  L_B + (mu/2)(theta-theta_ref)^T S (theta-theta_ref),
#  where S is a Frequent-Directions sketch of task A's LoRA-gradient second
#  moment (memory O(ell * P_lora)).  Benchmarked against TUNED learning-rate
#  schedules (constant / cosine / linear x three peak LRs, with early-stop
#  checkpoints) at MATCHED step budget on the same model and stream.
#
#  Model/stream: GPT-2 + manual LoRA (no peft), domain A = AG-News world,
#  domain B = AG-News sci/tech (the most conflicting pair by probe cosine).
#  Metrics: held-out loss on A (forgetting) and on B (adaptation) ->
#  retention-plasticity frontiers; paired comparison at matched forgetting.
#
#  Kaggle 2x T4: seeds split across GPUs; worker threads log to
#  interference_opt_progress.log (never to the notebook stream); results
#  dumped incrementally to interference_opt_results.json.  ~45 min.
# =====================================================================
import os, json, time
import numpy as np
import torch, torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, wait as fwait
from scipy import stats

SEEDS      = [0, 1, 2]
STEPS      = 300          # per configuration (matched budget)
BS, SEQ    = 8, 128
RANK_LORA  = 8
ELL        = 2 * RANK_LORA        # FD sketch rows (the ell>=2r sizing rule)
MU_GRID    = list(np.geomspace(0.03, 100.0, 8))
SCHEDULES  = [(kind, peak) for kind in ["constant", "cosine", "linear"]
              for peak in [1e-4, 3e-4, 1e-3]]
CKPT_EVERY = 30           # early-stop checkpoints along each schedule
DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
def tlog(m):
    with open("interference_opt_progress.log", "a") as f:
        f.write(f"[{time.time()-T0:7.1f}s] {m}\n")


# ---------------------- InterferenceSGD (the artifact) ----------------------
class InterferenceSGD(torch.optim.Optimizer):
    """SGD on L_B plus an interference trust-region penalty in a sketched metric.

    For each parameter p with reference p_ref and sketch factors (V, s2) --
    rows V (ell x numel) and squared singular values s2 (ell,) of a
    Frequent-Directions sketch of the old task's gradient second moment --
    the update is  g + mu * V^T diag(s2) V (p - p_ref),  cost O(ell * numel).
    mu -> 0 recovers SGD; mu -> inf recovers null-space projection (the gate).
    """
    def __init__(self, params, lr, mu=0.0):
        super().__init__(params, dict(lr=lr, mu=mu))
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                g = p.grad
                st = self.state.get(p, {})
                if group["mu"] > 0 and "V" in st:
                    dp = (p - st["ref"]).flatten()
                    pen = st["V"].T @ (st["s2"] * (st["V"] @ dp))
                    g = g + group["mu"] * pen.view_as(p)
                p.add_(g, alpha=-group["lr"])
    def set_metric(self, p, V, s2, ref):
        self.state[p] = dict(V=V, s2=s2, ref=ref.detach().clone())


def fd_sketch(G, ell):
    """Frequent Directions of gradient rows G (m x P) -> (V rows ell x P, s2)."""
    sk = np.zeros((ell, G.shape[1]), dtype=np.float32)
    for i in range(0, len(G), ell):
        B = np.vstack([sk, G[i:i + ell]])
        _, s, Vt = np.linalg.svd(B, full_matrices=False)
        delta = s[ell - 1] ** 2 if len(s) >= ell else 0.0
        sk = (np.sqrt(np.maximum(s[:ell] ** 2 - delta, 0))[:, None] * Vt[:ell]).astype(np.float32)
    _, s, Vt = np.linalg.svd(sk, full_matrices=False)
    return Vt[:ell].astype(np.float32), (s[:ell] ** 2).astype(np.float32)


# ------------------------- model + data (as probe-gate) ---------------------
class LoRA(nn.Module):
    def __init__(self, base, d_in, d_out, r, device):
        super().__init__(); self.base, self.s = base, 2.0 / r
        self.A = nn.Parameter(torch.zeros(d_in, r, device=device))
        self.B = nn.Parameter(torch.zeros(r, d_out, device=device))
        nn.init.normal_(self.A, std=0.02)
    def forward(self, x): return self.base(x) + (x @ self.A) @ self.B * self.s

def build_model(device):
    from transformers import GPT2LMHeadModel
    m = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    for p in m.parameters(): p.requires_grad_(False)
    loras = []
    for blk in m.transformer.h:
        l = LoRA(blk.attn.c_attn, 768, 2304, RANK_LORA, device)
        blk.attn.c_attn = l; loras.append(l)
    return m, loras

def get_domains(tok):
    from datasets import load_dataset
    ds = load_dataset("ag_news", split="train")
    out = []
    for c in [0, 3]:                                        # world vs sci/tech (most conflicting)
        texts = [x["text"] for x in ds if x["label"] == c][:1500]
        enc = tok("\n\n".join(texts), return_tensors="pt").input_ids[0]
        ch = enc[: (len(enc) // SEQ) * SEQ].view(-1, SEQ)
        out.append((ch[:120], ch[120:150]))
    return out

def lm_loss(model, ids, device): return model(ids.to(device), labels=ids.to(device)).loss

@torch.no_grad()
def heldout(model, chunks, device):
    model.eval()
    v = float(np.mean([lm_loss(model, chunks[i:i+BS], device).item()
                       for i in range(0, len(chunks), BS)]))
    model.train(); return v


def run_seed(seed, device, doms):
    torch.manual_seed(seed)
    (trA, teA), (trB, teB) = doms
    model, loras = build_model(device)
    params = [p for l in loras for p in (l.A, l.B)]
    # ---- train domain A ----
    optA = torch.optim.Adam(params, lr=3e-4)
    for k in range(STEPS):
        i = torch.randint(0, len(trA) - BS, (1,)).item()
        loss = lm_loss(model, trA[i:i+BS], device)
        optA.zero_grad(); loss.backward(); optA.step()
    LA0 = heldout(model, teA, device)
    theta_ref = [p.detach().clone() for p in params]
    # ---- collect task-A LoRA gradients and sketch the metric per-parameter ----
    grads = {p: [] for p in params}
    for k in range(48):
        i = torch.randint(0, len(trA) - BS, (1,)).item()
        loss = lm_loss(model, trA[i:i+BS], device)
        model.zero_grad(); loss.backward()
        for p in params: grads[p].append(p.grad.detach().flatten().cpu().numpy())
    metrics = {p: fd_sketch(np.stack(grads[p]), ELL) for p in params}
    snap = [p.detach().clone() for p in params]
    def restore():
        with torch.no_grad():
            for p, s in zip(params, snap): p.copy_(s)
    frontier = {"schedule": [], "interference": []}
    # ---- (a) tuned LR schedules with early-stop checkpoints ----
    for kind, peak in SCHEDULES:
        restore()
        opt = torch.optim.SGD(params, lr=peak)
        for k in range(STEPS):
            lr = peak * (1.0 if kind == "constant" else
                         0.5 * (1 + np.cos(np.pi * k / STEPS)) if kind == "cosine" else
                         (1 - k / STEPS))
            for gset in opt.param_groups: gset["lr"] = lr
            i = torch.randint(0, len(trB) - BS, (1,)).item()
            loss = lm_loss(model, trB[i:i+BS], device)
            opt.zero_grad(); loss.backward(); opt.step()
            if (k + 1) % CKPT_EVERY == 0:
                frontier["schedule"].append((heldout(model, teA, device) - LA0,
                                             heldout(model, teB, device)))
        tlog(f"seed{seed} schedule {kind}/{peak:g} done @{device}")
    # ---- (b) InterferenceSGD over the mu grid, fixed lr ----
    for mu in MU_GRID:
        restore()
        opt = InterferenceSGD(params, lr=3e-4, mu=mu)
        for p, ref in zip(params, theta_ref):
            V, s2 = metrics[p]
            opt.set_metric(p, torch.tensor(V, device=device),
                           torch.tensor(s2, device=device), ref)
        for k in range(STEPS):
            i = torch.randint(0, len(trB) - BS, (1,)).item()
            loss = lm_loss(model, trB[i:i+BS], device)
            opt.zero_grad(); loss.backward(); opt.step()
        frontier["interference"].append((heldout(model, teA, device) - LA0,
                                         heldout(model, teB, device)))
        tlog(f"seed{seed} InterferenceSGD mu={mu:g} done @{device}")
    del model; torch.cuda.empty_cache()
    return frontier


if __name__ == "__main__":
    log(f"devices: {DEVICES}")
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    doms = get_domains(tok)
    log("domains ready (world -> sci/tech)")
    per = {}
    with ThreadPoolExecutor(len(DEVICES)) as ex:
        futs = {ex.submit(run_seed, s, DEVICES[i % len(DEVICES)], doms): s
                for i, s in enumerate(SEEDS)}
        pending = set(futs)
        while pending:
            done, pending = fwait(pending, timeout=120)
            for f in done: per[futs[f]] = f.result()
            log(f"{len(SEEDS)-len(pending)}/{len(SEEDS)} seeds done "
                f"(details: interference_opt_progress.log)")
            with open("interference_opt_results.json", "w") as f:
                json.dump({str(k): v for k, v in per.items()}, f, indent=1)
    # ---- frontier comparison at matched forgetting ----
    print("\n===== new-domain loss at matched forgetting (mean over seeds) =====")
    gains = {f: [] for f in [0.05, 0.15, 0.4]}
    for s in SEEDS:
        fs = np.array(sorted(per[s]["schedule"])); fi = np.array(sorted(per[s]["interference"]))
        for f in gains:
            ls = np.interp(f, fs[:, 0], fs[:, 1]); li = np.interp(f, fi[:, 0], fi[:, 1])
            gains[f].append((ls, li))
    for f, v in gains.items():
        ls = np.mean([x[0] for x in v]); li = np.mean([x[1] for x in v])
        p = stats.ttest_rel([x[0] for x in v], [x[1] for x in v]).pvalue
        print(f"  forgetting {f:.2f}: tuned-LR {ls:.3f}   InterferenceSGD {li:.3f}   "
              f"({100*(ls-li)/ls:+.1f}%, p={p:.3f})")
    log("ALL DONE -> interference_opt_results.json")
