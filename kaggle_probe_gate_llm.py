# =====================================================================
#  kaggle_probe_gate_llm.py -- the open language-gate test, on real text.
#
#  Question (paper Sec. S6): can a PROBE-based similarity signal -- a few
#  fine-tuning steps on the incoming domain, compared to stored per-task
#  update directions by signed cosine -- recover transfer on SIMILAR text
#  tasks where the input-overlap gate collapses to share-all?
#
#  Setup: GPT-2 (124M) + manual LoRA (no peft -> no torchao issues) on all
#  attention projections, trained sequentially on FOUR SIMILAR text domains
#  (AG-News topic slices: same register/syntax, related content).
#  Methods: naive / protect-all (OGD) / input-overlap gate / probe gate.
#  Metrics: per-domain held-out loss after each stage -> forgetting and
#  final average loss; the gate's win condition is lower final average
#  loss than protect-all at comparable forgetting.
#
#  Kaggle: 2x T4. Seeds are split across GPUs; progress goes to
#  probe_gate_progress.log (worker threads never print -- ipykernel
#  deadlocks on multi-threaded prints). Results: probe_gate_results.json.
# =====================================================================
import os, json, time, threading
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, wait as fwait
from tqdm.auto import tqdm
from scipy import stats

SEEDS      = [0, 1, 2]
STEPS      = 250          # LoRA steps per domain
PROBE_STEPS= 30           # probe fine-tuning steps for the similarity signal
LR, BS, SEQ = 2e-4, 8, 128
RANK_LORA  = 8            # LoRA rank
RANK_PROT  = 8            # protected input-subspace rank per task/module
SSTAR      = 0.25         # probe-cosine share threshold (printed values allow re-calibration)
METHODS    = ["naive", "protect-all", "input-gate", "probe-gate"]
DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
T0 = time.time()
def log(msg): print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)
def tlog(msg):
    with open("probe_gate_progress.log", "a") as f:
        f.write(f"[{time.time()-T0:7.1f}s] {msg}\n")

# ----------------------------- data -----------------------------------
def get_domains(tok):
    """Four SIMILAR text domains: AG-News topic slices (world/sports/business/scitech)."""
    from datasets import load_dataset
    ds = load_dataset("ag_news", split="train")
    names = ["world", "sports", "business", "sci/tech"]
    doms = []
    for c in range(4):
        texts = [x["text"] for x in ds if x["label"] == c][:1500]
        enc = tok("\n\n".join(texts), return_tensors="pt").input_ids[0]
        chunks = enc[: (len(enc) // SEQ) * SEQ].view(-1, SEQ)
        doms.append((names[c], chunks[:120], chunks[120:150]))   # train, held-out
    return doms

# ------------------------- manual LoRA on GPT-2 ------------------------
class LoRA(nn.Module):
    """Additive LoRA on a frozen HF Conv1D/Linear: y = base(x) + (x @ A) @ B * s."""
    def __init__(self, base, d_in, d_out, r, device):
        super().__init__()
        self.base, self.s = base, 2.0 / r
        self.A = nn.Parameter(torch.zeros(d_in, r, device=device))
        self.B = nn.Parameter(torch.zeros(r, d_out, device=device))
        nn.init.normal_(self.A, std=0.02)
        self.x_sketch = None                                   # input-activation sketch
    def forward(self, x):
        if self.training and torch.rand(()) < 0.1:             # subsample activations
            with torch.no_grad():
                xs = x.detach().reshape(-1, x.shape[-1])
                idx = torch.randint(0, len(xs), (min(64, len(xs)),), device=x.device)
                self.x_sketch = xs[idx].float()
        return self.base(x) + (x @ self.A) @ self.B * self.s

def build_model(device):
    from transformers import GPT2LMHeadModel
    m = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    for p in m.parameters(): p.requires_grad_(False)
    loras = []
    for blk in m.transformer.h:
        base = blk.attn.c_attn                                 # Conv1D: weight (768, 2304)
        l = LoRA(base, 768, 2304, RANK_LORA, device)
        blk.attn.c_attn = l
        loras.append(l)
    return m, loras

def lm_loss(model, batch, device):
    ids = batch.to(device)
    out = model(ids, labels=ids)
    return out.loss

@torch.no_grad()
def eval_domain(model, chunks, device):
    model.eval(); tot = 0.0
    for i in range(0, len(chunks), BS):
        tot += lm_loss(model, chunks[i:i + BS], device).item() * len(chunks[i:i + BS])
    model.train()
    return tot / len(chunks)

def flat_delta(loras, snap):
    v = torch.cat([torch.cat([(l.A - a).flatten(), (l.B - b).flatten()])
                   for l, (a, b) in zip(loras, snap)])
    return v / (v.norm() + 1e-12)

def snapshot(loras): return [(l.A.detach().clone(), l.B.detach().clone()) for l in loras]
def restore(loras, snap):
    with torch.no_grad():
        for l, (a, b) in zip(loras, snap): l.A.copy_(a); l.B.copy_(b)

# --------------------------- one sequential run -------------------------
def run_stream(seed, method, device, doms_ids):
    torch.manual_seed(seed)
    from transformers import GPT2TokenizerFast
    model, loras = build_model(device)
    doms = doms_ids
    T = len(doms)
    prot_bases = [[] for _ in loras]                           # per-module per-task input bases
    task_deltas, in_overlaps = [], []
    held = np.zeros((T, T))                                    # held[i, t] = loss on dom i after stage t
    opt = torch.optim.Adam([p for l in loras for p in (l.A, l.B)], lr=LR)

    def train_steps(chunks, n, project_tasks):
        Qs = None
        if project_tasks:                                      # protect: per-module joint basis
            Qs = []
            for mi, l in enumerate(loras):
                cols = [prot_bases[mi][t] for t in project_tasks]
                Qs.append(torch.linalg.qr(torch.cat(cols, 1))[0])
        for k in range(n):
            i = torch.randint(0, len(chunks) - BS, (1,)).item()
            loss = lm_loss(model, chunks[i:i + BS], device)
            opt.zero_grad(); loss.backward()
            if Qs is not None:
                with torch.no_grad():
                    for l, Q in zip(loras, Qs):
                        l.A.grad -= Q @ (Q.T @ l.A.grad)       # input-side null-space routing
            opt.step()

    for t, (name, tr, te) in enumerate(doms):
        # 1) similarity signals against every stored task
        share_with = []
        if t > 0:
            snap = snapshot(loras)
            train_steps(tr, PROBE_STEPS, None)                 # probe: a few unprotected steps
            dprobe = flat_delta(loras, snap)
            restore(loras, snap)
            for u in range(t):
                s_hat = float(dprobe @ task_deltas[u])
                ov = float(np.mean([torch.linalg.norm(prot_bases[mi][u].T @ prot_bases[mi][u]).item()
                                    for mi in range(len(loras))]))  # placeholder self-norm
                in_overlaps.append((t, u, s_hat))
                tlog(f"seed{seed} {method}: dom{t} vs dom{u}: probe cos {s_hat:+.3f}")
                if method == "probe-gate" and s_hat > SSTAR: share_with.append(u)
                if method == "input-gate": share_with.append(u)   # overlap ~1 on same-register text
        protect = ([] if method == "naive" else
                   [u for u in range(t) if u not in share_with])
        # 2) train the domain with the chosen protection
        snap = snapshot(loras)
        train_steps(tr, STEPS, protect if protect else None)
        task_deltas.append(flat_delta(loras, snap))
        # 3) store per-module input bases for this task
        for mi, l in enumerate(loras):
            X = l.x_sketch if l.x_sketch is not None else torch.zeros(8, 768, device=device)
            U = torch.linalg.svd(X.T, full_matrices=False)[0][:, :RANK_PROT]
            prot_bases[mi].append(U)
        # 4) evaluate all seen domains
        for i in range(t + 1):
            held[i, t] = eval_domain(model, doms[i][2], device)
        tlog(f"seed{seed} {method}: dom{t} ({name}) done; held-out " +
             " ".join(f"{held[i, t]:.3f}" for i in range(t + 1)))
    final = held[:, T - 1]
    diag = np.array([held[i, i] for i in range(T)])
    forgetting = float(np.mean(final[:-1] - diag[:-1]))
    del model; torch.cuda.empty_cache()
    return float(final.mean()), forgetting

# ================================ MAIN =================================
if __name__ == "__main__":
    log(f"devices: {DEVICES}")
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    doms = get_domains(tok)
    log("domains ready: " + ", ".join(d[0] for d in doms))
    results = {}
    for method in METHODS:
        per = {}
        with ThreadPoolExecutor(len(DEVICES)) as ex:
            futs = {ex.submit(run_stream, s, method, DEVICES[i % len(DEVICES)], doms): s
                    for i, s in enumerate(SEEDS)}
            pending = set(futs)
            while pending:
                done, pending = fwait(pending, timeout=120)
                for f in done: per[futs[f]] = f.result()
                log(f"{method}: {len(SEEDS)-len(pending)}/{len(SEEDS)} seeds done "
                    f"(details: probe_gate_progress.log)")
        vals = [per[s] for s in SEEDS]
        results[method] = dict(final=[v[0] for v in vals], forget=[v[1] for v in vals])
        f_m = np.mean([v[0] for v in vals]); f_s = np.std([v[0] for v in vals])
        g_m = np.mean([v[1] for v in vals]); g_s = np.std([v[1] for v in vals])
        log(f"== {method:12s}: final avg loss {f_m:.3f}+/-{f_s:.3f}   "
            f"forgetting {g_m:+.3f}+/-{g_s:.3f}")
        with open("probe_gate_results.json", "w") as f: json.dump(results, f, indent=1)
    if all(m in results for m in ["probe-gate", "protect-all"]):
        p = stats.ttest_rel(results["probe-gate"]["final"],
                            results["protect-all"]["final"]).pvalue
        log(f"paired t-test final loss probe-gate vs protect-all: p={p:.4f}")
    log("ALL DONE -> probe_gate_results.json (probe cosines in probe_gate_progress.log)")
