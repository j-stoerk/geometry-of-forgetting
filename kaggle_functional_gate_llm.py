# =====================================================================
#  kaggle_functional_gate_llm.py -- the language extension of the gate.
#
#  The input-overlap gate fails on text (paper Sec. S6) because it reads
#  input geometry, which is uninformative when all domains share syntax.
#  The function-space formalism (Sec. si:beyond) says the correct signal
#  is the INTERFERENCE RATE  <grad L_A, g_B>  -- targets included, no
#  threshold: keep every component of the new-domain update that does
#  not increase a stored domain's loss, remove only those that do
#  (monotone retention + full transfer in continuous time).
#
#  Stream: GPT-2 + manual LoRA on news -> imdb -> wikitext -> tweets
#  (the Table-S3 regime with real measured forgetting).  Methods:
#    naive        : plain Adam
#    protect-all  : null-space routing off per-past-domain activation bases
#    input-gate   : share/protect by activation-subspace overlap (the
#                   signal known to collapse on language)
#    func-sign    : the functional gate; grad-hat L_u from a MICRO-CACHE
#                   of 6 held-out chunks per past domain (~4.6k tokens),
#                   one sampled past domain per step (A-GEM-style
#                   stochastic constraint, derived here from the exact
#                   path integral; buffer-light, and the anchor/dream
#                   variants of Secs. si:cheapgn/si:memory can replace
#                   the cache entirely).
#  Also logs the measured interference-rate map between all domain pairs
#  -- the language similarity signal itself.
#
#  Kaggle 2x T4, ~45 min, SEEDS=[0,1,2].  Worker threads log to
#  func_gate_progress.log; results dumped incrementally.
# =====================================================================
import os, json, time
import numpy as np
import torch, torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, wait as fwait

SEEDS      = [0, 1, 2]
STEPS      = 250
BS, SEQ    = 8, 128
RANK_LORA  = 8
RANK_PROT  = 8
CACHE_CH   = 6                 # micro-cache chunks per past domain
SSTAR_IN   = 0.6               # input-gate share threshold (overlap)
DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
def tlog(m):
    with open("func_gate_progress.log", "a") as f:
        f.write(f"[{time.time()-T0:7.1f}s] {m}\n")


class LoRA(nn.Module):
    def __init__(self, base, d_in, d_out, r, device):
        super().__init__(); self.base, self.s = base, 2.0 / r
        self.A = nn.Parameter(torch.zeros(d_in, r, device=device))
        self.B = nn.Parameter(torch.zeros(r, d_out, device=device))
        nn.init.normal_(self.A, std=0.02)
        self.x_sketch = None
    def forward(self, x):
        if self.training and torch.rand(()) < 0.1:
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
        l = LoRA(blk.attn.c_attn, 768, 2304, RANK_LORA, device)
        blk.attn.c_attn = l; loras.append(l)
    return m, loras

def get_domains(tok):
    from datasets import load_dataset
    texts = []
    news = load_dataset("ag_news", split="train")
    texts.append("\n\n".join([x["text"] for x in news if x["label"] == 0][:1500]))
    imdb = load_dataset("imdb", split="train")
    texts.append("\n\n".join([x["text"] for x in imdb][:600]))
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts.append("\n\n".join([x["text"] for x in wiki if len(x["text"]) > 200][:1200]))
    tw = load_dataset("tweet_eval", "sentiment", split="train")
    texts.append("\n\n".join([x["text"] for x in tw][:4000]))
    out = []
    for t in texts:
        enc = tok(t, return_tensors="pt").input_ids[0]
        ch = enc[: (len(enc) // SEQ) * SEQ].view(-1, SEQ)
        out.append((ch[:110], ch[110:130], ch[130:130 + CACHE_CH]))   # train, test, cache
    return out

def lm_loss(model, ids, device): return model(ids.to(device), labels=ids.to(device)).loss

@torch.no_grad()
def heldout(model, chunks, device):
    model.eval()
    v = float(np.mean([lm_loss(model, chunks[i:i+BS], device).item()
                       for i in range(0, len(chunks), BS)]))
    model.train(); return v


def run_stream(seed, method, device, doms):
    torch.manual_seed(seed)
    model, loras = build_model(device)
    params = [p for l in loras for p in (l.A, l.B)]
    opt = torch.optim.Adam(params, lr=3e-4)
    def gvec(): return torch.cat([p.grad.detach().flatten() for p in params])
    def setgrad(g):
        i = 0
        for p in params:
            n = p.numel(); p.grad = g[i:i+n].view_as(p).clone(); i += n
    prot_bases = [[] for _ in loras]
    T = len(doms)
    held = np.zeros((T, T)); align_log = []
    for t, (tr, te, cache) in enumerate(doms):
        # ---- protection set for structural baselines ----
        Qs, share = None, list(range(t))
        if method in ("protect-all", "input-gate") and t > 0:
            protect = []
            for u in range(t):
                if method == "input-gate":
                    ov = np.mean([float(torch.linalg.norm(
                        prot_bases[mi][u].T @ prot_bases[mi][t - 1]).item() ** 2 / RANK_PROT)
                        for mi in range(len(loras))]) if t >= 1 else 0.0
                    if ov >= SSTAR_IN: continue           # 'similar' -> share (the failure mode)
                protect.append(u)
            if protect:
                Qs = []
                for mi in range(len(loras)):
                    cols = [prot_bases[mi][u] for u in protect]
                    Qs.append(torch.linalg.qr(torch.cat(cols, 1))[0])
        for k in range(STEPS):
            i = torch.randint(0, len(tr) - BS, (1,)).item()
            loss = lm_loss(model, tr[i:i+BS], device)
            opt.zero_grad(); loss.backward()
            if method in ("protect-all", "input-gate") and Qs is not None:
                with torch.no_grad():
                    for l, Q in zip(loras, Qs):
                        l.A.grad -= Q @ (Q.T @ l.A.grad)
            elif method == "func-sign" and t > 0:
                g = gvec()
                u = int(torch.randint(0, t, (1,)).item())  # sampled past domain
                cu = doms[u][2]
                j = torch.randint(0, max(len(cu) - 2, 1), (1,)).item()
                opt.zero_grad()
                lm_loss(model, cu[j:j+2], device).backward()
                nA = gvec()
                dot = float(torch.dot(g, nA)); nrm2 = float(nA.norm() ** 2)
                if k % 50 == 0:
                    align_log.append((t, u, dot / (np.sqrt(nrm2) * float(g.norm()) + 1e-12)))
                if nrm2 > 1e-12 and dot < 0:               # -g would increase L_u: project
                    g = g - dot / nrm2 * nA
                setgrad(g)
            opt.step()
        # per-module activation bases (for the structural baselines' next tasks)
        for mi, l in enumerate(loras):
            X = l.x_sketch if l.x_sketch is not None else torch.zeros(8, 768, device=device)
            U = torch.linalg.svd(X.T, full_matrices=False)[0][:, :RANK_PROT]
            prot_bases[mi].append(U)
        for i in range(t + 1):
            held[i, t] = heldout(model, doms[i][1], device)
        tlog(f"seed{seed} {method}: dom{t} done; held " +
             " ".join(f"{held[i, t]:.3f}" for i in range(t + 1)))
    final = held[:, T - 1]; diag = np.array([held[i, i] for i in range(T)])
    fgt = float(np.mean(final[:-1] - diag[:-1]))
    for (t, u, c) in align_log[:24]:
        tlog(f"seed{seed} func-sign alignment dom{t} vs dom{u}: {c:+.3f}")
    del model; torch.cuda.empty_cache()
    return dict(final=float(final.mean()), forget=fgt,
                align=[(int(a), int(b), float(c)) for a, b, c in align_log])


if __name__ == "__main__":
    log(f"devices: {DEVICES}")
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    doms = get_domains(tok)
    log("domains ready: news -> imdb -> wikitext -> tweets  (micro-cache "
        f"{CACHE_CH} chunks = {CACHE_CH*SEQ} tokens per past domain for func-sign)")
    results = {}
    for method in ["naive", "protect-all", "input-gate", "func-sign"]:
        per = {}
        with ThreadPoolExecutor(len(DEVICES)) as ex:
            futs = {ex.submit(run_stream, s, method, DEVICES[i % len(DEVICES)], doms): s
                    for i, s in enumerate(SEEDS)}
            pending = set(futs)
            while pending:
                done, pending = fwait(pending, timeout=120)
                for f in done: per[futs[f]] = f.result()
                log(f"{method}: {len(SEEDS)-len(pending)}/{len(SEEDS)} seeds done "
                    f"(details: func_gate_progress.log)")
        results[method] = {str(s): per[s] for s in SEEDS}
        fm = np.mean([per[s]["final"] for s in SEEDS]); fs = np.std([per[s]["final"] for s in SEEDS])
        gm = np.mean([per[s]["forget"] for s in SEEDS]); gs = np.std([per[s]["forget"] for s in SEEDS])
        log(f"== {method:12s}: avg final loss {fm:.3f}+/-{fs:.3f}   forgetting {gm:+.3f}+/-{gs:.3f}")
        with open("func_gate_results.json", "w") as f: json.dump(results, f, indent=1)
    # interference-rate map (the language similarity signal, measured)
    amap = {}
    for s in SEEDS:
        for (t, u, c) in results["func-sign"][str(s)]["align"]:
            amap.setdefault((t, u), []).append(c)
    print("\n===== measured interference-rate map (mean cos<g_B, grad L_u>) =====")
    for (t, u), v in sorted(amap.items()):
        print(f"  training dom{t}, stored dom{u}: {np.mean(v):+.3f} "
              f"({'mostly transfer' if np.mean(v) > 0.02 else 'mostly conflict' if np.mean(v) < -0.02 else 'neutral'})")
    log("ALL DONE -> func_gate_results.json")
