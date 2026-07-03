# =====================================================================
#  kaggle_functional_gate_v2.py -- two upgrades to the functional gate,
#  in one self-contained script (LLM notebook, 2x T4, ~2 h total; parts
#  are independently toggleable).
#
#  P1  ADAM-AWARE sign gate (GPT-2, 5 seeds).  The v1 gate constrains the
#      RAW gradient, but Adam's preconditioner rescales it afterwards, so
#      the enforced condition is distorted -- the suspected cap on v1's
#      modest margins.  The Adam-aware test constrains the actual update:
#      with P = 1/(sqrt(v_hat)+eps), project g -> g - c*n with
#      c = <P.g, n>/<P.n, n> whenever <P.g, n> < 0, so the PRECONDITIONED
#      step is exactly orthogonal to grad L_u.  Arms: naive, protect-all,
#      func-sign (v1 reproduction), func-sign-adam.
#
#  P2  SCALE (pythia-410m, 3 seeds; set MODEL_P2 to pythia-1b if time
#      allows).  Arms: naive, protect-all, func-sign-adam.
#
#  Stream: news -> imdb -> wikitext -> tweets (real measured forgetting).
#  Micro-cache: 6 held-out chunks per past domain.  Worker threads log to
#  func_gate_v2_progress.log; results dumped incrementally.
# =====================================================================
import os, json, time
import numpy as np
import torch, torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, wait as fwait
from scipy import stats

RUN_P1, RUN_P2 = True, True
SEEDS_P1   = [0, 1, 2, 3, 4]
SEEDS_P2   = [0, 1, 2]
MODEL_P2   = "EleutherAI/pythia-410m"     # or "EleutherAI/pythia-1b"
STEPS      = 250
BS, SEQ    = 8, 128
RANK_LORA  = 8
RANK_PROT  = 8
CACHE_CH   = 6
DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
def tlog(m):
    with open("func_gate_v2_progress.log", "a") as f:
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


def build_model(name, device):
    from transformers import AutoModelForCausalLM
    # pythia checkpoints are fp16; newer transformers loaders can ignore the
    # torch_dtype kwarg, so force fp32 AFTER loading -- .float() converts every
    # parameter and buffer unconditionally (LoRA params are fp32).
    m = AutoModelForCausalLM.from_pretrained(name)
    m = m.float().to(device)
    bad = [n for n, p in list(m.named_parameters()) + list(m.named_buffers())
           if p.is_floating_point() and p.dtype != torch.float32]
    assert not bad, f"non-fp32 tensors remain after .float(): {bad[:5]} (running OLD cell?)"
    for p in m.parameters(): p.requires_grad_(False)
    loras = []
    if hasattr(m, "transformer"):                       # GPT-2
        blocks, get = m.transformer.h, lambda b: b.attn.c_attn
        din, dout = m.config.n_embd, 3 * m.config.n_embd
        def put(b, l): b.attn.c_attn = l
    else:                                               # pythia / gpt-neox
        blocks = m.gpt_neox.layers
        get = lambda b: b.attention.query_key_value
        din, dout = m.config.hidden_size, 3 * m.config.hidden_size
        def put(b, l): b.attention.query_key_value = l
    for b in blocks:
        l = LoRA(get(b), din, dout, RANK_LORA, device)
        put(b, l); loras.append(l)
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
        out.append((ch[:110], ch[110:130], ch[130:130 + CACHE_CH]))
    return out


def lm_loss(model, ids, device): return model(ids.to(device), labels=ids.to(device)).loss

@torch.no_grad()
def heldout(model, chunks, device):
    model.eval()
    v = float(np.mean([lm_loss(model, chunks[i:i+BS], device).item()
                       for i in range(0, len(chunks), BS)]))
    model.train(); return v


def run_stream(seed, method, device, doms, model_name):
    torch.manual_seed(seed)
    model, loras = build_model(model_name, device)
    params = [p for l in loras for p in (l.A, l.B)]
    opt = torch.optim.Adam(params, lr=3e-4)
    def gvec(): return torch.cat([p.grad.detach().flatten() for p in params])
    def setgrad(g):
        i = 0
        for p in params:
            n = p.numel(); p.grad = g[i:i+n].view_as(p).clone(); i += n
    def precond():
        """Adam's current elementwise preconditioner 1/(sqrt(v_hat)+eps)."""
        segs = []
        for p in params:
            st = opt.state.get(p, {})
            if "exp_avg_sq" in st:
                v = st["exp_avg_sq"]
                bc = 1 - 0.999 ** max(st.get("step", torch.tensor(1)).item()
                                      if torch.is_tensor(st.get("step", 1)) else st.get("step", 1), 1)
                segs.append((1.0 / (torch.sqrt(v / bc) + 1e-8)).flatten())
            else:
                segs.append(torch.ones(p.numel(), device=p.device))
        return torch.cat(segs)
    prot_bases = [[] for _ in loras]
    T = len(doms)
    held = np.zeros((T, T))
    for t, (tr, te, cache) in enumerate(doms):
        Qs = None
        if method == "protect-all" and t > 0:
            Qs = []
            for mi in range(len(loras)):
                cols = [prot_bases[mi][u] for u in range(t)]
                Qs.append(torch.linalg.qr(torch.cat(cols, 1))[0])
        for k in range(STEPS):
            i = torch.randint(0, len(tr) - BS, (1,)).item()
            loss = lm_loss(model, tr[i:i+BS], device)
            opt.zero_grad(); loss.backward()
            if method == "protect-all" and Qs is not None:
                with torch.no_grad():
                    for l, Q in zip(loras, Qs):
                        l.A.grad -= Q @ (Q.T @ l.A.grad)
            elif method in ("func-sign", "func-sign-adam") and t > 0:
                g = gvec()
                u = int(torch.randint(0, t, (1,)).item())
                cu = doms[u][2]
                j = torch.randint(0, max(len(cu) - 2, 1), (1,)).item()
                opt.zero_grad()
                lm_loss(model, cu[j:j+2], device).backward()
                nA = gvec()
                if method == "func-sign":                          # v1: raw-gradient test
                    dot = float(torch.dot(g, nA)); nrm2 = float(nA.norm() ** 2)
                    if nrm2 > 1e-12 and dot < 0:
                        g = g - dot / nrm2 * nA
                else:                                              # Adam-aware: test the UPDATE
                    P = precond()
                    dot = float(torch.dot(P * g, nA))
                    den = float(torch.dot(P * nA, nA))
                    if den > 1e-12 and dot < 0:
                        g = g - dot / den * nA                     # preconditioned step _|_ grad L_u
                setgrad(g)
            opt.step()
        for mi, l in enumerate(loras):
            X = l.x_sketch if l.x_sketch is not None else torch.zeros(8, loras[0].A.shape[0], device=device)
            U = torch.linalg.svd(X.T, full_matrices=False)[0][:, :RANK_PROT]
            prot_bases[mi].append(U)
        for i in range(t + 1):
            held[i, t] = heldout(model, doms[i][1], device)
        tlog(f"[{model_name.split('/')[-1]}] seed{seed} {method}: dom{t} done; held "
             + " ".join(f"{held[i, t]:.3f}" for i in range(t + 1)))
    final = held[:, T - 1]; diag = np.array([held[i, i] for i in range(T)])
    del model; torch.cuda.empty_cache()
    return dict(final=float(final.mean()),
                forget=float(np.mean(final[:-1] - diag[:-1])))


def run_part(tag, model_name, methods, seeds, doms, results):
    for method in methods:
        per = {}
        with ThreadPoolExecutor(len(DEVICES)) as ex:
            futs = {ex.submit(run_stream, s, method, DEVICES[i % len(DEVICES)],
                              doms, model_name): s for i, s in enumerate(seeds)}
            pending = set(futs)
            while pending:
                done, pending = fwait(pending, timeout=120)
                for f in done: per[futs[f]] = f.result()
                log(f"{tag}/{method}: {len(seeds)-len(pending)}/{len(seeds)} seeds done")
        results[f"{tag}/{method}"] = {str(s): per[s] for s in seeds}
        fm = [per[s]["final"] for s in seeds]; gm = [per[s]["forget"] for s in seeds]
        log(f"== {tag}/{method:15s}: avg final {np.mean(fm):.3f}+/-{np.std(fm):.3f}   "
            f"forgetting {np.mean(gm):+.3f}+/-{np.std(gm):.3f}")
        with open("func_gate_v2_results.json", "w") as f: json.dump(results, f, indent=1)
    # paired tests vs the two key baselines
    for a, b in [("func-sign-adam", "func-sign"), ("func-sign-adam", "protect-all"),
                 ("func-sign-adam", "naive"), ("func-sign", "protect-all")]:
        ka, kb = f"{tag}/{a}", f"{tag}/{b}"
        if ka in results and kb in results:
            for metric in ["final", "forget"]:
                xa = [results[ka][str(s)][metric] for s in seeds]
                xb = [results[kb][str(s)][metric] for s in seeds]
                p = stats.ttest_rel(xb, xa).pvalue
                print(f"  {tag}: {a} vs {b} [{metric}]: {np.mean(xa):.4f} vs {np.mean(xb):.4f} "
                      f"(diff {np.mean(xb)-np.mean(xa):+.4f}, p={p:.4f}, "
                      f"better {sum(y > x for x, y in zip(xa, xb))}/{len(seeds)})")


if __name__ == "__main__":
    log(f"devices: {DEVICES}")
    results = {}
    if RUN_P1:
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        doms = get_domains(tok)
        log("P1 (gpt2): Adam-aware sign gate, 5 seeds")
        run_part("P1-gpt2", "gpt2", ["naive", "protect-all", "func-sign", "func-sign-adam"],
                 SEEDS_P1, doms, results)
    if RUN_P2:
        from transformers import AutoTokenizer
        tok2 = AutoTokenizer.from_pretrained(MODEL_P2)
        doms2 = get_domains(tok2)
        log(f"P2 ({MODEL_P2}): scale check, 3 seeds")
        run_part("P2-" + MODEL_P2.split('/')[-1], MODEL_P2,
                 ["naive", "protect-all", "func-sign-adam"], SEEDS_P2, doms2, results)
    log("ALL DONE -> func_gate_v2_results.json")
