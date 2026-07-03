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

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, wait as fwait

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

RUN_P1 = False
RUN_P2 = True

SEEDS_P1 = [0, 1, 2, 3, 4]
SEEDS_P2 = [0, 1, 2]
MODEL_P2 = "EleutherAI/pythia-410m"   # or "EleutherAI/pythia-1b"
STEPS = 250
BS, SEQ = 8, 128
RANK_LORA = 8
RANK_PROT = 8
CACHE_CH = 6
USE_THREADS = False  # safer in notebooks / multi-GPU Kaggle
LR_P1 = 3e-4
LR_P2 = 1e-5
ADAM_EPS = 1e-6
MAX_GRAD_NORM = 0.5

DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)



def tlog(m):
    with open("func_gate_v2_progress.log", "a") as f:
        f.write(f"[{time.time()-T0:7.1f}s] {m}\n")


SCRIPT_VERSION = "v2.9-randint-fixed"


class LoRA(nn.Module):
    """Dtype-agnostic LoRA: adapter params stay fp32 for optimizer/gate math,
    while forward passes cast around whatever dtype the base model uses."""

    def __init__(self, base, d_in, d_out, r, device):
        super().__init__()
        self.base = base
        self.s = 2.0 / r
        self.A = nn.Parameter(torch.zeros(d_in, r, device=device))
        self.B = nn.Parameter(torch.empty(r, d_out, device=device))
        nn.init.normal_(self.B, std=1e-6)
        self.x_sketch = None

    def forward(self, x):
        if self.training and torch.rand(()) < 0.1:
            with torch.no_grad():
                xs = x.detach().reshape(-1, x.shape[-1])
                idx = torch.randint(0, len(xs), (min(64, len(xs)),), device=x.device)
                self.x_sketch = xs[idx].float()
        base = self.base(x)
        lora = (x.to(self.A.dtype) @ self.A) @ self.B * self.s
        return base + lora.to(base.dtype)



def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def check_named_tensors_finite(named_tensors, where):
    for name, tensor in named_tensors:
        if tensor is None:
            continue
        if not torch.isfinite(tensor).all():
            finite = tensor[torch.isfinite(tensor)]
            maxabs = float(finite.abs().max()) if finite.numel() > 0 else float('nan')
            raise ValueError(f"Non-finite tensor at {where}: {name}, shape={tuple(tensor.shape)}, finite_maxabs={maxabs}")



def build_model(name, device):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(name)
    m = m.float().to(device)
    if getattr(m.config, "use_cache", None) is not None:
        m.config.use_cache = False

    bad = [
        n for n, p in list(m.named_parameters()) + list(m.named_buffers())
        if p.is_floating_point() and p.dtype != torch.float32
    ]
    if bad:
        tlog(
            f"NOTE: {len(bad)} tensors not fp32 after .float() "
            f"(e.g. {bad[:3]}); LoRA forward is dtype-safe, continuing."
        )

    for p in m.parameters():
        p.requires_grad_(False)

    loras = []
    if hasattr(m, "transformer"):  # GPT-2
        blocks = m.transformer.h
        get = lambda b: b.attn.c_attn
        din, dout = m.config.n_embd, 3 * m.config.n_embd

        def put(b, l):
            b.attn.c_attn = l
    else:  # pythia / gpt-neox
        blocks = m.gpt_neox.layers
        get = lambda b: b.attention.query_key_value
        din, dout = m.config.hidden_size, 3 * m.config.hidden_size

        def put(b, l):
            b.attention.query_key_value = l

    for b in blocks:
        l = LoRA(get(b), din, dout, RANK_LORA, device)
        put(b, l)
        loras.append(l)

    return m, loras



def get_domains(tok):
    from datasets import load_dataset

    texts = []
    news = load_dataset("ag_news", split="train")
    texts.append(("news", "\n\n".join([x["text"] for x in news if x["label"] == 0][:1500])))

    imdb = load_dataset("imdb", split="train")
    texts.append(("imdb", "\n\n".join([x["text"] for x in imdb][:600])))

    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts.append(("wiki", "\n\n".join([x["text"] for x in wiki if len(x["text"]) > 200][:1200])))

    tw = load_dataset("tweet_eval", "sentiment", split="train")
    texts.append(("tweets", "\n\n".join([x["text"] for x in tw][:4000])))

    out = []
    need = 110 + 20 + CACHE_CH
    for name, t in texts:
        enc = tok(t, return_tensors="pt").input_ids[0]
        ch = enc[: (len(enc) // SEQ) * SEQ].view(-1, SEQ)
        log(f"domain {name}: {len(ch)} chunks")

        if len(ch) < need:
            raise ValueError(
                f"Domain {name} has only {len(ch)} chunks, but need at least {need} "
                f"(train=110, test=20, cache={CACHE_CH})."
            )

        tr = ch[:110]
        te = ch[110:130]
        cache = ch[130:130 + CACHE_CH]
        out.append((tr, te, cache))
    return out



def check_finite_tensor(name, x):
    if not torch.isfinite(x).all():
        bad = x[~torch.isfinite(x)]
        sample = bad[:5].detach().cpu().tolist() if bad.numel() > 0 else []
        raise ValueError(f"Non-finite values in {name}; sample={sample}")



def lm_loss(model, ids, device):
    ids = ids.to(device)
    attn = torch.ones_like(ids, device=device)
    out = model(ids, attention_mask=attn, labels=ids)
    loss = out.loss
    if not torch.isfinite(loss):
        logits = out.logits.detach()
        raise ValueError(
            f"Non-finite lm_loss: loss={loss.item()}, "
            f"logits[min,max]=({float(torch.nan_to_num(logits, nan=0.0).min()):.4e}, "
            f"{float(torch.nan_to_num(logits, nan=0.0).max()):.4e})"
        )
    return loss


@torch.no_grad()
def heldout(model, chunks, device):
    if len(chunks) == 0:
        raise ValueError("heldout() received an empty chunk tensor")
    model.eval()
    vals = [lm_loss(model, chunks[i:i + BS], device).item() for i in range(0, len(chunks), BS)]
    model.train()
    return float(np.mean(vals))



def safe_basis_from_sketch(X, rank, device):
    X = torch.nan_to_num(X.detach(), nan=0.0, posinf=0.0, neginf=0.0).float()
    XT = X.T.contiguous()

    if XT.numel() == 0:
        return torch.zeros((XT.shape[0], 0), device=device, dtype=torch.float32)

    maxabs = float(XT.abs().max()) if XT.numel() > 0 else 0.0
    if maxabs < 1e-12:
        return torch.zeros((XT.shape[0], 0), device=device, dtype=torch.float32)

    XT = XT + 1e-6 * torch.randn_like(XT)

    try:
        Q, _ = torch.linalg.qr(XT, mode="reduced")
    except RuntimeError:
        Q, _ = torch.linalg.qr(XT.cpu(), mode="reduced")
        Q = Q.to(device)

    if Q.ndim != 2 or Q.shape[1] == 0:
        return torch.zeros((XT.shape[0], 0), device=device, dtype=torch.float32)

    return Q[:, :min(rank, Q.shape[1])]



def run_stream(seed, method, device, doms, model_name):
    set_all_seeds(seed)

    model, loras = build_model(model_name, device)
    model.train()
    named_params = [(f"lora_{li}.A", l.A) for li, l in enumerate(loras)] + [(f"lora_{li}.B", l.B) for li, l in enumerate(loras)]
    params = [p for _, p in named_params]
    lr = LR_P1 if "gpt2" in model_name.lower() else LR_P2
    opt = torch.optim.AdamW(params, lr=lr, eps=ADAM_EPS, weight_decay=0.0)

    def gvec():
        grads = []
        for p in params:
            if p.grad is None:
                grads.append(torch.zeros_like(p).flatten())
            else:
                grads.append(p.grad.detach().flatten())
        return torch.cat(grads)

    def setgrad(g):
        i = 0
        for p in params:
            n = p.numel()
            p.grad = g[i:i + n].view_as(p).clone()
            i += n

    def precond():
        segs = []
        for p in params:
            st = opt.state.get(p, {})
            if "exp_avg_sq" in st:
                v = st["exp_avg_sq"]
                step = st.get("step", torch.tensor(1))
                if torch.is_tensor(step):
                    step = step.item()
                step = max(step, 1)
                beta2 = opt.defaults.get("betas", (0.9, 0.999))[1]
                eps = opt.defaults.get("eps", 1e-8)
                bc = 1 - beta2 ** step
                segs.append((1.0 / (torch.sqrt(v / bc) + eps)).flatten())
            else:
                segs.append(torch.ones(p.numel(), device=p.device))
        return torch.cat(segs)

    prot_bases = [[] for _ in loras]
    T = len(doms)
    held = np.full((T, T), np.nan, dtype=np.float64)

    for t, (tr, te, cache) in enumerate(doms):
        if len(tr) < BS:
            raise ValueError(f"Training split for domain {t} is too small: len(tr)={len(tr)}, BS={BS}")
        if method in ("func-sign", "func-sign-adam") and t > 0 and len(cache) == 0:
            raise ValueError(f"Cache split for domain {t} is empty, func-sign methods need past cache chunks")

        Qs = None
        if method == "protect-all" and t > 0:
            Qs = []
            for mi in range(len(loras)):
                cols = [u for u in prot_bases[mi][:t] if u is not None and u.numel() > 0 and u.shape[1] > 0]
                if len(cols) == 0:
                    Qs.append(None)
                else:
                    M = torch.cat(cols, dim=1)
                    try:
                        Q = torch.linalg.qr(M, mode="reduced")[0]
                    except RuntimeError:
                        Q = torch.linalg.qr(M.cpu(), mode="reduced")[0].to(device)
                    Qs.append(Q)

        for _ in range(STEPS):
            i = random.randint(0, len(tr) - BS)          # plain CPU int: no device kwarg
            loss = lm_loss(model, tr[i:i + BS], device)
            opt.zero_grad(set_to_none=True)
            loss.backward()

            if method == "protect-all" and Qs is not None:
                with torch.no_grad():
                    for l, Q in zip(loras, Qs):
                        if Q is not None and Q.numel() > 0 and Q.shape[1] > 0:
                            l.A.grad -= Q @ (Q.T @ l.A.grad)

            elif method in ("func-sign", "func-sign-adam") and t > 0:
                g = gvec()

                u = random.randint(0, t - 1)
                cu = doms[u][2]
                j = random.randint(0, max(len(cu) - 2, 0))

                opt.zero_grad(set_to_none=True)
                lm_loss(model, cu[j:j + 2], device).backward()
                nA = gvec()

                if method == "func-sign":
                    dot = float(torch.dot(g, nA))
                    nrm2 = float(nA.norm() ** 2)
                    if nrm2 > 1e-12 and dot < 0:
                        g = g - dot / nrm2 * nA
                else:
                    P = precond()
                    dot = float(torch.dot(P * g, nA))
                    den = float(torch.dot(P * nA, nA))
                    if den > 1e-12 and dot < 0:
                        g = g - dot / den * nA

                setgrad(g)

            total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=MAX_GRAD_NORM)
            if not torch.isfinite(total_norm):
                raise ValueError(f"Non-finite gradient norm before step: {float(total_norm)}")
            check_named_tensors_finite([(n, p.grad) for n, p in named_params], "grad-before-step")
            check_named_tensors_finite(named_params, "param-before-step")

            opt.step()

            check_named_tensors_finite(named_params, "param-after-step")
            for p_name, p in named_params:
                st = opt.state.get(p, {})
                for k, v in st.items():
                    if torch.is_tensor(v) and not torch.isfinite(v).all():
                        raise ValueError(f"Non-finite optimizer state after step: {p_name}/{k}")

        empty_basis = 0
        for mi, l in enumerate(loras):
            X = l.x_sketch
            if X is None:
                X = torch.zeros(8, l.A.shape[0], device=device, dtype=torch.float32)
            U = safe_basis_from_sketch(X, RANK_PROT, device)
            if U.shape[1] == 0:
                empty_basis += 1
            prot_bases[mi].append(U)

        for i in range(t + 1):
            held[i, t] = heldout(model, doms[i][1], device)

        tlog(
            f"[{model_name.split('/')[-1]}] seed{seed} {method}: dom{t} done; "
            f"empty_bases={empty_basis}/{len(loras)}; held "
            + " ".join(f"{held[i, t]:.3f}" for i in range(t + 1))
        )

    final = held[:, T - 1]
    diag = np.array([held[i, i] for i in range(T)])

    if np.isnan(final).any() or np.isnan(diag).any():
        raise ValueError(f"NaN detected in heldout matrix for seed={seed}, method={method}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dict(
        final=float(final.mean()),
        forget=float(np.mean(final[:-1] - diag[:-1]))
    )



def run_part(tag, model_name, methods, seeds, doms, results):
    for method in methods:
        per = {}
        if USE_THREADS and len(DEVICES) > 1:
            with ThreadPoolExecutor(len(DEVICES)) as ex:
                futs = {
                    ex.submit(run_stream, s, method, DEVICES[i % len(DEVICES)], doms, model_name): s
                    for i, s in enumerate(seeds)
                }
                pending = set(futs)
                while pending:
                    done, pending = fwait(pending, timeout=120)
                    for f in done:
                        seed = futs[f]
                        try:
                            per[seed] = f.result()
                        except Exception as e:
                            raise RuntimeError(f"{tag}/{method} failed on seed {seed}: {e}") from e
                    log(f"{tag}/{method}: {len(seeds) - len(pending)}/{len(seeds)} seeds done")
        else:
            for i, s in enumerate(seeds):
                device = DEVICES[i % len(DEVICES)]
                try:
                    per[s] = run_stream(s, method, device, doms, model_name)
                except Exception as e:
                    raise RuntimeError(f"{tag}/{method} failed on seed {s}: {e}") from e
                log(f"{tag}/{method}: {i + 1}/{len(seeds)} seeds done")

        results[f"{tag}/{method}"] = {str(s): per[s] for s in seeds}

        fm = [per[s]["final"] for s in seeds]
        gm = [per[s]["forget"] for s in seeds]
        log(
            f"== {tag}/{method:15s}: avg final {np.mean(fm):.3f}+/-{np.std(fm):.3f}   "
            f"forgetting {np.mean(gm):.3f}+/-{np.std(gm):.3f}"
        )

        with open("func_gate_v2_results.json", "w") as f:
            json.dump(results, f, indent=1)



def paired_tests(results, seeds, tag):
    pairs = [
        ("func-sign-adam", "func-sign"),
        ("func-sign-adam", "protect-all"),
        ("func-sign-adam", "naive"),
        ("func-sign", "protect-all"),
    ]
    for a, b in pairs:
        ka = f"{tag}/{a}"
        kb = f"{tag}/{b}"
        if ka not in results or kb not in results:
            continue
        for metric in ("final", "forget"):
            xa = [results[ka][str(s)][metric] for s in seeds]
            xb = [results[kb][str(s)][metric] for s in seeds]
            p = stats.ttest_rel(xb, xa).pvalue
            log(
                f"paired {tag}: {a} vs {b} on {metric}: "
                f"{np.mean(xa):.4f} vs {np.mean(xb):.4f}, diff={np.mean(xb) - np.mean(xa):.4f}, p={p:.4f}"
            )


if __name__ == "__main__":
    log(f"SCRIPT VERSION {SCRIPT_VERSION}")
    log(f"devices = {DEVICES}")
    log(f"USE_THREADS = {USE_THREADS}, LR_P1 = {LR_P1}, LR_P2 = {LR_P2}, ADAM_EPS = {ADAM_EPS}, MAX_GRAD_NORM = {MAX_GRAD_NORM}")

    results = {}

    if RUN_P1:
        from transformers import GPT2TokenizerFast

        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        doms = get_domains(tok)
        log("P1 (gpt2): Adam-aware sign gate, 5 seeds")
        run_part(
            "P1-gpt2",
            "gpt2",
            ["naive", "protect-all", "func-sign", "func-sign-adam"],
            SEEDS_P1,
            doms,
            results,
        )
        paired_tests(results, SEEDS_P1, "P1-gpt2")

    if RUN_P2:
        from transformers import AutoTokenizer

        tok2 = AutoTokenizer.from_pretrained(MODEL_P2)
        if tok2.pad_token_id is None:
            tok2.pad_token = tok2.eos_token
        doms2 = get_domains(tok2)
        log(f"P2 ({MODEL_P2}): scale check, 3 seeds")
        run_part(
            "P2-" + MODEL_P2.split("/")[-1],
            MODEL_P2,
            ["naive", "protect-all", "func-sign-adam"],
            SEEDS_P2,
            doms2,
            results,
        )

    log("ALL DONE -> func_gate_v2_results.json")