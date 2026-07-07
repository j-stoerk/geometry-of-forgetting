# =====================================================================
# kaggle_jspace_interference_exact.py
#
# Exact bridging experiment with two parallel H1 measurements:
#   H1-grad : interference energy on A through g = d CE_A / d h_l
#   H1-KL   : actual KL on A between old and new predictive distributions
#
# H2 compares:
#   naive, protect-Jspace, protect-Sigma, protect-random
#
# Writes: jspace_interference_results.json
# =====================================================================

import os
import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL = os.getenv("JSPACE_MODEL", "EleutherAI/pythia-410m")
RANK_J = int(os.getenv("JSPACE_RANK", 64))
HEAD_R = int(os.getenv("JSPACE_HEAD_R", 16))
SEQ = int(os.getenv("JSPACE_SEQ", 128))
BS = int(os.getenv("JSPACE_BS", 8))
LR = float(os.getenv("JSPACE_LR", 3e-4))
STEPS = int(os.getenv("JSPACE_STEPS", 200))
JS_CTX = int(os.getenv("JSPACE_JS_CTX", 200))
JS_PROBES = int(os.getenv("JSPACE_JS_PROBES", 96))
JS_TOPK = int(os.getenv("JSPACE_JS_TOPK", 20))
SIGMA_CTX = int(os.getenv("JSPACE_SIGMA_CTX", 24))
EVAL_BATCH = int(os.getenv("JSPACE_EVAL_BATCH", 4))
SEED = int(os.getenv("JSPACE_SEED", 0))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def choose_device():
    if not torch.cuda.is_available():
        log("CUDA not available -> using CPU")
        return torch.device("cpu")
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    caps = [torch.cuda.get_device_capability(i) for i in range(n)]
    log("Visible CUDA devices: " + " | ".join(f"{i}:{names[i]} sm{caps[i][0]}{caps[i][1]}" for i in range(n)))
    return torch.device("cuda:0")


DEV = choose_device()


def orthonormal_cols(x):
    q, _ = torch.linalg.qr(x, mode="reduced")
    return q


def causal_ce(logits, ids):
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        ids[:, 1:].reshape(-1),
    )


def principal_overlap(P1, P2):
    s = torch.linalg.svdvals(P1.T @ P2)
    return {
        "principal_cosines": [float(v) for v in s.detach().cpu()],
        "mean_squared_overlap": float(s.square().mean().item()),
        "max_cosine": float(s.max().item()),
        "min_cosine": float(s.min().item()),
    }


def build(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(DEV)
    model.config.use_cache = False
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def get_layer_module(model, layer_idx):
    return model.gpt_neox.layers[layer_idx]


def get_domains(tok):
    from datasets import load_dataset

    out = []
    specs = [
        ("news",   load_dataset("ag_news", split="train"), lambda x: x["label"] == 0, 1500),
        ("imdb",   load_dataset("imdb", split="train"), lambda x: True,                700),
        ("wiki",   load_dataset("wikitext", "wikitext-2-raw-v1", split="train"),
                   lambda x: len(x["text"]) > 200, 1400),
        ("tweets", load_dataset("tweet_eval", "sentiment", split="train"), lambda x: True, 4500),
    ]

    for name, ds, filt, n in specs:
        texts = []
        for x in ds:
            if filt(x):
                t = x["text"]
                if isinstance(t, str) and t.strip():
                    texts.append(t)
            if len(texts) >= n:
                break
        ids = tok("\n\n".join(texts), return_tensors="pt").input_ids[0]
        usable = (len(ids) // SEQ) * SEQ
        chunks = ids[:usable].view(-1, SEQ)
        out.append((name, chunks[:110], chunks[110:134]))
    return out


def sample_batch(chunks, batch_size):
    idx = torch.randint(0, len(chunks), (batch_size,))
    return chunks[idx].to(DEV)


def mixed_contexts_for_jspace(doms, n_ctx):
    xs = []
    for _, tr, va in doms:
        for i in range(len(va)):
            xs.append(va[i])
        for i in range(min(len(tr), 32)):
            xs.append(tr[i])
    random.shuffle(xs)
    return xs[:n_ctx]


class Head(nn.Module):
    def __init__(self, d, r=HEAD_R):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(d, r))
        self.B = nn.Parameter(torch.randn(r, d) * 1e-3)

    def delta(self, h):
        return (h @ self.A) @ self.B


class Injector:
    def __init__(self, model, layer_idx, head=None, capture_grad=False, basis=None, mode="full"):
        self.layer = get_layer_module(model, layer_idx)
        self.head = head
        self.capture_grad = capture_grad
        self.basis = basis
        self.mode = mode  # full | proj | orth
        self.handle = None
        self.grad_h = None
        self.hidden_h = None

    def _apply_basis(self, dh):
        if self.basis is None or self.mode == "full":
            return dh
        dh_proj = (dh @ self.basis) @ self.basis.T
        if self.mode == "proj":
            return dh_proj
        if self.mode == "orth":
            return dh - dh_proj
        raise ValueError(f"unknown mode: {self.mode}")

    def _hook(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        self.hidden_h = h
        if self.capture_grad:
            h.requires_grad_(True)
            h.retain_grad()
            self.grad_h = h
        if self.head is not None:
            dh = self.head.delta(h)
            dh = self._apply_basis(dh)
            h = h + dh
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()


def fit_jspace_basis(model, layer_idx, contexts, rank):
    d = model.config.hidden_size
    sketch = torch.zeros(JS_PROBES, d, device=DEV)

    for p in range(JS_PROBES):
        acc = torch.zeros(d, device=DEV)
        k = 0
        for ids in contexts:
            ids = ids.view(1, -1).to(DEV)
            model.zero_grad(set_to_none=True)
            with Injector(model, layer_idx, capture_grad=True) as inj:
                logits = model(ids, attention_mask=torch.ones_like(ids)).logits[0]
                last = logits[-1]
                top = last.topk(JS_TOPK).indices
                probe = torch.zeros_like(last)
                probe[top] = torch.randn(JS_TOPK, device=DEV)
                (probe @ last).backward()
                g_last = inj.grad_h.grad[0, -1].detach()
                acc += g_last
                k += 1
        sketch[p] = acc / max(k, 1)

    return orthonormal_cols(sketch.T)[:, :rank]


def cached_or_fit_jspace_basis(model, layer_idx, contexts, rank):
    safe = MODEL.replace("/", "__")
    cache_path = Path(f"jspace_basis_{safe}_L{layer_idx}_r{rank}.pt")
    if cache_path.exists():
        obj = torch.load(cache_path, map_location=DEV)
        P = obj["basis"].to(DEV)
        log(f"loaded cached J-space basis from {cache_path}")
        return orthonormal_cols(P)[:, :rank]

    fit_ctx = contexts[:JS_CTX]
    log(f"no cached J-space basis found -> fitting from {len(fit_ctx)} prompts")
    P = fit_jspace_basis(model, layer_idx, fit_ctx, rank).detach().cpu()
    torch.save({"basis": P, "model": MODEL, "layer": layer_idx, "rank": rank}, cache_path)
    log(f"saved J-space basis to {cache_path}")
    return P.to(DEV)


def sigma_A_basis(model, layer_idx, A_contexts, rank):
    G = []
    for ids in A_contexts[:SIGMA_CTX]:
        ids = ids.view(1, -1).to(DEV)
        model.zero_grad(set_to_none=True)
        with Injector(model, layer_idx, capture_grad=True) as inj:
            logits = model(ids, attention_mask=torch.ones_like(ids)).logits
            loss = causal_ce(logits, ids)
            loss.backward()
            g = inj.grad_h.grad[0].detach()
            G.append(g)
    G = torch.cat(G, dim=0)
    Sigma = (G.T @ G) / max(len(G), 1)
    _, evecs = torch.linalg.eigh(Sigma)
    return orthonormal_cols(evecs[:, -rank:])


def true_ce(model, chunks, head=None, layer_idx=None, basis=None, mode="full"):
    total = 0.0
    total_n = 0
    with torch.no_grad():
        for i in range(0, len(chunks), EVAL_BATCH):
            ids = chunks[i:i+EVAL_BATCH].to(DEV)
            if head is None:
                logits = model(ids, attention_mask=torch.ones_like(ids)).logits
            else:
                with Injector(model, layer_idx, head=head, basis=basis, mode=mode):
                    logits = model(ids, attention_mask=torch.ones_like(ids)).logits
            loss = causal_ce(logits, ids)
            total += float(loss) * len(ids)
            total_n += len(ids)
    return total / max(total_n, 1)


def tokenwise_kl_from_logits(base_logits, new_logits):
    base = base_logits[:, :-1, :]
    new = new_logits[:, :-1, :]
    logp0 = F.log_softmax(base, dim=-1)
    p0 = logp0.exp()
    logp1 = F.log_softmax(new, dim=-1)
    return (p0 * (logp0 - logp1)).sum(dim=-1)


def h1_dual_metrics_on_A(model, layer_idx, A_chunks, head, P_J):
    grad_E_full = 0.0
    grad_E_J = 0.0
    grad_E_orth = 0.0

    kl_full = 0.0
    kl_J = 0.0
    kl_orth = 0.0

    n_tokens_grad = 0
    n_tokens_kl = 0

    for i in range(0, len(A_chunks), EVAL_BATCH):
        ids = A_chunks[i:i+EVAL_BATCH].to(DEV)

        model.zero_grad(set_to_none=True)
        with Injector(model, layer_idx, capture_grad=True) as inj:
            base_logits = model(ids, attention_mask=torch.ones_like(ids)).logits
            loss = causal_ce(base_logits, ids)
            loss.backward()
            g = inj.grad_h.grad.detach()     # [B, S, D]
            h = inj.hidden_h.detach()        # [B, S, D]

        with torch.no_grad():
            dh_full = head.delta(h)
            dh_J = (dh_full @ P_J) @ P_J.T
            dh_orth = dh_full - dh_J

            s_full = (g * dh_full).sum(dim=-1)
            s_J = (g * dh_J).sum(dim=-1)
            s_orth = (g * dh_orth).sum(dim=-1)

            grad_E_full += float(s_full.square().sum().item())
            grad_E_J += float(s_J.square().sum().item())
            grad_E_orth += float(s_orth.square().sum().item())
            n_tokens_grad += s_full.numel()

            with Injector(model, layer_idx, head=head, basis=P_J, mode="full"):
                logits_full = model(ids, attention_mask=torch.ones_like(ids)).logits
            with Injector(model, layer_idx, head=head, basis=P_J, mode="proj"):
                logits_J = model(ids, attention_mask=torch.ones_like(ids)).logits
            with Injector(model, layer_idx, head=head, basis=P_J, mode="orth"):
                logits_orth = model(ids, attention_mask=torch.ones_like(ids)).logits

            kf = tokenwise_kl_from_logits(base_logits.detach(), logits_full)
            kj = tokenwise_kl_from_logits(base_logits.detach(), logits_J)
            ko = tokenwise_kl_from_logits(base_logits.detach(), logits_orth)

            kl_full += float(kf.sum().item())
            kl_J += float(kj.sum().item())
            kl_orth += float(ko.sum().item())
            n_tokens_kl += kf.numel()

    return {
        "grad_energy_full": grad_E_full / max(n_tokens_grad, 1),
        "grad_energy_J": grad_E_J / max(n_tokens_grad, 1),
        "grad_energy_orth": grad_E_orth / max(n_tokens_grad, 1),
        "grad_energy_frac_J": grad_E_J / max(grad_E_full, 1e-12),
        "grad_energy_frac_orth": grad_E_orth / max(grad_E_full, 1e-12),
        "kl_full": kl_full / max(n_tokens_kl, 1),
        "kl_J": kl_J / max(n_tokens_kl, 1),
        "kl_orth": kl_orth / max(n_tokens_kl, 1),
        "kl_frac_J": kl_J / max(kl_full, 1e-12),
        "kl_frac_orth": kl_orth / max(kl_full, 1e-12),
    }


def project_B_off_subspace_(head, basis):
    with torch.no_grad():
        head.B -= (head.B @ basis) @ basis.T


def train_head(model, doms, layer_idx, protect_basis=None):
    d = model.config.hidden_size
    head = Head(d, r=HEAD_R).to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=LR)

    with Injector(model, layer_idx, head=head):
        for t in (1, 2, 3):
            tr = doms[t][1]
            for _ in range(STEPS):
                ids = sample_batch(tr, BS)
                logits = model(ids, attention_mask=torch.ones_like(ids)).logits
                loss = causal_ce(logits, ids)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if protect_basis is not None:
                    project_B_off_subspace_(head, protect_basis)
    return head


if __name__ == "__main__":
    model, tok = build(MODEL)
    d = model.config.hidden_size
    layer_idx = model.config.num_hidden_layers * 2 // 3
    rank_over_d = RANK_J / d

    log(f"model={MODEL} layer={layer_idx} d={d} rank={RANK_J} head_r={HEAD_R}")
    doms = get_domains(tok)
    A_name, A_train, A_val = doms[0]
    A_contexts = [A_val[i] for i in range(len(A_val))]

    fit_contexts = mixed_contexts_for_jspace(doms, JS_CTX)
    P_J = cached_or_fit_jspace_basis(model, layer_idx, fit_contexts, RANK_J)

    log("estimating Sigma_A top-r subspace from empirical Fisher / Gauss-Newton geometry on A")
    P_S = sigma_A_basis(model, layer_idx, A_contexts, RANK_J)
    P_R = orthonormal_cols(torch.randn(d, RANK_J, device=DEV))

    ov = principal_overlap(P_J, P_S)
    log(f"principal overlap J vs Sigma_A: mean(sv^2)={ov['mean_squared_overlap']:.4f} (random baseline ~ {rank_over_d:.4f})")
    log("principal cosines (top 10): " + " ".join(f"{x:.3f}" for x in ov["principal_cosines"][:10]))

    log("training naive head for H1/H2")
    head_naive = train_head(model, doms, layer_idx, protect_basis=None)

    H1 = h1_dual_metrics_on_A(model, layer_idx, A_val, head_naive, P_J)
    log(
        f"H1-grad on A: J={100*H1['grad_energy_frac_J']:.1f}% orth={100*H1['grad_energy_frac_orth']:.1f}% baseline={100*rank_over_d:.1f}%"
    )
    log(
        f"H1-KL on A:   J={100*H1['kl_frac_J']:.1f}% orth={100*H1['kl_frac_orth']:.1f}% baseline={100*rank_over_d:.1f}%"
    )

    base_A = true_ce(model, A_val)
    H2 = {}
    for name, P in [
        ("naive", None),
        ("protect-Jspace", P_J),
        ("protect-Sigma", P_S),
        ("protect-random", P_R),
    ]:
        log(f"training H2 arm: {name}")
        h = train_head(model, doms, layer_idx, protect_basis=P)
        H2[name] = float(true_ce(model, A_val, head=h, layer_idx=layer_idx) - base_A)

    log("H2 forgetting on A (nats): " + " ".join(f"{k}={v:+.4f}" for k, v in H2.items()))

    out = {
        "model": MODEL,
        "device": str(DEV),
        "layer": layer_idx,
        "rank": RANK_J,
        "head_rank": HEAD_R,
        "seq": SEQ,
        "steps": STEPS,
        "batch_size": BS,
        "random_baseline_rank_over_d": rank_over_d,
        "overlap_J_vs_Sigma": ov,
        "H1": H1,
        "H2": H2,
        "notes": {
            "H1_grad_definition": "Gradient-space interference energy on A using g = d CE_A / d h_l and the later-task-induced hidden change Δh, reported for full / J-projected / J-orthogonal components as E[(g·Δh)^2].",
            "H1_KL_definition": "Actual KL on A between baseline and perturbed predictive distributions, reported for full / J-projected / J-orthogonal components.",
            "Sigma_definition": "Top-r eigenspace of empirical Fisher / Gauss-Newton geometry E[g g^T] at the hooked layer on A.",
        },
    }

    with open("jspace_interference_results.json", "w") as f:
        json.dump(out, f, indent=2)

    log("ALL DONE -> jspace_interference_results.json")
    print("\nReading:")
    print(" H1-grad supports the bridge if J >> rank/d and orth is small under E[(g·Δh)^2].")
    print(" H1-KL supports the bridge if KL(J) >> rank/d and KL(orth) is small on A.")
    print(" H2 supports it if protect-Jspace ≈ protect-Sigma << naive while protect-random ≈ naive.")