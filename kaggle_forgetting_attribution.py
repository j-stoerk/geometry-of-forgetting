# kaggle_forgetting_attribution_v2.py
#
# Real-LLM forgetting attribution in the spirit of main.pdf:
# - Exact forgetting observable: anchored per-token KL on old domains
# - Certified incompatibility upper bound: domain-probe CE on frozen features
# - Class-limited term at rank r: joint KL to single-domain references
# - Control term: sequential KL minus joint KL
# - Gate recovery: how much control the active-set QP removes
#
# The probe bound is reported honestly as an upper bound. When it is small,
# the joint term is capacity-dominated and the scaling curve is the capacity curve.

import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import nnls


SCRIPT_VERSION = "attr-v2.0"
MODELS = [
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    # "EleutherAI/pythia-2.8b",
]

DOMAINS = ["news", "imdb", "wiki", "tweets"]

SEEDS_SINGLE = [0, 1]
SEEDS_JOINT = [0, 1]
SEEDS_SEQ = [0, 1, 2]

RANKS = [2, 8, 32]
RANK_DEPLOY = 8

STEPS_PER_DOMAIN = 250
BATCH_SIZE = 8
SEQ_LEN = 128
ANCHOR_CHUNKS = 8
EVAL_CHUNKS = 24
TRAIN_CHUNKS = 110

LR = 1e-4
ADAM_EPS = 1e-6
MAX_GRAD_NORM = 0.5
WEIGHT_DECAY = 0.0

RESULTS_FILE = "forgetting_attribution_v2_results.json"
CKPT_DIR = Path("attr_ckpts_v2")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32

T0 = time.time()


def log(msg):
    dt = time.time() - T0
    print(f"[{dt:8.1f}s] {msg}", flush=True)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LoRA(nn.Module):
    def __init__(self, base, d_in, d_out, r, device):
        super().__init__()
        self.base = base
        self.scale = 2.0 / r
        self.A = nn.Parameter(torch.zeros(d_in, r, device=device, dtype=torch.float32))
        self.B = nn.Parameter(torch.empty(r, d_out, device=device, dtype=torch.float32))
        nn.init.normal_(self.B, std=1e-6)

    def forward(self, x):
        base_out = self.base(x)
        delta = ((x.to(self.A.dtype) @ self.A) @ self.B) * self.scale
        return base_out + delta.to(base_out.dtype)


def build_model(model_name, rank, device=DEVICE, dtype=BASE_DTYPE):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device)

    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False

    for p in model.parameters():
        p.requires_grad_(False)

    if not hasattr(model, "gpt_neox"):
        raise ValueError("This script currently expects a GPT-NeoX/Pythia-style model.")

    loras = []
    hidden = model.config.hidden_size
    d_in, d_out = hidden, 3 * hidden
    for block in model.gpt_neox.layers:
        base = block.attention.query_key_value
        wrapped = LoRA(base, d_in, d_out, rank, device)
        block.attention.query_key_value = wrapped
        loras.append(wrapped)

    return model, loras


def lora_state_dict(loras):
    return {
        k: v
        for i, l in enumerate(loras)
        for k, v in {
            f"{i}.A": l.A.detach().cpu(),
            f"{i}.B": l.B.detach().cpu(),
        }.items()
    }

def save_lora(loras, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    for i, l in enumerate(loras):
        state[f"{i}.A"] = l.A.detach().cpu()
        state[f"{i}.B"] = l.B.detach().cpu()
    torch.save(state, path)


def load_lora(loras, path, device=DEVICE):
    state = torch.load(path, map_location="cpu")
    for i, l in enumerate(loras):
        l.A.data.copy_(state[f"{i}.A"].to(device=device, dtype=l.A.dtype))
        l.B.data.copy_(state[f"{i}.B"].to(device=device, dtype=l.B.dtype))


def lora_params(loras):
    return [p for l in loras for p in (l.A, l.B)]


def build_optimizer(loras):
    return torch.optim.AdamW(
        lora_params(loras),
        lr=LR,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
    )


def shift_logits_and_labels(logits, ids):
    return logits[:, :-1, :].contiguous(), ids[:, 1:].contiguous()


def lm_loss(model, ids, device=DEVICE):
    ids = ids.to(device)
    out = model(ids, attention_mask=torch.ones_like(ids), labels=ids)
    if not torch.isfinite(out.loss):
        raise ValueError("Non-finite LM loss.")
    return out.loss


@torch.no_grad()
def mean_ce(model, chunks, batch_size=4, device=DEVICE):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for i in range(0, len(chunks), batch_size):
        ids = chunks[i:i + batch_size].to(device)
        loss = lm_loss(model, ids, device)
        total_loss += float(loss) * len(ids)
        total_batches += len(ids)
    model.train()
    return total_loss / max(total_batches, 1)


@torch.no_grad()
def token_kl(old_model, new_model, chunks, batch_size=2, device=DEVICE):
    old_model.eval()
    new_model.eval()
    total_kl = 0.0
    total_tokens = 0

    for i in range(0, len(chunks), batch_size):
        ids = chunks[i:i + batch_size].to(device)

        old_logits = old_model(ids, attention_mask=torch.ones_like(ids)).logits
        new_logits = new_model(ids, attention_mask=torch.ones_like(ids)).logits

        old_logits, labels = shift_logits_and_labels(old_logits, ids)
        new_logits, _ = shift_logits_and_labels(new_logits, ids)

        old_logp = F.log_softmax(old_logits.float(), dim=-1)
        new_logp = F.log_softmax(new_logits.float(), dim=-1)
        old_p = old_logp.exp()

        kl = F.kl_div(new_logp, old_p, reduction="sum", log_target=False)
        total_kl += float(kl)
        total_tokens += int(labels.numel())

    old_model.train()
    new_model.train()
    return total_kl / max(total_tokens, 1)


def get_domains(tokenizer):
    from datasets import load_dataset

    texts = []

    news = load_dataset("ag_news", split="train")
    texts.append(("news", "\n\n".join([x["text"] for x in news if x["label"] == 0][:1500])))

    imdb = load_dataset("imdb", split="train")
    texts.append(("imdb", "\n\n".join([x["text"] for x in imdb][:700])))

    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts.append(("wiki", "\n\n".join([x["text"] for x in wiki if len(x["text"]) > 200][:1400])))

    tw = load_dataset("tweet_eval", "sentiment", split="train")
    texts.append(("tweets", "\n\n".join([x["text"] for x in tw][:4500])))

    need = TRAIN_CHUNKS + EVAL_CHUNKS + ANCHOR_CHUNKS
    out = []

    for name, text in texts:
        enc = tokenizer(text, return_tensors="pt").input_ids[0]
        ch = enc[: (len(enc) // SEQ_LEN) * SEQ_LEN].view(-1, SEQ_LEN)
        if len(ch) < need:
            raise ValueError(f"{name}: {len(ch)} chunks < {need}")
        tr = ch[:TRAIN_CHUNKS]
        te = ch[TRAIN_CHUNKS:TRAIN_CHUNKS + EVAL_CHUNKS]
        an = ch[TRAIN_CHUNKS + EVAL_CHUNKS:TRAIN_CHUNKS + EVAL_CHUNKS + ANCHOR_CHUNKS]
        out.append((name, tr, te, an))
        log(f"domain {name}: {len(ch)} chunks")
    return out


def grad_vector(params):
    return torch.cat([
        (torch.zeros(p.numel(), device=p.device, dtype=torch.float32)
         if p.grad is None else p.grad.detach().flatten().float())
        for p in params
    ])


def set_grad_vector(params, g):
    i = 0
    for p in params:
        n = p.numel()
        p.grad = g[i:i+n].view_as(p).to(p.dtype).clone()
        i += n


def train_one_step(model, opt, ids):
    opt.zero_grad(set_to_none=True)
    loss = lm_loss(model, ids)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    if not torch.isfinite(gn):
        raise ValueError("Non-finite grad norm.")
    opt.step()
    return float(loss)


def train_adapter_on_single_domain(model_name, rank, seed, domain_idx, domains):
    set_all_seeds(seed)
    model, loras = build_model(model_name, rank)
    opt = build_optimizer(loras)
    model.train()

    name, tr, te, _ = domains[domain_idx]
    for _ in range(STEPS_PER_DOMAIN):
        i = random.randint(0, len(tr) - BATCH_SIZE)
        ids = tr[i:i + BATCH_SIZE]
        opt.zero_grad(set_to_none=True)
        loss = lm_loss(model, ids)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(lora_params(loras), MAX_GRAD_NORM)
        if not torch.isfinite(gn):
            raise ValueError("Non-finite grad norm.")
        opt.step()

    eval_ce = mean_ce(model, te)
    ckpt = CKPT_DIR / model_name.split("/")[-1] / f"single_r{rank}_d{domain_idx}_s{seed}.pt"
    save_lora(loras, ckpt)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"domain": name, "eval_ce": float(eval_ce), "ckpt": str(ckpt)}


def train_joint(model_name, rank, seed, domains):
    set_all_seeds(seed)
    model, loras = build_model(model_name, rank)
    opt = build_optimizer(loras)
    model.train()

    T = len(domains)
    for s in range(T * STEPS_PER_DOMAIN):
        t = s % T
        _, tr, _, _ = domains[t]
        i = random.randint(0, len(tr) - BATCH_SIZE)
        ids = tr[i:i + BATCH_SIZE]
        opt.zero_grad(set_to_none=True)
        loss = lm_loss(model, ids)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(lora_params(loras), MAX_GRAD_NORM)
        if not torch.isfinite(gn):
            raise ValueError("Non-finite grad norm.")
        opt.step()

    evals = {}
    for t, (name, _, te, _) in enumerate(domains):
        evals[name] = float(mean_ce(model, te))

    ckpt = CKPT_DIR / model_name.split("/")[-1] / f"joint_r{rank}_s{seed}.pt"
    save_lora(loras, ckpt)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"eval_ce": evals, "ckpt": str(ckpt)}


def train_sequential(model_name, rank, seed, domains, method="naive"):
    set_all_seeds(seed)
    model, loras = build_model(model_name, rank)
    opt = build_optimizer(loras)
    params = lora_params(loras)
    model.train()

    T = len(domains)
    diag_ce = {}
    for t, (name, tr, te, anchor) in enumerate(domains):
        for _ in range(STEPS_PER_DOMAIN):
            i = random.randint(0, len(tr) - BATCH_SIZE)
            ids = tr[i:i + BATCH_SIZE]

            opt.zero_grad(set_to_none=True)
            loss = lm_loss(model, ids)
            loss.backward()

            if method == "func-active" and t > 0:
                g = grad_vector(params)
                cols = []
                for u in range(t):
                    _, _, _, cache = domains[u]
                    j = random.randint(0, max(len(cache) - 2, 0))
                    opt.zero_grad(set_to_none=True)
                    lm_loss(model, cache[j:j + 2]).backward()
                    gu = grad_vector(params)
                    if float(gu.norm() ** 2) > 1e-12:
                        cols.append(gu)

                if cols:
                    G = torch.stack(cols, dim=1).cpu().double().numpy()
                    lam, _ = nnls(G, (-g).cpu().double().numpy())
                    if lam.any():
                        g = g + torch.as_tensor(G @ lam, dtype=g.dtype, device=g.device)
                    set_grad_vector(params, g)

            gn = torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
            if not torch.isfinite(gn):
                raise ValueError("Non-finite grad norm.")
            opt.step()

        diag_ce[name] = float(mean_ce(model, te))

    final_ce = {name: float(mean_ce(model, te)) for name, _, te, _ in domains}
    ckpt = CKPT_DIR / model_name.split("/")[-1] / f"seq_{method}_r{rank}_s{seed}.pt"
    save_lora(loras, ckpt)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"diag_ce": diag_ce, "final_ce": final_ce, "ckpt": str(ckpt)}


@torch.no_grad()
def load_model_with_adapter(model_name, rank, ckpt_path=None):
    model, loras = build_model(model_name, rank)
    if ckpt_path is not None:
        load_lora(loras, ckpt_path)
    model.eval()
    return model, loras


def probe_floor_upper_bound(model_name, domains):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=BASE_DTYPE
    ).to(DEVICE)
    model.eval()

    feats_tr, labs_tr = [], []
    feats_te, labs_te = [], []

    with torch.no_grad():
        for t, (_, tr, te, _) in enumerate(domains):
            for chunks, feat_store, lab_store in [
                (tr[:32], feats_tr, labs_tr),
                (te[:16], feats_te, labs_te),
            ]:
                for i in range(0, len(chunks), 4):
                    ids = chunks[i:i + 4].to(DEVICE)
                    hs = model(
                        ids,
                        attention_mask=torch.ones_like(ids),
                        output_hidden_states=True,
                    ).hidden_states[-1]
                    feat = hs.mean(dim=1).float().cpu()
                    feat_store.append(feat)
                    lab_store.append(torch.full((len(ids),), t, dtype=torch.long))

    Xtr = torch.cat(feats_tr, dim=0)
    ytr = torch.cat(labs_tr, dim=0)
    Xte = torch.cat(feats_te, dim=0)
    yte = torch.cat(labs_te, dim=0)

    mu = Xtr.mean(dim=0)
    sd = Xtr.std(dim=0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    W = torch.zeros(Xtr.shape[1], len(domains), requires_grad=True)
    opt = torch.optim.Adam([W], lr=1e-2)

    for _ in range(400):
        opt.zero_grad()
        logits = Xtr @ W
        loss = F.cross_entropy(logits, ytr) + 1e-3 * W.square().sum()
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = Xte @ W
        ce = float(F.cross_entropy(logits, yte))
        acc = float((logits.argmax(dim=1) == yte).float().mean())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"probe_ce": ce, "probe_acc": acc}


@torch.no_grad()
def mean_kl_to_single_refs(model_name, rank, current_ckpt, single_ref_ckpts, domains, include_last=False):
    current_model, _ = load_model_with_adapter(model_name, rank, current_ckpt)
    per_domain = {}

    stop = len(domains) if include_last else len(domains) - 1
    for d in range(stop):
        teacher_model, _ = load_model_with_adapter(model_name, rank, single_ref_ckpts[d])
        name, _, _, anchor = domains[d]
        per_domain[name] = float(token_kl(teacher_model, current_model, anchor))
        del teacher_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del current_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mean_val = float(np.mean(list(per_domain.values()))) if per_domain else 0.0
    return {"mean": mean_val, "per_domain": per_domain}


def summarize_model(results, model_name):
    key = model_name.split("/")[-1]
    R = results[key]

    out = {"model": key}

    out["probe"] = R["probe"]

    single_refs = {}
    for r in RANKS:
        single_refs[r] = {}
        for d in range(len(DOMAINS)):
            seeds = R[f"single/r{r}/d{d}"]
            single_refs[r][d] = [run["ckpt"] for run in seeds.values()]

    out["single_ref_ce"] = {}
    for r in RANKS:
        vals = []
        for d in range(len(DOMAINS)):
            vals.extend([run["eval_ce"] for run in R[f"single/r{r}/d{d}"].values()])
        out["single_ref_ce"][r] = float(np.mean(vals))

    out["joint_ce"] = {}
    out["joint_kl"] = {}
    for r in RANKS:
        runs = list(R[f"joint/r{r}"].values())
        out["joint_ce"][r] = float(np.mean([
            np.mean(list(run["eval_ce"].values())) for run in runs
        ]))
        out["joint_kl"][r] = float(np.mean([run["kl_to_single"]["mean"] for run in runs]))

    out["seq"] = {}
    for method in ["naive", "func-active"]:
        runs = list(R[f"seq/{method}/r{RANK_DEPLOY}"].values())
        out["seq"][method] = {
            "kl": float(np.mean([run["kl_to_single"]["mean"] for run in runs])),
            "kl_sd": float(np.std([run["kl_to_single"]["mean"] for run in runs])),
            "ce_final": float(np.mean([
                np.mean(list(run["final_ce"].values())[:-1]) for run in runs
            ])),
        }

    out["control"] = {
        "naive_minus_joint": out["seq"]["naive"]["kl"] - out["joint_kl"][RANK_DEPLOY],
        "gate_minus_joint": out["seq"]["func-active"]["kl"] - out["joint_kl"][RANK_DEPLOY],
        "gate_removed": out["seq"]["naive"]["kl"] - out["seq"]["func-active"]["kl"],
    }

    return out


def report(summary):
    print("\n" + "=" * 78)
    print(f"FORGETTING ATTRIBUTION -- {summary['model']}")
    print("=" * 78)
    print(f"Probe CE upper bound        : {summary['probe']['probe_ce']:.4f} nats   "
          f"(probe acc {summary['probe']['probe_acc']:.3f})")
    print("Joint KL to single refs     : " + "  ".join(
        [f"r={r}: {summary['joint_kl'][r]:.4f}" for r in RANKS]
    ))
    print(f"Seq KL naive (r={RANK_DEPLOY}) : {summary['seq']['naive']['kl']:.4f} "
          f"+/- {summary['seq']['naive']['kl_sd']:.4f}")
    print(f"Seq KL gate  (r={RANK_DEPLOY}) : {summary['seq']['func-active']['kl']:.4f} "
          f"+/- {summary['seq']['func-active']['kl_sd']:.4f}")
    print(f"Control naive - joint      : {summary['control']['naive_minus_joint']:+.4f}")
    print(f"Control gate  - joint      : {summary['control']['gate_minus_joint']:+.4f}")
    print(f"Gate removed               : {summary['control']['gate_removed']:+.4f}")
    print("Interpretation:")
    print("- Probe CE is a certified upper bound on incompatibility, not the floor itself.")
    print("- Joint KL is the class-limited shared-model term at that rank.")
    print("- If probe CE << joint KL, the joint curve is capacity-dominated.")
    print("- Sequential minus joint is the policy/control residual.")
    print("- Gate removed is the amount of control the active-set QP recovers.")


def ensure_key(results, model_key, key):
    if key not in results[model_key]:
        results[model_key][key] = {}


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


if __name__ == "__main__":
    from transformers import AutoTokenizer

    log(f"SCRIPT VERSION {SCRIPT_VERSION}; device={DEVICE}; dtype={BASE_DTYPE}")
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            results = json.load(f)
        log(f"Loaded resume file: {RESULTS_FILE}")

    for model_name in MODELS:
        key = model_name.split("/")[-1]
        results.setdefault(key, {})

        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        domains = get_domains(tok)

        if "probe" not in results[key]:
            log(f"[{key}] probe upper bound")
            results[key]["probe"] = probe_floor_upper_bound(model_name, domains)
            save_json(results, RESULTS_FILE)

        # single-domain references
        for r in RANKS:
            for d in range(len(domains)):
                kk = f"single/r{r}/d{d}"
                ensure_key(results, key, kk)
                for s in SEEDS_SINGLE:
                    if str(s) in results[key][kk]:
                        continue
                    log(f"[{key}] single r={r} d={d} s={s}")
                    results[key][kk][str(s)] = train_adapter_on_single_domain(
                        model_name=model_name,
                        rank=r,
                        seed=s,
                        domain_idx=d,
                        domains=domains,
                    )
                    save_json(results, RESULTS_FILE)

        # joint references + KL to single refs
        for r in RANKS:
            kk = f"joint/r{r}"
            ensure_key(results, key, kk)

            ref_ckpts = {}
            for d in range(len(domains)):
                ref_ckpts[d] = results[key][f"single/r{r}/d{d}"][str(SEEDS_SINGLE[0])]["ckpt"]

            for s in SEEDS_JOINT:
                if str(s) in results[key][kk]:
                    continue
                log(f"[{key}] joint r={r} s={s}")
                run = train_joint(model_name=model_name, rank=r, seed=s, domains=domains)
                run["kl_to_single"] = mean_kl_to_single_refs(
                    model_name=model_name,
                    rank=r,
                    current_ckpt=run["ckpt"],
                    single_ref_ckpts=ref_ckpts,
                    domains=domains,
                    include_last=False,
                )
                results[key][kk][str(s)] = run
                save_json(results, RESULTS_FILE)

        # sequential naive and gate at deployed rank
        ref_ckpts = {}
        for d in range(len(domains)):
            ref_ckpts[d] = results[key][f"single/r{RANK_DEPLOY}/d{d}"][str(SEEDS_SINGLE[0])]["ckpt"]

        for method in ["naive", "func-active"]:
            kk = f"seq/{method}/r{RANK_DEPLOY}"
            ensure_key(results, key, kk)
            for s in SEEDS_SEQ:
                if str(s) in results[key][kk]:
                    continue
                log(f"[{key}] seq {method} r={RANK_DEPLOY} s={s}")
                run = train_sequential(
                    model_name=model_name,
                    rank=RANK_DEPLOY,
                    seed=s,
                    domains=domains,
                    method=method,
                )
                run["kl_to_single"] = mean_kl_to_single_refs(
                    model_name=model_name,
                    rank=RANK_DEPLOY,
                    current_ckpt=run["ckpt"],
                    single_ref_ckpts=ref_ckpts,
                    domains=domains,
                    include_last=False,
                )
                results[key][kk][str(s)] = run
                save_json(results, RESULTS_FILE)

        summary = summarize_model(results, model_name)
        report(summary)

    log("ALL DONE")
[    18.7s] SCRIPT VERSION attr-v2.0; device=cuda; dtype=torch.float16
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[    45.3s] domain news: 689 chunks
[    46.2s] domain imdb: 1643 chunks
[    47.3s] domain wiki: 1857 chunks
[    47.8s] domain tweets: 1052 chunks
[    47.8s] [pythia-410m] probe upper bound
`torch_dtype` is deprecated! Use `dtype` instead!
[    59.0s] [pythia-410m] single r=2 d=0 s=0
[    88.7s] [pythia-410m] single r=2 d=0 s=1
[   119.4s] [pythia-410m] single r=2 d=1 s=0
[   151.8s] [pythia-410m] single r=2 d=1 s=1
[   185.1s] [pythia-410m] single r=2 d=2 s=0
[   217.2s] [pythia-410m] single r=2 d=2 s=1
[   249.8s] [pythia-410m] single r=2 d=3 s=0
[   282.4s] [pythia-410m] single r=2 d=3 s=1
[   315.0s] [pythia-410m] single r=8 d=0 s=0
[   347.7s] [pythia-410m] single r=8 d=0 s=1
[   380.4s] [pythia-410m] single r=8 d=1 s=0
[   413.1s] [pythia-410m] single r=8 d=1 s=1
[   445.8s] [pythia-410m] single r=8 d=2 s=0
[   478.5s] [pythia-410m] single r=8 d=2 s=1
[   511.2s] [pythia-410m] single r=8 d=3 s=0
[   544.0s] [pythia-410m] single r=8 d=3 s=1
[   576.7s] [pythia-410m] single r=32 d=0 s=0
[   610.5s] [pythia-410m] single r=32 d=0 s=1
[   644.0s] [pythia-410m] single r=32 d=1 s=0
[   677.5s] [pythia-410m] single r=32 d=1 s=1
[   711.0s] [pythia-410m] single r=32 d=2 s=0
[   744.5s] [pythia-410m] single r=32 d=2 s=1
[   777.8s] [pythia-410m] single r=32 d=3 s=0
[   811.2s] [pythia-410m] single r=32 d=3 s=1
[   844.7s] [pythia-410m] joint r=2 s=0
[   976.4s] [pythia-410m] joint r=2 s=1
[  1108.2s] [pythia-410m] joint r=8 s=0
[  1240.4s] [pythia-410m] joint r=8 s=1
[  1372.4s] [pythia-410m] joint r=32 s=0
[  1507.6s] [pythia-410m] joint r=32 s=1
[  1642.8s] [pythia-410m] seq naive r=8 s=0
[  1775.6s] [pythia-410m] seq naive r=8 s=1
[  1908.3s] [pythia-410m] seq naive r=8 s=2
[  2041.5s] [pythia-410m] seq func-active r=8 s=0
[  2289.2s] [pythia-410m] seq func-active r=8 s=1
[  2537.3s] [pythia-410m] seq func-active r=8 s=2
==============================================================================
FORGETTING ATTRIBUTION -- pythia-410m
==============================================================================
Probe CE upper bound        : 1.1401 nats   (probe acc 0.766)
Joint KL to single refs     : r=2: 0.1812  r=8: 0.1889  r=32: 0.1820
Seq KL naive (r=8) : 0.2969 +/- 0.0205
Seq KL gate  (r=8) : 0.2524 +/- 0.0108
Control naive - joint      : +0.1079
Control gate  - joint      : +0.0635
Gate removed               : +0.0444
Interpretation:
- Probe CE is a certified upper bound on incompatibility, not the floor itself.
- Joint KL is the class-limited shared-model term at that rank.
- If probe CE << joint KL, the joint curve is capacity-dominated.
- Sequential minus joint is the policy/control residual.
- Gate removed is the amount of control the active-set QP recovers.
[  2797.1s] domain news: 689 chunks
[  2798.0s] domain imdb: 1643 chunks
[  2799.0s] domain wiki: 1857 chunks
[  2799.6s] domain tweets: 1052 chunks
[  2799.6s] [pythia-1b] probe upper bound
[  2814.3s] [pythia-1b] single r=2 d=0 s=0
[  2880.8s] [pythia-1b] single r=2 d=0 s=1
[  2945.8s] [pythia-1b] single r=2 d=1 s=0
[  3010.6s] [pythia-1b] single r=2 d=1 s=1
[  3075.7s] [pythia-1b] single r=2 d=2 s=0
[  3141.1s] [pythia-1b] single r=2 d=2 s=1
[  3206.3s] [pythia-1b] single r=2 d=3 s=0
[  3271.3s] [pythia-1b] single r=2 d=3 s=1
[  3336.6s] [pythia-1b] single r=8 d=0 s=0
[  3402.3s] [pythia-1b] single r=8 d=0 s=1
[  3467.9s] [pythia-1b] single r=8 d=1 s=0
[  3533.4s] [pythia-1b] single r=8 d=1 s=1
[  3599.1s] [pythia-1b] single r=8 d=2 s=0
[  3664.9s] [pythia-1b] single r=8 d=2 s=1
[  3730.7s] [pythia-1b] single r=8 d=3 s=0
[  3796.5s] [pythia-1b] single r=8 d=3 s=1
[  3862.0s] [pythia-1b] single r=32 d=0 s=0
[  3928.4s] [pythia-1b] single r=32 d=0 s=1
[  3994.9s] [pythia-1b] single r=32 d=1 s=0
[  4061.3s] [pythia-1b] single r=32 d=1 s=1
[  4128.0s] [pythia-1b] single r=32 d=2 s=0
[  4194.5s] [pythia-1b] single r=32 d=2 s=1
[  4261.0s] [pythia-1b] single r=32 d=3 s=0
[  4327.7s] [pythia-1b] single r=32 d=3 s=1
[  4394.2s] [pythia-1b] joint r=2 s=0
[  4656.4s] [pythia-1b] joint r=2 s=1
[  4919.8s] [pythia-1b] joint r=8 s=0
[  5184.4s] [pythia-1b] joint r=8 s=1
[  5449.0s] [pythia-1b] joint r=32 s=0
[  5716.9s] [pythia-1b] joint r=32 s=1
[  5984.7s] [pythia-1b] seq naive r=8 s=0
[  6250.4s] [pythia-1b] seq naive r=8 s=1
[  6516.3s] [pythia-1b] seq naive r=8 s=2
[  6782.1s] [pythia-1b] seq func-active r=8 s=0
[  7171.3s] [pythia-1b] seq func-active r=8 s=1
[  7561.5s] [pythia-1b] seq func-active r=8 s=2
==============================================================================
FORGETTING ATTRIBUTION -- pythia-1b
==============================================================================
Probe CE upper bound        : 1.9423 nats   (probe acc 0.766)
Joint KL to single refs     : r=2: 0.2315  r=8: 0.2686  r=32: 0.2734
Seq KL naive (r=8) : 0.6432 +/- 0.0094
Seq KL gate  (r=8) : 0.5175 +/- 0.0043
Control naive - joint      : +0.3747
Control gate  - joint      : +0.2489
Gate removed               : +0.1258
Interpretation:
- Probe CE is a certified upper bound on incompatibility, not the floor itself.
- Joint KL is the class-limited shared-model term at that rank.
- If probe CE << joint KL, the joint curve is capacity-dominated.
- Sequential minus joint is the policy/control residual.
- Gate removed is the amount of control the active-set QP recovers.
[  7951.9s] ALL DONE