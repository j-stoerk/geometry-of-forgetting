# =====================================================================
#  kaggle_forgetting_attribution.py -- ATTRIBUTED FORGETTING CURVES:
#  the forgetting decomposition (L = floor + capacity + control),
#  measured on real LLM continual pretraining, per model scale.
#
#  For each model (pythia-410m, pythia-1b; optional 1.4b) on the
#  4-domain stream news->imdb->wiki->tweets:
#
#  FLOOR   certified upper bound, not assumed: a domain probe on frozen
#          base-model features gives, by data processing,
#             CE_probe >= H(T | phi(X)) >= H(T | X) >= I(T ; Y | X),
#          and I(T;Y|X) IS the per-token irreducible single-model floor
#          (paper, Eq. infofloor).  Expect ~0 for disjoint corpora ->
#          "this fraction of observed forgetting was avoidable in
#          principle" becomes a measured, certified statement.
#  CAPACITY(r)  joint (interleaved) training at the SAME LoRA rank and
#          the SAME per-domain step budget: its final per-domain excess
#          over the per-domain reference is what the deployed class
#          cannot fit simultaneously.  Swept over rank r in {2, 8, 32}.
#          (Joint training upper-bounds the class optimum, so this
#          over-estimates capacity and the control residual is
#          conservative -- stated honestly in the output.)
#  CONTROL sequential-minus-joint residual at the deployed rank: what
#          the training POLICY loses relative to the class optimum.
#          The active-set gate (QP corollary) is then run to measure
#          how much of the control term one policy removes.
#
#  Output: per model, the attributed table
#     forgetting(naive) = floor(<=bound) + capacity(r=8) + control,
#  the rank sweep capacity(2/8/32), and the gate's control recovery --
#  the first forgetting curves that say WHICH PART WAS EVER AVOIDABLE.
#
#  Runtime (2x T4, fp32): ~2h for 410m, ~4h for 1b.  Per-seed resume:
#  progress is checkpointed to forgetting_attribution_results.json
#  after every run; restart the cell to continue.
# =====================================================================
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from scipy.optimize import nnls

SCRIPT_VERSION = "attr-v1.0"

MODELS = ["EleutherAI/pythia-410m", "EleutherAI/pythia-1b"]   # add "EleutherAI/pythia-1.4b"
SEEDS_SEQ = [0, 1, 2]          # sequential arms (naive, func-active)
SEEDS_JOINT = [0, 1]           # joint arms (per rank)
RANKS_JOINT = [2, 8, 32]       # capacity sweep
RANK_DEPLOY = 8                # deployed rank for the attribution row
STEPS = 250                    # per domain (sequential) / per domain-equivalent (joint)
BS, SEQ = 8, 128
CACHE_CH = 6
LR = 1e-4                      # validated: real forgetting at 410m (naive ~0.32)
FORGET_MIN = 0.03              # pre-flight guard
ADAM_EPS = 1e-6
MAX_GRAD_NORM = 0.5
RESULTS_FILE = "forgetting_attribution_results.json"

DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
DEV = DEVICES[0]
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


class LoRA(nn.Module):
    """Dtype-agnostic LoRA (fp32 adapter params, cast around base dtype)."""
    def __init__(self, base, d_in, d_out, r, device):
        super().__init__()
        self.base = base
        self.s = 2.0 / r
        self.A = nn.Parameter(torch.zeros(d_in, r, device=device))
        self.B = nn.Parameter(torch.empty(r, d_out, device=device))
        nn.init.normal_(self.B, std=1e-6)

    def forward(self, x):
        base = self.base(x)
        lora = (x.to(self.A.dtype) @ self.A) @ self.B * self.s
        return base + lora.to(base.dtype)


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(name, device, rank):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(device)
    if getattr(m.config, "use_cache", None) is not None:
        m.config.use_cache = False
    with torch.no_grad():                                   # force fp32 in place
        for p in m.parameters():
            if p.is_floating_point() and p.dtype != torch.float32: p.data = p.data.float()
        for b in m.buffers():
            if b.is_floating_point() and b.dtype != torch.float32: b.data = b.data.float()
    for p in m.parameters():
        p.requires_grad_(False)
    loras = []
    blocks = m.gpt_neox.layers
    din, dout = m.config.hidden_size, 3 * m.config.hidden_size
    for b in blocks:
        l = LoRA(b.attention.query_key_value, din, dout, rank, device)
        b.attention.query_key_value = l
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
            raise ValueError(f"domain {name}: {len(ch)} chunks < {need}")
        out.append((ch[:110], ch[110:130], ch[130:130 + CACHE_CH]))
    return out


def lm_loss(model, ids, device):
    ids = ids.to(device)
    out = model(ids, attention_mask=torch.ones_like(ids), labels=ids)
    if not torch.isfinite(out.loss):
        raise ValueError("non-finite lm loss")
    return out.loss


@torch.no_grad()
def heldout(model, te, device):
    model.eval()
    tot = sum(float(lm_loss(model, te[i:i + 4], device)) * min(4, len(te) - i)
              for i in range(0, len(te), 4))
    model.train()
    return tot / len(te)


# ------------------------------------------------------------ FLOOR bound
@torch.no_grad()
def floor_bound(model_name, doms, device):
    """Certified floor bound: CE of a linear domain probe on frozen base
    features >= H(T|phi(X)) >= H(T|X) >= I(T; Y | X) = the per-token floor."""
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    m.eval()
    feats, labs = {"tr": [], "te": []}, {"tr": [], "te": []}
    for t, (tr, te, _) in enumerate(doms):
        for split, ch in [("tr", tr), ("te", te)]:
            for i in range(0, len(ch), 8):
                ids = ch[i:i + 8].to(device)
                h = m(ids, attention_mask=torch.ones_like(ids),
                      output_hidden_states=True).hidden_states[-1]
                feats[split].append(h.mean(1).float().cpu())
                labs[split].append(torch.full((len(ids),), t))
    del m
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    Xtr = torch.cat(feats["tr"]); ytr = torch.cat(labs["tr"])
    Xte = torch.cat(feats["te"]); yte = torch.cat(labs["te"])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    W = torch.zeros(Xtr.shape[1], len(doms), requires_grad=True)
    opt = torch.optim.Adam([W], lr=1e-2)
    for _ in range(400):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(Xtr @ W, ytr) + 1e-3 * W.square().sum()
        loss.backward(); opt.step()
    with torch.no_grad():
        logits = Xte @ W
        ce = float(nn.functional.cross_entropy(logits, yte))     # nats: >= H(T|X)
        acc = float((logits.argmax(1) == yte).float().mean())
    return ce, acc


# --------------------------------------------------- sequential / joint runs
def make_opt(loras):
    params = [p for l in loras for p in (l.A, l.B)]
    return params, torch.optim.AdamW(params, lr=LR, eps=ADAM_EPS, weight_decay=0.0)


def gvec(params):
    return torch.cat([(torch.zeros(p.numel(), device=p.device) if p.grad is None
                       else p.grad.detach().flatten().float()) for p in params])


def setgrad(params, g):
    i = 0
    for p in params:
        n = p.numel()
        p.grad = g[i:i + n].view_as(p).clone().to(p.dtype)
        i += n


def train_step(model, params, opt, ids, device):
    loss = lm_loss(model, ids, device)
    opt.zero_grad(set_to_none=True)
    loss.backward()


def run_sequential(seed, method, rank, doms, model_name, device):
    """naive or func-active; returns dict(final=[per-domain], diag=[per-domain])."""
    set_all_seeds(seed)
    model, loras = build_model(model_name, device, rank)
    model.train()
    params, opt = make_opt(loras)
    T = len(doms)
    held = np.full((T, T), np.nan)
    for t, (tr, te, cache) in enumerate(doms):
        for _ in range(STEPS):
            i = random.randint(0, len(tr) - BS)
            train_step(model, params, opt, tr[i:i + BS], device)
            if method == "func-active" and t > 0:
                g = gvec(params)
                cols = []
                for u in range(t):
                    cu = doms[u][2]
                    j = random.randint(0, max(len(cu) - 2, 0))
                    opt.zero_grad(set_to_none=True)
                    lm_loss(model, cu[j:j + 2], device).backward()
                    nA = gvec(params)
                    if float(nA.norm() ** 2) > 1e-12: cols.append(nA)
                if cols:
                    Gm = torch.stack(cols, 1).cpu().double().numpy()
                    lam, _ = nnls(Gm, (-g).cpu().double().numpy())
                    if lam.any():
                        g = g + torch.as_tensor(Gm @ lam, dtype=g.dtype, device=g.device)
                setgrad(params, g)
            tn = torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
            if not torch.isfinite(tn): raise ValueError("non-finite grad")
            opt.step()
        for i in range(t + 1):
            held[i, t] = heldout(model, doms[i][1], device)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return dict(final=[float(held[i, T - 1]) for i in range(T)],
                diag=[float(held[i, i]) for i in range(T)])


def run_joint(seed, rank, doms, model_name, device):
    """Interleaved training, same per-domain budget: T*STEPS total steps."""
    set_all_seeds(seed)
    model, loras = build_model(model_name, device, rank)
    model.train()
    params, opt = make_opt(loras)
    T = len(doms)
    for s in range(T * STEPS):
        t = s % T                                           # strict interleave
        tr = doms[t][0]
        i = random.randint(0, len(tr) - BS)
        train_step(model, params, opt, tr[i:i + BS], device)
        tn = torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
        if not torch.isfinite(tn): raise ValueError("non-finite grad")
        opt.step()
    final = [heldout(model, doms[i][1], device) for i in range(T)]
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return dict(final=[float(x) for x in final])


# ------------------------------------------------------------- attribution
def attribute(results, model_name):
    """Assemble the attributed table for one model from raw runs."""
    key = model_name.split("/")[-1]
    R = results.get(key, {})
    need = [f"seq/naive/r{RANK_DEPLOY}", f"seq/func-active/r{RANK_DEPLOY}"] + \
           [f"joint/r{r}" for r in RANKS_JOINT]
    if any(k not in R or not R[k] for k in need) or "floor" not in R:
        return None
    T = len(R[f"seq/naive/r{RANK_DEPLOY}"][next(iter(R[f'seq/naive/r{RANK_DEPLOY}']))]["final"])
    def seq_stats(arm):
        runs = list(R[arm].values())
        fgt = [np.mean([r["final"][i] - r["diag"][i] for i in range(T - 1)]) for r in runs]
        exc = [np.mean([r["final"][i] - r["diag"][i] for i in range(T - 1)]) for r in runs]
        return np.mean(fgt), np.std(fgt), runs
    fgt_naive, sd_naive, runs_n = seq_stats(f"seq/naive/r{RANK_DEPLOY}")
    fgt_gate, sd_gate, _ = seq_stats(f"seq/func-active/r{RANK_DEPLOY}")
    diag_ref = np.mean([[r["diag"][i] for i in range(T - 1)]
                        for r in runs_n], axis=0)           # per-domain oracle proxy
    cap = {}
    for r in RANKS_JOINT:
        joints = list(R[f"joint/r{r}"].values())
        cap[r] = float(np.mean([np.mean([j["final"][i] - diag_ref[i]
                                for i in range(T - 1)]) for j in joints]))
    floor_ce, floor_acc = R["floor"]
    control_naive = fgt_naive - cap[RANK_DEPLOY]
    control_gate = fgt_gate - cap[RANK_DEPLOY]
    return dict(model=key, floor_bound=floor_ce, probe_acc=floor_acc,
                forgetting_naive=fgt_naive, sd_naive=sd_naive,
                forgetting_gate=fgt_gate, sd_gate=sd_gate,
                capacity=cap, control_naive=control_naive, control_gate=control_gate)


def report(att):
    if att is None: return
    print("\n" + "=" * 72)
    print(f"ATTRIBUTED FORGETTING -- {att['model']}   (nats/token; mean over earlier domains)")
    print("=" * 72)
    fb, fn = att["floor_bound"], att["forgetting_naive"]
    print(f"  observed forgetting (naive, r={RANK_DEPLOY})  : {fn:.3f} +/- {att['sd_naive']:.3f}")
    print(f"  FLOOR   (certified bound via domain probe)    : <= {fb:.4f}"
          f"   (probe acc {att['probe_acc']:.3f})")
    print(f"  CAPACITY(r)  [joint-training excess]          : "
          + "  ".join(f"r={r}: {att['capacity'][r]:+.3f}" for r in RANKS_JOINT))
    print(f"  CONTROL (sequential - joint, r={RANK_DEPLOY})          : {att['control_naive']:+.3f}"
          f"   -> gate leaves {att['control_gate']:+.3f}"
          f"  ({100 * (1 - att['control_gate'] / max(att['control_naive'], 1e-9)):.0f}% of control removed)")
    print(f"  avoidable in principle (1 - floor/forgetting) : "
          f">= {100 * (1 - fb / max(fn, 1e-9)):.0f}%")
    print("  (capacity uses joint training as the class-optimum surrogate, so it is an")
    print("   over-estimate and the control residual is conservative.)")


# --------------------------------------------------------------------- main
if __name__ == "__main__":
    log(f"SCRIPT VERSION {SCRIPT_VERSION}; devices = {DEVICES}")
    results = {}
    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        log(f"RESUME: loaded {sum(len(v) if isinstance(v, dict) else 1 for v in results.values())} entries")

    def save():
        with open(RESULTS_FILE, "w") as f: json.dump(results, f, indent=1)

    from transformers import AutoTokenizer
    for model_name in MODELS:
        key = model_name.split("/")[-1]
        results.setdefault(key, {})
        R = results[key]
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token
        doms = get_domains(tok)

        if "floor" not in R:
            log(f"[{key}] measuring the certified floor bound (domain probe)")
            R["floor"] = floor_bound(model_name, doms, DEV); save()
            log(f"[{key}] floor <= {R['floor'][0]:.4f} nats (probe acc {R['floor'][1]:.3f})")

        for arm, seeds in [("naive", SEEDS_SEQ), ("func-active", SEEDS_SEQ)]:
            k = f"seq/{arm}/r{RANK_DEPLOY}"
            R.setdefault(k, {})
            for s in seeds:
                if str(s) in R[k]: continue
                log(f"[{key}] sequential {arm} r={RANK_DEPLOY} seed {s}")
                R[k][str(s)] = run_sequential(s, arm, RANK_DEPLOY, doms, model_name, DEV)
                save()
            fgt = np.mean([[r["final"][i] - r["diag"][i] for i in range(len(doms) - 1)]
                           for r in R[k].values()])
            log(f"[{key}] {k}: mean forgetting {fgt:.3f}")
            if arm == "naive" and abs(fgt) < FORGET_MIN:
                log(f"** WARNING [{key}]: naive forgetting {fgt:.3f} < {FORGET_MIN} -- "
                    f"stream barely forgets at this scale/LR; attribution will be vacuous **")

        for r in RANKS_JOINT:
            k = f"joint/r{r}"
            R.setdefault(k, {})
            for s in SEEDS_JOINT:
                if str(s) in R[k]: continue
                log(f"[{key}] joint r={r} seed {s}")
                R[k][str(s)] = run_joint(s, r, doms, model_name, DEV)
                save()

        report(attribute(results, model_name))

    log("ALL DONE -> " + RESULTS_FILE)
    for model_name in MODELS:
        report(attribute(results, model_name))
