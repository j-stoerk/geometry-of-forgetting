# =====================================================================
#  kaggle_floor_certificate.py -- the missing FLOOR CERTIFICATE for the
#  attribution experiment.  ~15 min on 1x T4 per model.
#
#  The information floor of a single model over T domains is
#  T * I(T ; Y | X), and ANY domain classifier upper-bounds it:
#      CE_classifier  >=  H(T | X)  >=  I(T ; Y | X).
#  The v2 linear probe on mean-pooled features was too weak (CE ~ ln 4,
#  vacuous).  This script builds the GENERATIVE classifier from the
#  saved single-domain LoRA checkpoints: for a held-out chunk x,
#      p(T=t | x)  propto  p(t) * exp( -len(x) * CE_t(x) ),
#  where CE_t is the per-token loss of the domain-t model.  Over a
#  128-token chunk the likelihood ratios are decisive, so this is
#  expected to certify floor <= a few 1e-2 nats/token -- turning
#  "avoidable in principle" into a measured statement.
#
#  Also measures the SEED-TO-SEED KL BASELINE (same-domain s0-vs-s1
#  adapters): two independently trained models differ in KL with zero
#  forgetting, so this offset is subtracted from the joint/sequential
#  KL terms of the attribution.
#
#  Checkpoints: looks in ./attr_ckpts_v2/ AND in any attached Kaggle
#  input (attach the v2 notebook's output as a dataset to reuse its
#  ckpts).  Anything missing is RETRAINED AUTOMATICALLY (~3 min per
#  adapter at 410m, ~7 min at 1b; 8 adapters per model worst case)
#  and cached, so the script runs from scratch in a fresh session.
# =====================================================================
import json, os, random, time
import numpy as np
import torch
import torch.nn as nn

MODELS = ["EleutherAI/pythia-410m", "EleutherAI/pythia-1b"]
CKPT_DIR = "attr_ckpts_v2"
RANK, STEPS, BS, SEQ, LR, CACHE_CH = 8, 250, 8, 128, 1e-4, 6
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


class LoRA(nn.Module):
    def __init__(self, base, d_in, d_out, r, device):
        super().__init__()
        self.base = base; self.s = 2.0 / r
        self.A = nn.Parameter(torch.zeros(d_in, r, device=device))
        self.B = nn.Parameter(torch.empty(r, d_out, device=device))
        nn.init.normal_(self.B, std=1e-6)
    def forward(self, x):
        base = self.base(x)
        lora = (x.to(self.A.dtype) @ self.A) @ self.B * self.s
        return base + lora.to(base.dtype)


def build_model(name, device, rank=RANK):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(device)
    if getattr(m.config, "use_cache", None) is not None: m.config.use_cache = False
    with torch.no_grad():
        for p in m.parameters():
            if p.is_floating_point() and p.dtype != torch.float32: p.data = p.data.float()
    for p in m.parameters(): p.requires_grad_(False)
    loras = []
    din, dout = m.config.hidden_size, 3 * m.config.hidden_size
    for b in m.gpt_neox.layers:
        l = LoRA(b.attention.query_key_value, din, dout, rank, device)
        b.attention.query_key_value = l; loras.append(l)
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
    for name, t in texts:
        enc = tok(t, return_tensors="pt").input_ids[0]
        ch = enc[: (len(enc) // SEQ) * SEQ].view(-1, SEQ)
        tr = ch[:110]; te = ch[110:134]; an = ch[134:142]   # v2 splits: 110/24/8
        out.append((tr, te, an))
    return out


@torch.no_grad()
def chunk_ce(model, ids, device):
    """Per-chunk mean CE (nats/token), one value per chunk row."""
    out = []
    for i in range(0, len(ids), 4):
        b = ids[i:i + 4].to(device)
        lo = model(b, attention_mask=torch.ones_like(b)).logits[:, :-1]
        ce = nn.functional.cross_entropy(
            lo.reshape(-1, lo.shape[-1]).float(), b[:, 1:].reshape(-1),
            reduction="none").view(len(b), -1).mean(1)
        out.append(ce.cpu())
    return torch.cat(out)


def lora_state(loras): return [(l.A.detach().clone(), l.B.detach().clone()) for l in loras]
def load_lora(loras, st):
    with torch.no_grad():
        for l, (A, B) in zip(loras, st): l.A.copy_(A); l.B.copy_(B)


def read_ckpt(path, n_layers, device):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, (list, tuple)):
        st = list(sd)
    else:
        st = [(sd[f"{i}.A"], sd[f"{i}.B"]) for i in range(n_layers)]
    return [(a.to(device).float(), b.to(device).float()) for a, b in st]


@torch.no_grad()
def token_kl_between(model, loras, st_old, st_new, chunks, device):
    """Mean per-token KL(p_old || p_new) on chunks, swapping adapters per batch."""
    tot, ntok = 0.0, 0
    for i in range(0, len(chunks), 2):
        b = chunks[i:i + 2].to(device)
        load_lora(loras, st_old)
        lp0 = nn.functional.log_softmax(
            model(b, attention_mask=torch.ones_like(b)).logits[:, :-1].float(), -1)
        load_lora(loras, st_new)
        lp1 = nn.functional.log_softmax(
            model(b, attention_mask=torch.ones_like(b)).logits[:, :-1].float(), -1)
        tot += float((lp0.exp() * (lp0 - lp1)).sum())
        ntok += int(lp0.shape[0] * lp0.shape[1])
    return tot / max(ntok, 1)


def find_ckpt(key, fname):
    """Look for a saved adapter in the local dir and in any attached Kaggle
    dataset (attach the v2 notebook's output as an input to reuse its ckpts)."""
    import glob
    cands = [os.path.join(CKPT_DIR, key, fname)]
    cands += glob.glob(f"/kaggle/input/*/attr_ckpts_v2/{key}/{fname}")
    cands += glob.glob(f"/kaggle/input/*/{key}/{fname}")
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def get_single_state(model, loras, doms, key, t, seed):
    """Adapter state for single-domain t, seed: load if found, else RETRAIN
    (~3 min at 410m, ~7 min at 1b) and cache locally."""
    fname = f"single_r{RANK}_d{t}_s{seed}.pt"
    ck = find_ckpt(key, fname)
    if ck is not None:
        st = read_ckpt(ck, len(loras), DEV)
        log(f"[{key}] loaded {ck}")
        return st
    log(f"[{key}] ckpt {fname} not found -- retraining (seed {seed})")
    random.seed(seed); torch.manual_seed(seed)
    with torch.no_grad():
        for l in loras:
            l.A.zero_(); nn.init.normal_(l.B, std=1e-6)
    params = [p for l in loras for p in (l.A, l.B)]
    opt = torch.optim.AdamW(params, lr=LR, eps=1e-6, weight_decay=0.0)
    model.train()
    tr = doms[t][0]
    for _ in range(STEPS):
        i = random.randint(0, len(tr) - BS)
        b = tr[i:i + BS].to(DEV)
        loss = model(b, attention_mask=torch.ones_like(b), labels=b).loss
        if not torch.isfinite(loss): raise ValueError("non-finite loss in retrain")
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 0.5); opt.step()
    model.eval()
    st = lora_state(loras)
    out = os.path.join(CKPT_DIR, key, fname)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save([(a.cpu(), b.cpu()) for a, b in st], out)
    return st


if __name__ == "__main__":
    from transformers import AutoTokenizer
    results = {}
    for model_name in MODELS:
        key = model_name.split("/")[-1]
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token
        doms = get_domains(tok)
        model, loras = build_model(model_name, DEV)
        model.eval()
        T = len(doms)
        # single-domain adapter states, seed 0 (load or retrain)
        states0 = [get_single_state(model, loras, doms, key, t, 0) for t in range(T)]
        # per-domain per-chunk CE under each single-domain model
        CE = [[None] * T for _ in range(T)]
        for t in range(T):
            load_lora(loras, states0[t])
            for dom in range(T):
                CE[t][dom] = chunk_ce(model, doms[dom][1], DEV)
        # generative classifier: p(T=t|x) propto exp(-SEQ * CE_t(x)) (uniform prior)
        ces, labels = [], []
        for dom in range(T):
            n = len(CE[0][dom])
            ll = torch.stack([-SEQ * CE[t][dom] for t in range(T)], 1)   # (n, T)
            logp = ll - torch.logsumexp(ll, dim=1, keepdim=True)
            ces.append(-logp[:, dom])                       # per-chunk classifier CE
            labels.append((logp.argmax(1) == dom).float())
        ce_all = torch.cat(ces); acc = torch.cat(labels).mean()
        # certified chain: classifier CE >= H(T|X) >= I(T;Y|X) = per-token floor
        results[key] = dict(classifier_ce=float(ce_all.mean()),
                            classifier_acc=float(acc),
                            ce_90pct=float(ce_all.quantile(0.9)))
        log(f"[{key}] FLOOR CERTIFICATE: I(T;Y|X) <= H(T|X) <= {ce_all.mean():.4f} "
            f"nats/token  (classifier acc {acc:.3f}; 90th pct {ce_all.quantile(0.9):.4f})")
        # ---- seed-to-seed KL baseline: two independently trained models of the
        # SAME domain differ in KL with zero forgetting -- this offset should be
        # subtracted from the joint/sequential KL terms of the attribution.
        base_kls = []
        for t in range(T):
            st1 = get_single_state(model, loras, doms, key, t, 1)
            anchors = doms[t][2]
            base_kls.append(token_kl_between(model, loras, states0[t], st1, anchors, DEV))
        results[key]["seed_baseline_kl"] = float(np.mean(base_kls))
        results[key]["seed_baseline_per_domain"] = [float(x) for x in base_kls]
        log(f"[{key}] seed-to-seed KL baseline (same domain, s0 vs s1): "
            f"{np.mean(base_kls):.4f} nats/token -- subtract from joint/seq KL "
            f"terms to net out the two-models offset")
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    with open("floor_certificate_results.json", "w") as f:
        json.dump(results, f, indent=1)
    log("ALL DONE -> floor_certificate_results.json")
