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
#  Needs: attr_ckpts_v2/<model>/single_r8_d{0..3}_s0.pt (from the v2
#  run).  If the checkpoints are gone, set RETRAIN=True: it retrains
#  the four single-domain LoRAs (r=8, 1 seed) first (~35 min/model).
# =====================================================================
import json, os, random, time
import numpy as np
import torch
import torch.nn as nn

MODELS = ["EleutherAI/pythia-410m", "EleutherAI/pythia-1b"]
CKPT_DIR = "attr_ckpts_v2"
RETRAIN = False
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
        # per-domain per-chunk CE under each single-domain model: (T_models, T_doms, n_chunks)
        CE = [[None] * T for _ in range(T)]
        for t in range(T):
            ck = os.path.join(CKPT_DIR, key, f"single_r8_d{t}_s0.pt")
            if os.path.exists(ck):
                sd = torch.load(ck, map_location=DEV)
                # accept: list of (A,B) tuples, or a dict of tensors whose sorted
                # keys pair up as (A,B) per layer (covers the common save formats)
                if isinstance(sd, (list, tuple)):
                    st = [(a, b) for a, b in sd]
                elif isinstance(sd, dict):
                    ks = sorted(sd.keys())
                    a_keys = [k for k in ks if k.endswith("A") or ".A" in k or "lora_A" in k]
                    b_keys = [k for k in ks if k.endswith("B") or ".B" in k or "lora_B" in k]
                    if len(a_keys) == len(loras) and len(b_keys) == len(loras):
                        st = list(zip((sd[k] for k in a_keys), (sd[k] for k in b_keys)))
                    else:
                        raise ValueError(f"unrecognized ckpt format: {ck}; keys {ks[:4]}...")
                else:
                    raise ValueError(f"unrecognized ckpt type in {ck}: {type(sd)}")
                load_lora(loras, [(a.to(DEV).float(), b.to(DEV).float()) for a, b in st])
                log(f"[{key}] loaded {ck}")
            elif RETRAIN:
                log(f"[{key}] retraining single-domain d{t}")
                for l in loras:                              # reset adapters
                    with torch.no_grad():
                        l.A.zero_(); nn.init.normal_(l.B, std=1e-6)
                params = [p for l in loras for p in (l.A, l.B)]
                opt = torch.optim.AdamW(params, lr=LR, eps=1e-6, weight_decay=0.0)
                model.train(); random.seed(0)
                tr = doms[t][0]
                for _ in range(STEPS):
                    i = random.randint(0, len(tr) - BS)
                    b = tr[i:i + BS].to(DEV)
                    loss = model(b, attention_mask=torch.ones_like(b), labels=b).loss
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 0.5); opt.step()
                model.eval()
            else:
                raise FileNotFoundError(f"{ck} missing; set RETRAIN=True")
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
            c0 = os.path.join(CKPT_DIR, key, f"single_r8_d{t}_s0.pt")
            c1 = os.path.join(CKPT_DIR, key, f"single_r8_d{t}_s1.pt")
            if os.path.exists(c0) and os.path.exists(c1):
                st0 = read_ckpt(c0, len(loras), DEV)
                st1 = read_ckpt(c1, len(loras), DEV)
                anchors = doms[t][2]
                base_kls.append(token_kl_between(model, loras, st0, st1, anchors, DEV))
        if base_kls:
            results[key]["seed_baseline_kl"] = float(np.mean(base_kls))
            log(f"[{key}] seed-to-seed KL baseline (same domain, s0 vs s1): "
                f"{np.mean(base_kls):.4f} nats/token -- subtract from joint/seq KL "
                f"terms to net out the two-models offset")
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    with open("floor_certificate_results.json", "w") as f:
        json.dump(results, f, indent=1)
    log("ALL DONE -> floor_certificate_results.json")
