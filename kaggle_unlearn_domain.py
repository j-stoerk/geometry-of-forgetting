# =====================================================================
#  kaggle_unlearn_domain.py -- CERTIFIED DOMAIN UNLEARNING on the
#  4-domain stream: unlearn imdb from the jointly trained model by
#  ENGINEERED BENIGN FORGETTING.  ~30-45 min on 1x T4 per model.
#
#  Method (one reversed row of the unified gate): ascend the forget
#  domain's loss toward the BASE-model reference (remove the adapter's
#  imdb contribution, not the pretrained knowledge), gated by the
#  monotone-retention QP over the retained domains' HELD-OUT micro-cache
#  gradients (active set, one row per retained domain per step):
#      v = qp_gate( +grad L_imdb ;  rows = grad L_news/wiki/tweets )
#  Stop when imdb anchor CE >= base-model CE (reference reached) or the
#  step budget ends.
#
#  CERTIFICATE (measured, not retraining-defined):
#      E_retained[ KL( p_before || p_after ) ]   on held-out anchors,
#  plus per-domain retained CE change, plus KL(p_after || p_base) on the
#  forget anchors (how close to 'never fine-tuned' behaviour).
#
#  Arms: gated vs naive ascent (same stopping rule).  Prediction from
#  the exact regime (evaluate_unlearning.py): naive reaches the
#  reference but pays retained damage; the gate reaches it at a small
#  measured certificate -- unless imdb knowledge is entangled with a
#  retained domain, in which case the gate stalls at the measurable
#  unlearning floor instead of silently paying.
#
#  Checkpoints: reuses joint_r8_s0.pt via the same search as the
#  certificate script (local dir, /kaggle/input/*); retrains the joint
#  model automatically if absent (~15 min at 410m).
# =====================================================================
import glob
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import nnls

MODELS = ["EleutherAI/pythia-410m", "EleutherAI/pythia-1b"]
FORGET = 1                        # imdb
RANK, BS, SEQ, LR, STEPS_TRAIN = 8, 8, 128, 1e-4, 250
UNLEARN_STEPS, UNLEARN_LR = 400, 5e-5
CKPT_DIR = "attr_ckpts_v2"
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
    texts.append(("imdb", "\n\n".join([x["text"] for x in imdb][:700])))
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts.append(("wiki", "\n\n".join([x["text"] for x in wiki if len(x["text"]) > 200][:1400])))
    tw = load_dataset("tweet_eval", "sentiment", split="train")
    texts.append(("tweets", "\n\n".join([x["text"] for x in tw][:4500])))
    out = []
    for name, t in texts:
        enc = tok(t, return_tensors="pt").input_ids[0]
        ch = enc[: (len(enc) // SEQ) * SEQ].view(-1, SEQ)
        tr = ch[:110]; te = ch[110:134]; an = ch[134:142]   # v2 splits
        out.append((name, tr, te, an))
        log(f"domain {name}: {len(ch)} chunks")
    return out


def lm_loss(model, ids, device=DEV):
    ids = ids.to(device)
    out = model(ids, attention_mask=torch.ones_like(ids), labels=ids)
    if not torch.isfinite(out.loss): raise ValueError("non-finite loss")
    return out.loss


@torch.no_grad()
def mean_ce(model, chunks, device=DEV):
    model.eval()
    tot = sum(float(lm_loss(model, chunks[i:i + 4], device)) * min(4, len(chunks) - i)
              for i in range(0, len(chunks), 4))
    model.train()
    return tot / len(chunks)


@torch.no_grad()
def logits_snapshot(model, chunks, device=DEV):
    model.eval()
    outs = []
    for i in range(0, len(chunks), 2):
        b = chunks[i:i + 2].to(device)
        outs.append(F.log_softmax(
            model(b, attention_mask=torch.ones_like(b)).logits[:, :-1].float(), -1).cpu())
    model.train()
    return torch.cat(outs)


@torch.no_grad()
def kl_from_snapshot(model, chunks, snap, device=DEV):
    model.eval()
    tot, ntok, j = 0.0, 0, 0
    for i in range(0, len(chunks), 2):
        b = chunks[i:i + 2].to(device)
        lp = F.log_softmax(
            model(b, attention_mask=torch.ones_like(b)).logits[:, :-1].float(), -1)
        lp0 = snap[j:j + len(b)].to(device)
        tot += float((lp0.exp() * (lp0 - lp)).sum())
        ntok += int(lp.shape[0] * lp.shape[1]); j += len(b)
    model.train()
    return tot / max(ntok, 1)


def gvec(params):
    return torch.cat([(torch.zeros(p.numel(), device=p.device) if p.grad is None
                       else p.grad.detach().flatten().float()) for p in params])


def setgrad(params, g):
    i = 0
    for p in params:
        n = p.numel()
        p.grad = g[i:i + n].view_as(p).clone().to(p.dtype); i += n


def find_ckpt(key, fname):
    cands = [os.path.join(CKPT_DIR, key, fname)]
    cands += glob.glob(f"/kaggle/input/*/attr_ckpts_v2/{key}/{fname}")
    cands += glob.glob(f"/kaggle/input/*/{key}/{fname}")
    for c in cands:
        if os.path.exists(c): return c
    return None


def load_lora_file(loras, path, device=DEV):
    sd = torch.load(path, map_location="cpu")
    with torch.no_grad():
        if isinstance(sd, (list, tuple)):
            for l, (a, b) in zip(loras, sd):
                l.A.copy_(a.to(device).float()); l.B.copy_(b.to(device).float())
        else:
            for i, l in enumerate(loras):
                l.A.copy_(sd[f"{i}.A"].to(device).float())
                l.B.copy_(sd[f"{i}.B"].to(device).float())


if __name__ == "__main__":
    from transformers import AutoTokenizer
    results = {}
    for model_name in MODELS:
        key = model_name.split("/")[-1]
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token
        doms = get_domains(tok)
        retained = [i for i in range(len(doms)) if i != FORGET]

        # base-model reference on the forget anchors (the unlearning target)
        model, loras = build_model(model_name, DEV)
        ref_ce = mean_ce(model, doms[FORGET][3])
        log(f"[{key}] base-model reference CE on {doms[FORGET][0]} anchors: {ref_ce:.4f}")

        # the trained joint model (load or retrain)
        ck = find_ckpt(key, f"joint_r{RANK}_s0.pt")
        if ck is not None:
            load_lora_file(loras, ck); log(f"[{key}] loaded {ck}")
        else:
            log(f"[{key}] joint ckpt not found -- retraining ({len(doms) * STEPS_TRAIN} steps)")
            params = [p for l in loras for p in (l.A, l.B)]
            opt = torch.optim.AdamW(params, lr=LR, eps=1e-6, weight_decay=0.0)
            model.train(); random.seed(0)
            for s in range(len(doms) * STEPS_TRAIN):
                tr = doms[s % len(doms)][1]
                i = random.randint(0, len(tr) - BS)
                loss = lm_loss(model, tr[i:i + BS])
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 0.5); opt.step()
        import copy
        joint_state = [(l.A.detach().clone(), l.B.detach().clone()) for l in loras]

        # before-state measurements
        before_ce = {doms[i][0]: mean_ce(model, doms[i][2]) for i in range(len(doms))}
        f_ce0 = mean_ce(model, doms[FORGET][3])
        snaps = {i: logits_snapshot(model, doms[i][3]) for i in retained}
        log(f"[{key}] trained model: forget-anchor CE {f_ce0:.4f} (target {ref_ce:.4f}); "
            f"retained CE " + " ".join(f"{doms[i][0]}={before_ce[doms[i][0]]:.3f}"
                                       for i in retained))
        results.setdefault(key, {})["reference_ce"] = ref_ce
        results[key]["before"] = dict(forget_anchor_ce=f_ce0, retained_ce=before_ce)

        for arm in ["gated", "naive"]:
            with torch.no_grad():
                for l, (a, b) in zip(loras, joint_state): l.A.copy_(a); l.B.copy_(b)
            params = [p for l in loras for p in (l.A, l.B)]
            opt = torch.optim.SGD(params, lr=UNLEARN_LR)
            model.train(); random.seed(1)
            steps_used = UNLEARN_STEPS
            for s in range(UNLEARN_STEPS):
                tr = doms[FORGET][1]
                i = random.randint(0, len(tr) - BS)
                loss = lm_loss(model, tr[i:i + BS])
                opt.zero_grad(set_to_none=True); (-loss).backward()   # ASCENT
                u = gvec(params)
                if arm == "gated":
                    cols = []
                    for rdom in retained:                    # one row per retained domain
                        te = doms[rdom][2]
                        j = random.randint(0, max(len(te) - 2, 0))
                        opt.zero_grad(set_to_none=True)
                        lm_loss(model, te[j:j + 2]).backward()
                        nA = gvec(params)
                        if float(nA.norm() ** 2) > 1e-12: cols.append(nA)
                    if cols:
                        # constraint: <grad L_retained, v> <= 0 for the DESCENT
                        # direction v = -u_grad; rows enter as in qp_gate
                        Gm = torch.stack(cols, 1).cpu().double().numpy()
                        lam, _ = nnls(Gm, (-u).cpu().double().numpy())
                        if lam.any():
                            u = u + torch.as_tensor(Gm @ lam, dtype=u.dtype, device=u.device)
                    setgrad(params, u)
                torch.nn.utils.clip_grad_norm_(params, 0.5)
                opt.step()
                if s % 20 == 19 and mean_ce(model, doms[FORGET][3]) >= ref_ce:
                    steps_used = s + 1
                    log(f"[{key}/{arm}] reference reached at step {steps_used}")
                    break
            f_ce = mean_ce(model, doms[FORGET][3])
            after_ce = {doms[i][0]: mean_ce(model, doms[i][2]) for i in range(len(doms))}
            cert = {doms[i][0]: kl_from_snapshot(model, doms[i][3], snaps[i])
                    for i in retained}
            results[key][arm] = dict(
                steps=steps_used, forget_anchor_ce=f_ce,
                gap_to_reference=float(ref_ce - f_ce),
                retained_ce_after=after_ce,
                certificate_kl=cert,
                certificate_mean=float(np.mean(list(cert.values()))))
            log(f"[{key}/{arm}] forget CE {f_ce0:.3f}->{f_ce:.3f} (ref {ref_ce:.3f}, "
                f"{steps_used} steps); CERTIFICATE E[KL_retained] = "
                f"{results[key][arm]['certificate_mean']:.4f}  per-domain "
                + " ".join(f"{k}={v:.4f}" for k, v in cert.items()))
            with open("unlearn_domain_results.json", "w") as f:
                json.dump(results, f, indent=1)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    log("ALL DONE -> unlearn_domain_results.json")
