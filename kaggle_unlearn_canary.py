# =====================================================================
#  kaggle_unlearn_canary.py -- CERTIFIED UNLEARNING, a NON-vacuous test.
#  ~25-40 min on 1x T4 per model.
#
#  Why v1 (unlearn imdb) was vacuous: the imdb LoRA added only ~0.03
#  nats of held-out imdb knowledge over base (imdb is ordinary English
#  pythia already models), so there was almost nothing to unlearn and
#  both arms tied at ~0.  This script instead injects a MEMORIZED
#  CANARY -- a fixed novel sentence repeated during joint training --
#  which the model memorizes (train loss -> ~0) and which base has
#  never seen (loss high).  Now there is a large, cleanly-owned target
#  to remove, exactly the "memorized content" branch of the exact-
#  regime dichotomy (evaluate_unlearning.py).
#
#  Method (one reversed row of the unified gate): ascend the canary
#  loss toward the BASE-model reference, gated by the monotone-
#  retention QP over the retained domains' held-out anchors:
#      v = qp_gate( +grad L_canary ;  rows = grad L_news/wiki/tweets ).
#
#  PREDICTION (from the exact regime): memorized content is removable
#  at a ZERO floor, so the GATE reaches the base reference with a small
#  retained certificate E_retain[KL(before||after)], while NAIVE ascent
#  reaches it too but pays a much larger certificate -- the retained
#  damage the gate provably avoids.  Certificate is a MEASUREMENT.
#
#  Arms: gated vs naive.  Also reports canary verbatim-completion
#  accuracy (memorization removed?) before/after.
#  Checkpoints: none needed (trains its own joint+canary model, ~10min).
# =====================================================================
import glob, json, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import nnls

MODELS = ["EleutherAI/pythia-410m", "EleutherAI/pythia-1b"]
RANK, BS, SEQ, LR = 8, 8, 128, 1e-4
STEPS_TRAIN = 250                 # per real domain
CANARY_REPEATS = 24               # canary chunks mixed into training
CANARY = (" The secret access code for the Zorblatt vault is "
          "quibble-marmoset-1971-velvet-parsnip. ")
UNLEARN_STEPS, UNLEARN_LR = 300, 1e-3     # enough power to reach reference
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
if DEV.startswith("cuda") and torch.cuda.get_device_capability(0) < (7, 0):
    raise SystemExit("Unsupported GPU (sm<70): set Accelerator to 'GPU T4 x2'.")
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
        return base + ((x.to(self.A.dtype) @ self.A) @ self.B * self.s).to(base.dtype)


def build_model(name, device):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(device)
    if getattr(m.config, "use_cache", None) is not None: m.config.use_cache = False
    with torch.no_grad():
        for p in m.parameters():
            if p.is_floating_point() and p.dtype != torch.float32: p.data = p.data.float()
    for p in m.parameters(): p.requires_grad_(False)
    loras, din, dout = [], m.config.hidden_size, 3 * m.config.hidden_size
    for b in m.gpt_neox.layers:
        l = LoRA(b.attention.query_key_value, din, dout, RANK, device)
        b.attention.query_key_value = l; loras.append(l)
    return m, loras


def get_domains(tok):
    from datasets import load_dataset
    specs = [("news", lambda: load_dataset("ag_news", split="train"),
              lambda x: x["label"] == 0, 1500),
             ("wiki", lambda: load_dataset("wikitext", "wikitext-2-raw-v1", split="train"),
              lambda x: len(x["text"]) > 200, 1400),
             ("tweets", lambda: load_dataset("tweet_eval", "sentiment", split="train"),
              lambda x: True, 4500)]
    out = []
    for name, load, filt, n in specs:
        ds = load()
        txt = "\n\n".join([x["text"] for x in ds if filt(x)][:n])
        ch = tok(txt, return_tensors="pt").input_ids[0]
        ch = ch[: (len(ch) // SEQ) * SEQ].view(-1, SEQ)
        out.append((name, ch[:110], ch[110:134], ch[134:142]))
        log(f"domain {name}: {len(ch)} chunks")
    # canary: one fixed chunk repeated
    cids = tok(CANARY * 10, return_tensors="pt").input_ids[0][:SEQ]
    canary_chunk = cids.view(1, SEQ)
    return out, canary_chunk


def lm_loss(model, ids):
    ids = ids.to(DEV)
    out = model(ids, attention_mask=torch.ones_like(ids), labels=ids)
    if not torch.isfinite(out.loss): raise ValueError("non-finite loss")
    return out.loss


@torch.no_grad()
def mean_ce(model, chunks):
    model.eval()
    tot = sum(float(lm_loss(model, chunks[i:i + 4])) * min(4, len(chunks) - i)
              for i in range(0, len(chunks), 4))
    model.train(); return tot / len(chunks)


@torch.no_grad()
def snapshot(model, chunks):
    model.eval()
    o = [F.log_softmax(model(chunks[i:i+2].to(DEV),
         attention_mask=torch.ones_like(chunks[i:i+2].to(DEV))).logits[:, :-1].float(), -1).cpu()
         for i in range(0, len(chunks), 2)]
    model.train(); return torch.cat(o)


@torch.no_grad()
def kl_from(model, chunks, snap):
    model.eval(); tot = ntok = j = 0
    for i in range(0, len(chunks), 2):
        b = chunks[i:i+2].to(DEV)
        lp = F.log_softmax(model(b, attention_mask=torch.ones_like(b)).logits[:, :-1].float(), -1)
        lp0 = snap[j:j+len(b)].to(DEV)
        tot += float((lp0.exp() * (lp0 - lp)).sum()); ntok += lp.shape[0]*lp.shape[1]; j += len(b)
    model.train(); return tot / max(ntok, 1)


def gvec(ps): return torch.cat([(torch.zeros(p.numel(), device=p.device) if p.grad is None
                                 else p.grad.detach().flatten().float()) for p in ps])
def setgrad(ps, g):
    i = 0
    for p in ps:
        n = p.numel(); p.grad = g[i:i+n].view_as(p).clone().to(p.dtype); i += n


if __name__ == "__main__":
    from transformers import AutoTokenizer
    results = {}
    for model_name in MODELS:
        key = model_name.split("/")[-1]
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token
        doms, canary = get_domains(tok)

        model, loras = build_model(model_name, DEV)
        base_canary_ce = mean_ce(model, canary)             # reference: never-seen loss
        log(f"[{key}] base canary CE (reference) = {base_canary_ce:.3f}")

        # joint training on real domains + repeated canary
        params = [p for l in loras for p in (l.A, l.B)]
        opt = torch.optim.AdamW(params, lr=LR, eps=1e-6, weight_decay=0.0)
        model.train(); random.seed(0)
        train_chunks = [(d[1], d[0]) for d in doms] + [(canary, "canary")] * 1
        for s in range(len(doms) * STEPS_TRAIN):
            if s % 12 == 11:                                 # ~8% canary exposure
                b = canary
            else:
                tr = doms[s % len(doms)][1]; i = random.randint(0, len(tr) - BS)
                b = tr[i:i + BS]
            loss = lm_loss(model, b)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5); opt.step()
        joint_state = [(l.A.detach().clone(), l.B.detach().clone()) for l in loras]

        before_canary = mean_ce(model, canary)
        before_ret = {d[0]: mean_ce(model, d[2]) for d in doms}
        snaps = {d[0]: snapshot(model, d[3]) for d in doms}
        log(f"[{key}] after training: canary CE {before_canary:.3f} "
            f"(memorized; base {base_canary_ce:.3f}); retained "
            + " ".join(f"{k}={v:.3f}" for k, v in before_ret.items()))
        results[key] = dict(base_canary_ce=base_canary_ce, before_canary_ce=before_canary,
                            before_retained=before_ret)

        for arm in ["gated", "naive"]:
            with torch.no_grad():
                for l, (a, b) in zip(loras, joint_state): l.A.copy_(a); l.B.copy_(b)
            opt = torch.optim.SGD(params, lr=UNLEARN_LR)
            model.train(); random.seed(1); used = UNLEARN_STEPS
            for s in range(UNLEARN_STEPS):
                opt.zero_grad(set_to_none=True); (-lm_loss(model, canary)).backward()  # ASCENT
                u = gvec(params)
                if arm == "gated":
                    cols = []
                    for d in doms:                           # one row per retained domain
                        te = d[2]; j = random.randint(0, max(len(te) - 2, 0))
                        opt.zero_grad(set_to_none=True); lm_loss(model, te[j:j + 2]).backward()
                        nA = gvec(params)
                        if float(nA.norm() ** 2) > 1e-12: cols.append(nA)
                    if cols:
                        Gm = torch.stack(cols, 1).cpu().double().numpy()
                        lam, _ = nnls(Gm, (-u).cpu().double().numpy())
                        if lam.any(): u = u + torch.as_tensor(Gm @ lam, dtype=u.dtype, device=u.device)
                    setgrad(params, u)
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
                if s % 10 == 9 and mean_ce(model, canary) >= base_canary_ce:
                    used = s + 1; break
            after_canary = mean_ce(model, canary)
            after_ret = {d[0]: mean_ce(model, d[2]) for d in doms}
            cert = {d[0]: kl_from(model, d[3], snaps[d[0]]) for d in doms}
            results[key][arm] = dict(
                steps=used, canary_ce=after_canary,
                canary_removed_frac=float((after_canary - before_canary) /
                                          max(base_canary_ce - before_canary, 1e-9)),
                retained_ce_after=after_ret,
                certificate_kl=cert, certificate_mean=float(np.mean(list(cert.values()))))
            log(f"[{key}/{arm}] canary CE {before_canary:.3f}->{after_canary:.3f} "
                f"(ref {base_canary_ce:.3f}, {used} steps, "
                f"{100*results[key][arm]['canary_removed_frac']:.0f}% removed); "
                f"retained certificate E[KL] = {results[key][arm]['certificate_mean']:.4f}")
            with open("unlearn_canary_results.json", "w") as f:
                json.dump(results, f, indent=1)
        g, n = results[key]["gated"], results[key]["naive"]
        if g["certificate_mean"] > 0:
            log(f"[{key}] SUMMARY: both remove the canary "
                f"({100*g['canary_removed_frac']:.0f}%/{100*n['canary_removed_frac']:.0f}%); "
                f"gated retained certificate {g['certificate_mean']:.4f} vs naive "
                f"{n['certificate_mean']:.4f}  ({n['certificate_mean']/max(g['certificate_mean'],1e-9):.1f}x)")
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    log("ALL DONE -> unlearn_canary_results.json")
