# =====================================================================
#  kaggle_jspace_layers.py -- make the J-space bridge CONCLUSIVE, cheaply.
#  ~8-12 min on 1x T4 (no continual-learning training loop).
#
#  The first probe (kaggle_jspace_interference) was inconclusive on the
#  CAUSAL test because at layer 16 no rank-64 projection protected -- not
#  even Sigma_A's own (6%) -- i.e. the interference was high-effective-
#  rank there, so low-rank protection cannot discriminate.  Two questions
#  were left open and are answered here PER LAYER, from geometry alone:
#
#   Q1 (feasibility)  is the interference EVER low-rank enough that a
#      workspace-sized (~k=25-64) subspace could carry it?  -> the
#      effective rank and the rank-to-80%-mass of Sigma_A = E[g g^T],
#      g = d CE_A / d h_l, at each layer.
#   Q2 (enrichment)   does the J-space carry more interference energy
#      than a random subspace of equal rank, WITHOUT training?  The
#      expected interference of a random update is tr(P^T Sigma P); so
#      enrichment(P_J) = [tr(P_J^T Sigma P_J)/tr(Sigma)] / (r/d), and
#      likewise the overlap of P_J with Sigma_A's own top-r subspace.
#
#  Reading: a layer with (80%-mass-rank <= a workspace size) AND
#  (enrichment >> 1) is where the strong bridge could hold -- run the
#  causal H2 there.  If NO layer is low-rank, the strong "same subspace"
#  claim is refuted and the honest result is the ~2x energy enrichment
#  (J-space carries more than its share, but not the whole of forgetting).
#
#  Self-contained: recomputes the J-lens object directly (no package).
# =====================================================================
import json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL = "EleutherAI/pythia-410m"
R_J = 32                       # workspace-sized J-space rank (~ their k)
JS_PROBES, JS_CTX = 64, 6      # randomized range finder for E[readout Jacobian]
SIGMA_TOKENS = 24              # token-batches for Sigma_A = E[g g^T] per layer
SEQ = 128
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
if DEV.startswith("cuda") and torch.cuda.get_device_capability(0) < (7, 0):
    raise SystemExit("Unsupported GPU (sm<70): set Accelerator to 'GPU T4 x2'.")
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def build(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(DEV)
    m.config.use_cache = False
    for p in m.parameters(): p.requires_grad_(False)
    return m, tok


class GradLeaf:
    """Hook that exposes layer-l residual stream h_l as a grad leaf."""
    def __init__(self, model, layer):
        self.layer = model.gpt_neox.layers[layer]; self.h = None; self.handle = None
    def _hook(self, m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        h.requires_grad_(True); h.retain_grad(); self.h = h; return o
    def __enter__(self): self.handle = self.layer.register_forward_hook(self._hook); return self
    def __exit__(self, *a): self.handle.remove()


def get_A_contexts(tok, n_tr=24, n_va=24):
    from datasets import load_dataset
    ds = load_dataset("ag_news", split="train")               # domain A = news
    ids = tok("\n\n".join([x["text"] for x in ds if x["label"] == 0][:1500]),
              return_tensors="pt").input_ids[0]
    ch = ids[: (len(ids) // SEQ) * SEQ].view(-1, SEQ)
    return ch[:n_tr], ch[n_tr:n_tr + n_va]


def sigma_at(model, layer, batches):
    """Sigma_A = E[g g^T], g = d CE_A / d h_l per token, as a d x d matrix."""
    d = model.config.hidden_size
    S = torch.zeros(d, d, device=DEV); n = 0
    for ids in batches:
        ids = ids.view(1, -1).to(DEV)
        model.zero_grad(set_to_none=True)
        with GradLeaf(model, layer) as gl:
            lo = model(ids, attention_mask=torch.ones_like(ids)).logits
            loss = F.cross_entropy(lo[0, :-1], ids[0, 1:])
            loss.backward()
            g = gl.h.grad[0].detach()                          # (T x d) per-token grads
        S += g.T @ g; n += g.shape[0]
    return S / max(n, 1)


def jspace_at(model, layer, contexts, rank):
    """Top-`rank` range of the context-averaged readout Jacobian E[d logits/d h_l]."""
    d = model.config.hidden_size
    sketch = torch.zeros(JS_PROBES, d, device=DEV)
    for p in range(JS_PROBES):
        acc = torch.zeros(d, device=DEV); k = 0
        for ids in contexts[:JS_CTX]:
            ids = ids.view(1, -1).to(DEV)
            model.zero_grad(set_to_none=True)
            with GradLeaf(model, layer) as gl:
                last = model(ids, attention_mask=torch.ones_like(ids)).logits[0, -1]
                top = last.topk(20).indices
                probe = torch.zeros_like(last); probe[top] = torch.randn(20, device=DEV)
                (probe @ last).backward()
                acc += gl.h.grad[0, -1].detach(); k += 1
        sketch[p] = acc / max(k, 1)
    return torch.linalg.qr(sketch.T)[0][:, :rank]


def mass_rank(evals, frac):
    c = torch.cumsum(evals, 0) / evals.sum()
    return int((c < frac).sum().item()) + 1


if __name__ == "__main__":
    model, tok = build(MODEL)
    d = model.config.hidden_size
    L = model.config.num_hidden_layers
    tr, va = get_A_contexts(tok)
    layers = list(range(2, L, 2))
    log(f"{MODEL}: d={d}, {L} layers; probing {layers}")
    print(f"\n  {'layer':>5s} {'eff_rank':>9s} {'r@50%':>6s} {'r@80%':>6s} "
          f"{'J-enrich':>9s} {'overlapJS':>10s}  {'baseline r/d':>12s}")
    rows = []
    for layer in layers:
        S = sigma_at(model, layer, tr)
        ev, V = torch.linalg.eigh(S); ev = ev.flip(0).clamp(min=0); V = V.flip(1)
        eff = float((ev.sum() ** 2) / (ev.square().sum() + 1e-12))    # participation ratio
        r50, r80 = mass_rank(ev, 0.5), mass_rank(ev, 0.8)
        P_J = jspace_at(model, layer, list(va), R_J)
        enrich = float((torch.trace(P_J.T @ S @ P_J) / (torch.trace(S) + 1e-12)) / (R_J / d))
        ov = float((P_J.T @ V[:, :R_J]).square().sum() / R_J)
        rows.append(dict(layer=layer, eff_rank=eff, r50=r50, r80=r80,
                         j_enrichment=enrich, overlap_JS=ov))
        print(f"  {layer:5d} {eff:9.1f} {r50:6d} {r80:6d} {enrich:9.2f} {ov:10.3f}"
              f"  {R_J/d:12.3f}")

    # conclusion
    cand = [r for r in rows if r["r80"] <= 64 and r["j_enrichment"] >= 1.5]
    best_enr = max(rows, key=lambda r: r["j_enrichment"])
    print("\n  CONCLUSION:")
    if cand:
        c = min(cand, key=lambda r: r["r80"])
        print(f"   feasible bridge layer(s): {[r['layer'] for r in cand]} -- interference is")
        print(f"   low-rank (80% mass in <=64 dims) AND J-space is enriched; run the causal")
        print(f"   H2 at layer {c['layer']} (r80={c['r80']}, enrichment {c['j_enrichment']:.1f}x).")
    else:
        print(f"   NO layer is workspace-low-rank: 80%-mass rank is >64 at every layer")
        print(f"   (min {min(r['r80'] for r in rows)}), so the low-dimensional J-space")
        print(f"   CANNOT carry the bulk of forgetting -- the strong 'same subspace' claim")
        print(f"   is refuted.  The honest, robust result is the energy ENRICHMENT: J-space")
        print(f"   carries {best_enr['j_enrichment']:.1f}x its share at layer {best_enr['layer']}")
        print(f"   (>1 at {sum(r['j_enrichment']>1 for r in rows)}/{len(rows)} layers).")
    json.dump(dict(model=MODEL, rank_J=R_J, d=d, layers=rows),
              open("jspace_layers_results.json", "w"), indent=1)
    log("ALL DONE -> jspace_layers_results.json")
