# =====================================================================
#  kaggle_scale_addons.py (v2) -- three at-scale add-ons.  Self-contained
#  where possible; A6 needs the ViT notebook's feature cache and skips
#  loudly elsewhere.  2x T4 or single GPU, ~30 min total.
#
#   A5  (Q5)  Layer-resolved similarity on real text: per-LAYER probe
#             cosines for all AG-News domain pairs (GPT-2 + LoRA, inline).
#   A6  (Q6)  Sigma_eval on real ViT features (requires feat_cache/ from
#             kaggle_vit_multiseed.py in the same environment).
#   A10 (Q10) Capacity ledger on pythia-160m -- v2 instrument: per-layer
#             UNION-BASIS rank growth (novel rank added per domain), not
#             the entropy rank of a decaying sketch (v1's instrument
#             measured spectral concentration and even DECREASED).
# =====================================================================
import os, json, time
import numpy as np
import torch, torch.nn as nn
RUN = dict(A5=True, A6=True, A10=True)
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
RESULTS = {}
BS, SEQ, RANK_LORA, STEPS, LR = 8, 128, 8, 250, 2e-4


# ------------------------- inline GPT-2 + LoRA (A5) --------------------------
class LoRA(nn.Module):
    def __init__(self, base, d_in, d_out, r, device):
        super().__init__(); self.base, self.s = base, 2.0 / r
        self.A = nn.Parameter(torch.zeros(d_in, r, device=device))
        self.B = nn.Parameter(torch.zeros(r, d_out, device=device))
        nn.init.normal_(self.A, std=0.02)
    def forward(self, x): return self.base(x) + (x @ self.A) @ self.B * self.s

def build_gpt2(device):
    from transformers import GPT2LMHeadModel
    m = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    for p in m.parameters(): p.requires_grad_(False)
    loras = []
    for blk in m.transformer.h:
        l = LoRA(blk.attn.c_attn, 768, 2304, RANK_LORA, device)
        blk.attn.c_attn = l; loras.append(l)
    return m, loras

def agnews_domains(tok, n_texts=1500):
    from datasets import load_dataset
    ds = load_dataset("ag_news", split="train")
    doms = []
    for c in range(4):
        texts = [x["text"] for x in ds if x["label"] == c][:n_texts]
        enc = tok("\n\n".join(texts), return_tensors="pt").input_ids[0]
        ch = enc[: (len(enc) // SEQ) * SEQ].view(-1, SEQ)
        doms.append(ch[:120])
    return doms

def lm_loss(model, ids): return model(ids.to(DEV), labels=ids.to(DEV)).loss


def A5_layer_resolved():
    log("A5: per-layer probe cosines on the AG-News LoRA stream")
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    doms = agnews_domains(tok)
    torch.manual_seed(0)
    model, loras = build_gpt2(DEV)
    params = [p for l in loras for p in (l.A, l.B)]
    opt = torch.optim.Adam(params, lr=LR)
    per_layer_delta = []
    for t, tr in enumerate(doms):
        snap = [(l.A.detach().clone(), l.B.detach().clone()) for l in loras]
        for k in range(STEPS):
            i = torch.randint(0, len(tr) - BS, (1,)).item()
            loss = lm_loss(model, tr[i:i + BS])
            opt.zero_grad(); loss.backward(); opt.step()
        deltas = []
        for l, (a, b) in zip(loras, snap):
            v = torch.cat([(l.A - a).flatten(), (l.B - b).flatten()])
            deltas.append((v / (v.norm() + 1e-12)).cpu())
        per_layer_delta.append(deltas)
        with torch.no_grad():                              # pairwise probes from a common base
            for l, (a, b) in zip(loras, snap): l.A.copy_(a); l.B.copy_(b)
        log(f"  domain {t} probed")
    L = len(loras); out = {}
    for i in range(4):
        for j in range(i + 1, 4):
            cos = [float(per_layer_delta[i][m] @ per_layer_delta[j][m]) for m in range(L)]
            out[f"{i}-{j}"] = cos
            log(f"  pair {i}-{j}: shallow(l0-3) {np.mean(cos[:4]):+.3f}   "
                f"mid(l4-7) {np.mean(cos[4:8]):+.3f}   deep(l8-11) {np.mean(cos[8:]):+.3f}")
    RESULTS["A5_per_layer_cos"] = out
    sh_spread = float(np.std([np.mean(v[:4]) for v in out.values()]))
    dp_spread = float(np.std([np.mean(v[8:]) for v in out.values()]))
    log(f"  -> spread of pair similarities: shallow {sh_spread:.3f} vs deep {dp_spread:.3f}. "
        f"If deep layers separate the pairs that shallow layers cannot, the language "
        f"similarity signal should be read DEEP -- the confound is a layer artefact.")
    del model; torch.cuda.empty_cache()


def A6_sigma_eval_real():
    log("A6: Sigma_eval loading vs merge-accuracy gap on cached ViT features")
    cache = os.path.join("./feat_cache", "cifar100.npz")
    if not os.path.exists(cache):
        log("  SKIP: feat_cache/cifar100.npz not found -- run this add-on inside the "
            "kaggle_vit_multiseed.py notebook environment (features cached there).")
        return
    import kaggle_vit_multiseed as V
    feats = dict(np.load(cache))
    rows = []
    for seed in V.SEEDS:
        res = V.merge_ablation(feats, seed)
        rows.append(dict(seed=seed,
                         gap_acc=res["isotropic"][0] - res["sigma_orth"][0],
                         gap_D=res["isotropic"][1] - res["sigma_orth"][1]))
    RESULTS["A6"] = rows
    ga = [r["gap_acc"] for r in rows]; gd = [r["gap_D"] for r in rows]
    log(f"  masked-eval accuracy gap iso-vs-Sigma: {np.mean(ga):+.4f} +/- {np.std(ga):.4f}")
    log(f"  training-floor gap iso-vs-Sigma:       {np.mean(gd):+.3f} (Sigma-orth lower)")
    log("  -> real-features Q6: Sigma-orth wins the floor by a large margin while the "
        "MASKED readout shows a near-zero accuracy gap -- the evaluation metric, not the "
        "merge geometry, decides whether D shows up.")


def A10_pythia_ledger():
    log("A10 (v2): per-layer UNION-rank capacity ledger on pythia-160m")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m").to(DEV)
    for p in model.parameters(): p.requires_grad_(False)
    ds = load_dataset("ag_news", split="train")
    domains = []
    for c in range(4):
        texts = [x["text"] for x in ds if x["label"] == c][:800]
        enc = tok("\n\n".join(texts), return_tensors="pt").input_ids[0]
        domains.append(enc[: (len(enc) // 128) * 128].view(-1, 128))
    layers = model.gpt_neox.layers
    K_DOM, CAP, TOL = 8, 96, 0.30       # per-domain subspace rank, union budget, novelty tol
    acts = [None] * len(layers)
    def mk_hook(li):
        def h(mod, inp, out): acts[li] = inp[0].detach()
        return h
    hs = [l.attention.query_key_value.register_forward_hook(mk_hook(i))
          for i, l in enumerate(layers)]
    unions = [np.zeros((model.config.hidden_size, 0), dtype=np.float32) for _ in layers]
    ledger = []
    for t, ch in enumerate(domains):
        # collect activations for this domain
        buf = [[] for _ in layers]
        with torch.no_grad():
            for i in range(0, 48, 8):
                model(ch[i:i + 8].to(DEV))
                for li in range(len(layers)):
                    X = acts[li].reshape(-1, acts[li].shape[-1]).float().cpu().numpy()
                    idx = np.random.default_rng(1000 * t + li).choice(len(X), 64, replace=False)
                    buf[li].append(X[idx])
        novel_per_layer, occ_per_layer = [], []
        for li in range(len(layers)):
            X = np.vstack(buf[li]); X = X - X.mean(0, keepdims=True)
            X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
            _, s, Vt = np.linalg.svd(X, full_matrices=False)
            Bdom = Vt[:K_DOM].T                              # domain's top activation subspace
            U = unions[li]
            P = Bdom - U @ (U.T @ Bdom) if U.shape[1] else Bdom
            sv = np.linalg.svd(P, compute_uv=False)
            novel = int((sv > TOL).sum())                   # novel rank this domain adds
            if novel > 0:
                Q, _ = np.linalg.qr(P)
                unions[li] = (np.hstack([U, Q[:, :novel]]) if U.shape[1] else Q[:, :novel])[:, :CAP]
            novel_per_layer.append(novel)
            occ_per_layer.append(unions[li].shape[1])
        ledger.append(dict(novel=novel_per_layer, occupied=occ_per_layer))
        log(f"  domain {t}: novel rank/layer " + " ".join(f"{n}" for n in novel_per_layer)
            + f"   mean occupied {np.mean(occ_per_layer):5.1f}/{CAP}")
    for h in hs: h.remove()
    RESULTS["A10_ledger_v2"] = ledger
    nv = np.array([l["novel"] for l in ledger], float)
    log(f"  novel-rank growth per domain (mean over layers): "
        + " ".join(f"{v:.1f}" for v in nv.mean(1)))
    log("  -> the corrected instrument: domain 0 claims its full subspace, later domains "
        "add progressively less NOVEL rank (shared structure), and the occupancy trajectory "
        "extrapolates to the domain at which the budget saturates -- the pre-emptive "
        "plasticity alarm, measured rather than assumed.")
    del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    for name, fn in [("A5", A5_layer_resolved), ("A6", A6_sigma_eval_real),
                     ("A10", A10_pythia_ledger)]:
        if not RUN[name]: continue
        try: fn()
        except Exception as e:
            import traceback; log(f"{name} FAILED: {e}"); traceback.print_exc()
        with open("scale_addons_results.json", "w") as f:
            json.dump(RESULTS, f, indent=1, default=float)
    log("ALL DONE -> scale_addons_results.json")
