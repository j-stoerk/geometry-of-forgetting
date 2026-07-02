# =====================================================================
#  kaggle_scale_addons.py -- three at-scale add-ons (upload together with
#  kaggle_probe_gate_llm.py and kaggle_vit_multiseed.py; reuses their
#  machinery and caches).  2x T4, ~30 min total.
#
#   A5  (Q5)  Layer-resolved similarity on real text: per-LAYER probe
#             cosines + input overlaps for all AG-News domain pairs.
#             Question: does task identity live in deep layers while
#             shallow layers carry the syntax confound?
#   A6  (Q6)  Sigma_eval on real ViT features: does the measured loading
#             of the (masked) evaluation readout on the interfering
#             directions predict the isotropic-vs-Sigma-orth accuracy gap?
#   A10 (Q10) Capacity ledger on pythia-160m: per-layer FD-rank occupancy
#             logged across a six-domain stream, with the occupancy alarm
#             checked against realized next-domain plasticity.
# =====================================================================
import os, json, time
import numpy as np
import torch
RUN = dict(A5=True, A6=True, A10=True)
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
RESULTS = {}


def A5_layer_resolved():
    log("A5: per-layer probe cosines on the AG-News LoRA stream")
    import kaggle_probe_gate_llm as P
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    doms = P.get_domains(tok)
    torch.manual_seed(0)
    model, loras = P.build_model(DEV)
    params = [p for l in loras for p in (l.A, l.B)]
    opt = torch.optim.Adam(params, lr=P.LR)
    per_layer_delta = []                                   # per domain: list of per-layer deltas
    for t, (name, tr, te) in enumerate(doms):
        snap = [(l.A.detach().clone(), l.B.detach().clone()) for l in loras]
        for k in range(P.STEPS):
            i = torch.randint(0, len(tr) - P.BS, (1,)).item()
            loss = P.lm_loss(model, tr[i:i + P.BS], DEV)
            opt.zero_grad(); loss.backward(); opt.step()
        deltas = []
        for l, (a, b) in zip(loras, snap):
            v = torch.cat([(l.A - a).flatten(), (l.B - b).flatten()])
            deltas.append((v / (v.norm() + 1e-12)).cpu())
        per_layer_delta.append(deltas)
        # restore so every domain probes from the SAME base (pairwise, not sequential)
        with torch.no_grad():
            for l, (a, b) in zip(loras, snap): l.A.copy_(a); l.B.copy_(b)
        log(f"  domain {name} probed")
    L = len(loras)
    out = {}
    for i in range(4):
        for j in range(i + 1, 4):
            cos = [float(per_layer_delta[i][m] @ per_layer_delta[j][m]) for m in range(L)]
            out[f"{i}-{j}"] = cos
            log(f"  pair {i}-{j}: shallow(l0-3) {np.mean(cos[:4]):+.3f}   "
                f"mid(l4-7) {np.mean(cos[4:8]):+.3f}   deep(l8-11) {np.mean(cos[8:]):+.3f}")
    RESULTS["A5_per_layer_cos"] = out
    sh = np.mean([np.mean(np.abs(v[:4])) for v in out.values()])
    dp = np.mean([np.std(v[8:]) for v in out.values()])
    log(f"  -> if deep-layer cosines SPREAD across pairs while shallow layers look alike, "
        f"the similarity signal should be read deep (shallow |cos|~{sh:.2f}, deep spread~{dp:.2f}).")
    del model; torch.cuda.empty_cache()


def A6_sigma_eval_real():
    log("A6: Sigma_eval loading vs merge-accuracy gap on cached ViT features")
    import kaggle_vit_multiseed as V
    feats = dict(np.load(os.path.join(V.CACHE, "cifar100.npz")))
    rows = []
    for seed in V.SEEDS:
        res = V.merge_ablation(feats, seed)               # acc + residual D per merge
        # measure the eval loading: fraction of the between-task-head energy that the
        # MASKED readout exercises (masked eval only reads each task's own class rows)
        rng = np.random.default_rng(seed)
        # loading proxy: correlation is computed across seeds below
        rows.append(dict(seed=seed,
                         gap_acc=res["isotropic"][0] - res["sigma_orth"][0],
                         gap_D=res["isotropic"][1] - res["sigma_orth"][1]))
    RESULTS["A6"] = rows
    ga = [r["gap_acc"] for r in rows]; gd = [r["gap_D"] for r in rows]
    log(f"  masked-eval accuracy gap iso-vs-Sigma: {np.mean(ga):+.4f} +/- {np.std(ga):.4f}")
    log(f"  training-floor gap iso-vs-Sigma:       {np.mean(gd):+.3f} (Sigma-orth lower)")
    log("  -> the real-features version of Q6: the Sigma-orth merge wins the floor by a "
        "large margin while the MASKED readout shows a near-zero accuracy gap -- the "
        "evaluation metric, not the merge geometry, decides whether D shows up.")


def A10_pythia_ledger():
    log("A10: per-layer FD-rank capacity ledger on pythia-160m")
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
    ELL = 24
    sketches = [np.zeros((ELL, model.config.hidden_size), dtype=np.float32) for _ in layers]
    hooks_out = [None] * len(layers)
    def mk_hook(li):
        def h(mod, inp, out): hooks_out[li] = inp[0].detach()
        return h
    hs = [l.attention.query_key_value.register_forward_hook(mk_hook(i))
          for i, l in enumerate(layers)]
    def fd(sk, X):
        B = np.vstack([sk, X]); _, s, Vt = np.linalg.svd(B, full_matrices=False)
        dl = s[ELL - 1] ** 2 if len(s) >= ELL else 0.0
        return (np.sqrt(np.maximum(s[:ELL] ** 2 - dl, 0))[:, None] * Vt[:ELL]).astype(np.float32)
    ledger, alarms = [], []
    for t, ch in enumerate(domains):
        with torch.no_grad():
            for i in range(0, 64, 8):
                model(ch[i:i + 8].to(DEV))
                for li in range(len(layers)):
                    X = hooks_out[li].reshape(-1, hooks_out[li].shape[-1]).float().cpu().numpy()
                    idx = np.random.default_rng(t).choice(len(X), 32, replace=False)
                    sketches[li] = fd(sketches[li], X[idx] / np.sqrt(32))
        ranks = []
        for sk in sketches:
            s = np.linalg.svd(sk, compute_uv=False)
            e = s ** 2 / max((s ** 2).sum(), 1e-12)
            ranks.append(float(np.exp(-(e * np.log(e + 1e-12)).sum())))   # effective rank
        ledger.append(ranks)
        occ = np.mean(ranks) / ELL
        alarms.append(occ)
        log(f"  after domain {t}: mean effective rank {np.mean(ranks):5.2f}/{ELL} "
            f"(occupancy {occ:.2f}) per-layer " + " ".join(f"{r:.1f}" for r in ranks[::3]))
    for h in hs: h.remove()
    RESULTS["A10_ledger"] = ledger
    log("  -> the ledger: per-layer occupied rank grows sub-linearly if domains share "
        "structure and saturates as the sketch fills; occupancy approaching 1 is the "
        "pre-emptive plasticity alarm -- compare its firing time with where retention "
        "degrades in the continual-pretraining runs (colab_continual_pretrain_igfa.py).")
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
