# =====================================================================
#  Geometry of Forgetting -- H2 / H3 / H12 in one Kaggle file.
#
#  Paste into a single Kaggle GPU cell and run.  Frozen LM + LoRA, continual
#  pretraining over several text domains, with:
#
#    H2  Eigengate Null-Space Router: a backward-pass gradient projection
#        g <- g - U (U^T g) that routes each new domain's update strictly into
#        the BOTTOM d-k eigendirections of a per-layer Frequent-Directions (FD)
#        sketch -- isolated continual pretraining, no external experts.
#    H3  Layer-Wise Capacity Budgets (T ~ d/r): the protected rank cap k_l is
#        assigned PER LAYER from its trace ratio / effective rank, so shallow
#        (high-rank, general) layers stay near-unconstrained and deep
#        (low-rank, specific) layers enforce strict orthogonalisation.
#    H12 Billion-Parameter Pipeline: the topological switch -- layers below L/2
#        pass gradients unaltered (shared isotropic), layers >= L/2 route
#        through the Eigengate into the FD null space -- combining H2+H3+FD.
#
#  Logs (better too much than too little): domain x domain loss matrix, top-50
#  eigenvalue spectra, subspace trace ratio, consecutive-domain similarity s,
#  per-layer k_l budget array over depth, gate/projection activation %, and a
#  VRAM comparison of the FD sketch vs full-covariance OGD vs AdamW state.
#
#  Defaults: EleutherAI/pythia-160m (set pythia-410m/1b if you have VRAM), ~20-40
#  min on a T4.  HF_TOKEN is read ONLY from the environment -- never hard-code it.
# =====================================================================
import os, sys, re, time, json, math, subprocess, warnings
warnings.filterwarnings("ignore")

CONFIG = dict(
    MODEL="EleutherAI/pythia-160m",       # -> pythia-410m / pythia-1b for scale
    STEPS_PER_DOMAIN=150, SEQ_LEN=128, LORA_R=16, LR=2e-3,
    ELL=64, RANK_KEEP=32, GAMMA=0.9,      # FD sketch rows, max protected rank, EMA decay
    UPDATE_EVERY=25,                      # refresh U and k_l every N steps
    DEEP_FROM=0.5,                        # layers with depth-fraction >= this are routed (H12)
    SEED=0,
)

def pip(*p): subprocess.run([sys.executable, "-m", "pip", "install", "-q", *p], check=False)
try: import torch, transformers, peft, datasets, tqdm  # noqa
except Exception: pip("torch", "transformers", "peft", "datasets", "accelerate", "tqdm")

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
# Kaggle ships an old torchao that peft's LoRA dispatcher rejects; neutralise it (unused here).
for _m in ["peft.tuners.lora.torchao", "peft.import_utils"]:
    try:
        import importlib; _mod = importlib.import_module(_m)
        if hasattr(_mod, "is_torchao_available"): _mod.is_torchao_available = lambda: False
    except Exception: pass

torch.manual_seed(CONFIG["SEED"]); np.random.seed(CONFIG["SEED"])
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()
def stamp(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


# =====================================================================
#  H2: Eigengate autograd Function (documented) -- routes the weight gradient
#  into the null space of U.  We APPLY it through a robust parameter backward
#  hook (below), which intercepts the same gradient without monkeypatching peft.
# =====================================================================
class Eigengate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, U):
        ctx.U = U; return W
    @staticmethod
    def backward(ctx, gW):
        U = ctx.U
        if U is not None and U.numel():
            gW = gW - (gW @ U) @ U.T           # project rows (in_dim space) off the top-k
        return gW, None


# =====================================================================
#  model + LoRA
# =====================================================================
tok = AutoTokenizer.from_pretrained(CONFIG["MODEL"], token=os.environ.get("HF_TOKEN"))
if tok.pad_token is None: tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(CONFIG["MODEL"], token=os.environ.get("HF_TOKEN")).to(DEV)
def lora_targets(mid):
    mid = mid.lower()
    if "gpt2" in mid: return ["c_attn"]
    if "pythia" in mid or "neox" in mid: return ["query_key_value"]
    return ["q_proj", "v_proj"]
model = get_peft_model(base, LoraConfig(r=CONFIG["LORA_R"], lora_alpha=2 * CONFIG["LORA_R"],
                                        target_modules=lora_targets(CONFIG["MODEL"]), lora_dropout=0.0)).to(DEV)
model.config.use_cache = False

# order LoRA-A layers by transformer depth so we can apply the shallow/deep switch (H12)
def depth_of(name):
    m = re.search(r"\.(\d+)\.", name); return int(m.group(1)) if m else 0
lora_A = {n: m for n, m in model.named_modules() if n.endswith("lora_A.default") and isinstance(m, nn.Linear)}
order = sorted(lora_A, key=depth_of)
Lmax = max(depth_of(n) for n in order) + 1
in_dim = next(iter(lora_A.values())).in_features
stamp(f"{len(lora_A)} LoRA-A layers over depth 0..{Lmax-1}, in_dim={in_dim}, device={DEV}")

# per-layer state
ELL, KMAX = CONFIG["ELL"], CONFIG["RANK_KEEP"]
ST = {n: dict(FD=torch.zeros(ELL, in_dim, device=DEV), U=None, Uprot=None, k=0,
              deep=(depth_of(n) >= CONFIG["DEEP_FROM"] * Lmax),
              proj_energy=[]) for n in order}

# capture input activations to each LoRA-A (for the FD sketch)
cap = {}
for n, m in lora_A.items():
    m.register_forward_pre_hook(lambda _, i, nm=n: cap.__setitem__(
        nm, (i[0] if isinstance(i, tuple) else i).detach().reshape(-1, in_dim).float()))

# H2 via robust parameter backward hook: project the weight grad off U (deep layers only)
def make_hook(nm):
    def hook(g):
        s = ST[nm]
        # protect the subspace of PAST domains only (Uprot, frozen at each domain boundary);
        # never the live/current subspace -> the current domain still learns freely.
        if s["deep"] and s["Uprot"] is not None:
            Ug = s["Uprot"].to(g.dtype); gp = (g @ Ug) @ Ug.T
            s["proj_energy"].append(float((gp.norm() / (g.norm() + 1e-9)).item()))
            return g - gp                       # route update into the null space (bottom d-k)
        return g
    return hook
for n, m in lora_A.items():
    m.weight.register_hook(make_hook(n))

def fd_insert(nm, rows):
    B = ST[nm]["FD"]
    for x in rows:
        z = (~B.any(1)).nonzero()
        if len(z) == 0:                          # densify (Liberty): shrink singular values, free half the rows
            U, s, Vt = torch.linalg.svd(B, full_matrices=False)
            delta = s[ELL // 2] ** 2
            B = torch.diag(torch.sqrt(torch.clamp(s ** 2 - delta, min=0))) @ Vt
            ST[nm]["FD"] = B; z = (~B.any(1)).nonzero()
        B[z[0, 0]] = x

def refresh(nm):
    """H3: set the per-layer rank cap k_l from the FD spectrum (effective rank), 0 for shallow layers."""
    s = ST[nm]
    _, sv, Vt = torch.linalg.svd(s["FD"], full_matrices=False)
    sv = sv.clamp(min=0)
    eff = float(((sv.sum() ** 2) / (sv ** 2).sum() + 1e-9).item()) if sv.sum() > 0 else 0.0
    if not s["deep"]:
        s["k"] = 0; s["U"] = None                # shallow = isotropic shared (H12)
    else:
        s["k"] = int(max(1, min(KMAX, math.ceil(eff))))
        s["U"] = Vt[:s["k"]].T.contiguous()      # in_dim x k  (top-k of the accumulated FD)
    s["eff"], s["sv"] = eff, sv[:50].detach().cpu().numpy()


# =====================================================================
#  data
# =====================================================================
DOMAINS = [("ag_news", None, "text"), ("imdb", None, "text"),
           ("wikitext", "wikitext-2-raw-v1", "text"), ("dbpedia_14", None, "content")]
def make(name, cfg, field, n=400):
    try:
        ds = load_dataset(name, cfg, split="train") if cfg else load_dataset(name, split="train")
        txt = [t for t in ds.select(range(min(len(ds), n * 8)))[field] if isinstance(t, str) and len(t.strip()) > 64][:n]
    except Exception as e:
        stamp(f"  dataset {name} unavailable ({e})"); return None
    e = tok(txt, return_tensors="pt", padding="max_length", truncation=True, max_length=CONFIG["SEQ_LEN"])
    return e["input_ids"], e["attention_mask"]
domains = [d for d in (make(*x) for x in DOMAINS) if d is not None]
if len(domains) < 2: stamp("not enough datasets -> abort"); sys.exit(0)
ND = len(domains)
stamp(f"{ND} domains loaded")

def lm_loss(ids, am):
    lg = model(input_ids=ids, attention_mask=am).logits
    lab = ids.clone(); lab[am == 0] = -100
    return F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)), lab[:, 1:].reshape(-1), ignore_index=-100)

@torch.no_grad()
def evloss(d):
    ids, am = domains[d]; tot = n = 0
    for i in range(0, min(len(ids), 96), 16):
        b, m = ids[i:i+16].to(DEV), am[i:i+16].to(DEV)
        tot += lm_loss(b, m).item() * len(b); n += len(b)
    return tot / max(n, 1)


# =====================================================================
#  continual pretraining with the H12 pipeline + heavy logging
# =====================================================================
def subspace_sim(prev, cur):
    ov = []
    for n in order:
        if prev.get(n) is not None and cur.get(n) is not None:
            A, B = prev[n], cur[n]; k = min(A.shape[1], B.shape[1])
            ov.append(float((torch.linalg.norm(A[:, :k].T @ B[:, :k], "fro") ** 2 / k).item()))
    return float(np.mean(ov)) if ov else float("nan")

opt = torch.optim.Adam([p for n, p in model.named_parameters() if p.requires_grad], lr=CONFIG["LR"])
lossM = np.full((ND, ND), np.nan)          # lossM[i,j] = loss on domain j after training domain i
vram, sims, gate_pct = [], [], []
prevU = {n: None for n in order}
if DEV == "cuda": torch.cuda.reset_peak_memory_stats()

for di in range(ND):
    ids, am = domains[di]; model.train()
    for n in ST: ST[n]["proj_energy"] = []
    # FREEZE the protection basis from PAST domains (the FD currently reflects domains 0..di-1).
    # Domain 0 has no past -> Uprot=None everywhere -> it learns freely (H12 pipeline start).
    for n in order:
        refresh(n)
        ST[n]["Uprot"] = ST[n]["U"] if di > 0 else None
    for step in tqdm(range(CONFIG["STEPS_PER_DOMAIN"]), desc=f"domain {di} ({DOMAINS[di][0]})", leave=False):
        i = (step * 16) % max(1, len(ids) - 16)
        b, m = ids[i:i+16].to(DEV), am[i:i+16].to(DEV)
        opt.zero_grad(); lm_loss(b, m).backward()             # backward hooks project deep-layer grads (H2)
        for n in order:                                        # FD sketch keeps accumulating (for the NEXT boundary)
            x = cap.get(n)
            if x is not None and x.shape[1] == in_dim:
                fd_insert(n, x[torch.randperm(len(x))[:4]])
        opt.step()
    # ---- per-domain diagnostics (recompute U/k/spectra from the now-updated FD) ----
    for n in order: refresh(n)
    for dj in range(ND): lossM[di, dj] = evloss(dj)
    kl = [ST[n]["k"] for n in order]
    curU = {n: ST[n]["U"] for n in order}
    sims.append(subspace_sim(prevU, curU)); prevU = {n: (U.clone() if U is not None else None) for n, U in curU.items()}
    ge = np.mean([np.mean(ST[n]["proj_energy"]) for n in order if ST[n]["proj_energy"]]) if any(ST[n]["proj_energy"] for n in order) else 0.0
    gate_pct.append(float(ge))
    if DEV == "cuda": vram.append(torch.cuda.max_memory_allocated() / 1e6)
    d0 = lossM[di, 0] - lossM[0, 0]
    stamp(f"domain {di} done | loss(dom0 rise)={d0:+.3f} | avg-loss={np.nanmean(lossM[di]):.3f} "
          f"| mean deep-layer grad-energy routed={ge:.2f} | consec-subspace-sim={sims[-1]:.3f}"
          + (f" | peakVRAM={vram[-1]:.0f}MB" if vram else ""))
    stamp(f"  H3 per-layer k_l over depth 0..{Lmax-1}: {kl}  (0=shallow/isotropic, >0=deep/protected)")
    rep = order[-1]                                            # a deep representative layer
    stamp(f"  layer '{rep.split('.default')[0][-40:]}': eff-rank={ST[rep]['eff']:.1f} "
          f"top8 eig={np.round(ST[rep]['sv'][:8]**2, 3).tolist()}")

# =====================================================================
#  memory accounting (H2/H12) + final report
# =====================================================================
nL = len(order)
mem_fd = nL * ELL * in_dim
mem_fullcov = nL * in_dim * in_dim                             # full-covariance OGD
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
mem_adam = 2 * n_train
tr_ratio = float(np.mean([ (ST[n]['sv'][:ST[n]['k']]**2).sum() / ((ST[n]['sv']**2).sum()+1e-9)
                           for n in order if ST[n]['k']>0 ])) if any(ST[n]['k']>0 for n in order) else 0.0

print("\n" + "=" * 70)
stamp("SUMMARY")
print("domain x domain loss matrix (rows = after training domain i):")
for r in lossM: print("   " + " ".join(f"{v:5.2f}" if not math.isnan(v) else "  .  " for v in r))
diag = np.array([lossM[t, t] for t in range(ND)]); final = lossM[ND-1]
stamp(f"mean forgetting (loss rise on earlier domains) = {np.nanmean(final[:-1]-diag[:-1]):+.3f}")
stamp(f"subspace trace ratio captured by U (deep layers) = {tr_ratio:.3f}")
stamp(f"gate/route activation: mean fraction of deep-layer grad energy sent to null space = {np.mean(gate_pct):.2f}")
stamp("MEMORY (floats):  FD sketch = {:,}  |  full-cov OGD = {:,} ({:.1f}x FD)  |  AdamW state = {:,}"
      .format(mem_fd, mem_fullcov, mem_fullcov / max(mem_fd, 1), mem_adam))
if vram: stamp(f"peak VRAM per domain (MB): {[round(v) for v in vram]}")
out = dict(loss_matrix=lossM.tolist(), forgetting=float(np.nanmean(final[:-1]-diag[:-1])),
           k_per_layer=[ST[n]["k"] for n in order], trace_ratio=tr_ratio,
           route_energy=gate_pct, consec_sim=sims, vram_MB=vram,
           mem_floats=dict(fd=mem_fd, fullcov=mem_fullcov, adam=mem_adam))
json.dump(out, open("H2_H3_H12_results.json", "w"), indent=2)
stamp(f"ALL DONE in {time.time()-T0:.1f}s -> H2_H3_H12_results.json")
