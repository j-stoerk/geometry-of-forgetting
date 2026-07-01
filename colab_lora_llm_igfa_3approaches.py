# %% [markdown]
# # Three new gates for IGFA on language (sequential LoRA, GPT-2)
#
# The input-subspace-overlap gate fails on language because every text task excites
# the same inputs. Each gate below replaces input overlap with the quantity the
# paper actually says governs interference -- and each is one line of the thesis:
#
#   A1 GAUSS-NEWTON GATE      "forgetting lives at the loss, so judge similarity at
#                              the output": protect the subspace of past *gradients*
#                              (output-weighted), not past inputs.  (Theorem 2)
#   A2 TASK-VECTOR GATE       "tasks conflict when they pull the weights the opposite
#                              way": block a direction only if the new update UNDOES a
#                              frozen pre-task reference of past learning. (merge identity)
#   A3 FUNCTIONAL CONTROLLER  "the forgetting formula already prices each direction --
#                              charge it": softly damp each direction in proportion to
#                              the forgetting energy it would cause; no threshold. (Theorem 1)
#
# Baselines: naive (no protection) and OGD (protect past INPUT subspace, hard).
# Run once on Kaggle/Colab GPU (~15-25 min).  Reproduces the comparison table.

# %%
# !pip install -q transformers peft datasets accelerate

import numpy as np, torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "gpt2"
LORA_TARGETS = ["c_attn"]
SEQ_LEN, STEPS, LR = 128, 250, 2e-3
LORA_R, LORA_ALPHA = 16, 32
N_TASKS = 4
GAMMA = 0.7          # recursive covariance decay (tracks drift)
RANK_KEEP = 32       # protected directions per layer
BETA = 9.0           # A3 damping strength: top-energy dir keeps ~1/(1+BETA)
METHODS = ["naive", "ogd", "gn", "taskvec", "functional"]
torch.manual_seed(0); np.random.seed(0)

# %% [markdown]
# ## Model + LoRA, four dissimilar corpora

# %%
tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
model = get_peft_model(base, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                                        target_modules=LORA_TARGETS, lora_dropout=0.0)).to(DEVICE)
model.print_trainable_parameters()

SOURCES_ALL = [
    ("ag_news",         None,                "text",    "test"),   # news
    ("imdb",            None,                "text",    "test"),   # movie reviews
    ("wikitext",        "wikitext-2-raw-v1", "text",    "test"),   # encyclopedia
    ("tweet_eval",      "sentiment",         "text",    "test"),   # tweets
    ("yelp_polarity",   None,                "text",    "test"),
    ("dbpedia_14",      None,                "content", "test"),
]
SOURCES = SOURCES_ALL[:N_TASKS]; TASKS = list(range(N_TASKS))

def _texts(name, config, field, split, n):
    ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    col = ds.select(range(min(len(ds), n * 12)))[field]
    return [t for t in col if isinstance(t, str) and len(t.strip()) > 64][:n]

def make_task(idx, split, n):
    name, config, field, test_split = SOURCES[idx]
    sp = split if split == "train" else test_split
    try:    texts = _texts(name, config, field, sp, n)
    except Exception: texts = _texts(name, config, field, "train", n)
    enc = tok(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=SEQ_LEN)
    return enc["input_ids"], enc["attention_mask"]

train_data = {t: make_task(t, "train", 1200) for t in TASKS}
test_data  = {t: make_task(t, "test", 400)  for t in TASKS}

def masked_labels(b, m):
    lab = b.clone(); lab[m == 0] = -100; return lab

@torch.no_grad()
def eval_loss(t):
    ids, am = test_data[t]; tot, n = 0.0, 0
    for i in range(0, len(ids), 16):
        b = ids[i:i+16].to(DEVICE); m = am[i:i+16].to(DEVICE)
        tot += model(input_ids=b, attention_mask=m, labels=masked_labels(b, m)).loss.item() * len(b); n += len(b)
    return tot / n

# %% [markdown]
# ## Per-layer state, covariances, subspaces, and the three projections

# %%
lora_layers = {n: m for n, m in model.named_modules()
               if n.endswith("lora_A.default") and isinstance(m, nn.Linear)}
in_dim = next(iter(lora_layers.values())).in_features
print(f"{len(lora_layers)} LoRA-A layers, input dim {in_dim}")

captured = {}                                            # forward inputs x per layer
for name, mod in lora_layers.items():
    mod.register_forward_pre_hook(
        lambda _, inp, nm=name: captured.__setitem__(
            nm, (inp[0] if isinstance(inp, tuple) else inp).detach().reshape(-1, in_dim).float()))

C_in = {n: torch.zeros(in_dim, in_dim, device=DEVICE) for n in lora_layers}   # input covariance
C_gn = {n: torch.zeros(in_dim, in_dim, device=DEVICE) for n in lora_layers}   # gradient (GN) covariance
prot_U, prot_lam, A_ref = {}, {}, {}

def eig_subspace(C, k=RANK_KEEP, tol=1e-2):
    w, V = torch.linalg.eigh(C)
    keep = w > tol * w.max().clamp(min=1e-12)
    idx = torch.where(keep)[0]
    if len(idx) == 0:
        return None, None
    idx = idx[-k:]
    return V[:, idx], w[idx]

def proj_hard(mod, U):                                   # OGD / A1: remove protected component
    g = mod.weight.grad; mod.weight.grad = g - (g @ U) @ U.T

def proj_signed(mod, U, R):                              # A2: block only directions that UNDO R
    g = mod.weight.grad; conflict = ((g @ U) * (R @ U)).sum(0) > 0
    if conflict.any():
        Uc = U[:, conflict]; mod.weight.grad = g - (g @ Uc) @ Uc.T

def proj_soft(mod, U, lam, beta):                        # A3: damp each dir by its forgetting energy
    g = mod.weight.grad
    keep = 1.0 / (1.0 + beta * (lam / lam.max().clamp(min=1e-12)))   # (k,)
    mod.weight.grad = g - ((g @ U) * (1.0 - keep)) @ U.T

# %% [markdown]
# ## Sequential training for each method

# %%
def reset_lora():
    for n, p in model.named_parameters():
        if "lora" in n: nn.init.zeros_(p) if "lora_B" in n else nn.init.normal_(p, std=0.02)

def run(method):
    reset_lora()
    for nm in lora_layers: C_in[nm].zero_(); C_gn[nm].zero_()
    opt = torch.optim.Adam([p for n, p in model.named_parameters() if p.requires_grad], lr=LR)
    acc = []
    for t in TASKS:
        # protection subspace at task start (from PAST tasks' accumulated geometry)
        for nm, mod in lora_layers.items():
            src = C_gn[nm] if method == "gn" else C_in[nm]
            prot_U[nm], prot_lam[nm] = eig_subspace(src)
            if method == "taskvec":
                A_ref[nm] = mod.weight.data.clone()       # FROZEN pre-task reference (fixes instability)
        ids, am = train_data[t]; model.train()
        for step in range(STEPS):
            i = (step * 16) % (len(ids) - 16)
            b = ids[i:i+16].to(DEVICE); m = am[i:i+16].to(DEVICE)
            opt.zero_grad()
            model(input_ids=b, attention_mask=m, labels=masked_labels(b, m)).loss.backward()
            for nm, mod in lora_layers.items():
                if mod.weight.grad is None: continue
                # recursive covariance update (raw grad / activations -> no extra forward)
                if method == "gn":
                    G = mod.weight.grad; C_gn[nm] = GAMMA * C_gn[nm] + (G.T @ G) / G.shape[0]
                elif method in ("ogd", "taskvec", "functional"):
                    x = captured.get(nm)
                    if x is not None: C_in[nm] = GAMMA * C_in[nm] + (x.T @ x) / x.shape[0]
                U = prot_U[nm]
                if U is None: continue
                if method in ("ogd", "gn"):  proj_hard(mod, U)
                elif method == "taskvec":    proj_signed(mod, U, A_ref[nm])
                elif method == "functional": proj_soft(mod, U, prot_lam[nm], BETA)
            opt.step()
        model.eval(); acc.append([eval_loss(tt) for tt in TASKS])
    A = np.array(acc); final = A[-1]; learned = np.array([A[t, t] for t in TASKS])
    return float(final.mean()), float(np.mean(final[:-1] - learned[:-1]))

results = {}
for m in METHODS:
    a, f = run(m); results[m] = (a, f)
    print(f"{m:11s}  avg final loss {a:.3f}   forgetting {f:.3f}")

print("\n=== Sequential LoRA, GPT-2, four dissimilar domains ===")
label = {"naive": "naive", "ogd": "OGD (input subspace)", "gn": "A1 Gauss-Newton gate",
         "taskvec": "A2 task-vector gate", "functional": "A3 functional controller"}
print(f"{'method':28s}{'avg loss':>10s}{'forgetting':>12s}")
for m in METHODS:
    print(f"{label[m]:28s}{results[m][0]:>10.3f}{results[m][1]:>12.3f}")
# A good gate should give forgetting <= OGD (ideally lower loss too, by sharing where
# it helps). naive is the no-protection ceiling. If a gate >> OGD it mis-fired; if it
# == OGD it protects unconditionally (safe but no sharing benefit).
