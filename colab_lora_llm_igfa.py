# =====================================================================
# IGFA on sequential LoRA fine-tuning: BOTH regimes in one run.
#
# Paste this whole file into a single Kaggle/Colab GPU cell and run it. It loads a
# frozen EleutherAI/pythia-1b in 4-bit (device_map="auto" -> uses all visible GPUs),
# attaches LoRA, and compares naive/OGD/IGFA on:
#   (1) DISSIMILAR tasks (news, reviews, encyclopedia, tweets) -> IGFA reduces to OGD;
#   (2) SIMILAR tasks   (four related review corpora)          -> the gate can share.
# Prints a forgetting table for each, with a tqdm progress bar. Deps auto-install.
#
# Multi-GPU: device_map="auto" spreads the model across all GPUs. NOTE a 1B model
# in 4-bit (~0.7GB) fits one T4, so it may stay on GPU 0; pick a bigger MODEL_ID
# (e.g. "EleutherAI/pythia-6.9b") to actually use 2x T4. Per-layer state lives on
# each layer's device, so single- and multi-GPU are both correct.
# Fast iteration: MODEL_ID="gpt2", LOAD_4BIT=False (~15 min).
# =====================================================================

import subprocess, sys
def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)
try:
    import transformers, peft, datasets, tqdm  # noqa: F401
except ImportError:
    _pip("transformers", "peft", "datasets", "accelerate", "tqdm")

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_ID  = "EleutherAI/pythia-1b"   # billion-parameter default; "gpt2" for a fast 124M run
LOAD_4BIT = True                     # 4-bit base; set False for fp32 small models
SEQ_LEN, STEPS, LR = 128, 150, 2e-3  # fewer steps since the 1B base is slower
LORA_R, LORA_ALPHA = 16, 32
N_TASKS = 4
GAMMA, RANK_KEEP, SSTAR = 0.7, 32, 0.55     # recursive decay, kept dirs, overlap-gate threshold
torch.manual_seed(0); np.random.seed(0)

def lora_targets(mid):                # attention projection module names differ by architecture
    mid = mid.lower()
    if "gpt2" in mid:                      return ["c_attn"]
    if "pythia" in mid or "neox" in mid:   return ["query_key_value"]   # GPT-NeoX (Pythia)
    return ["q_proj", "v_proj"]                                         # Llama / TinyLlama
LORA_TARGETS = lora_targets(MODEL_ID)
if LOAD_4BIT:
    try:    import bitsandbytes  # noqa: F401
    except ImportError: _pip("bitsandbytes")

DISSIMILAR = [
    ("ag_news",         None,                "text",    "test"),   # news
    ("imdb",            None,                "text",    "test"),   # movie reviews
    ("wikitext",        "wikitext-2-raw-v1", "text",    "test"),   # encyclopedia
    ("tweet_eval",      "sentiment",         "text",    "test"),   # tweets
]
SIMILAR = [
    ("imdb",            None,                "text",    "test"),   # movie reviews
    ("rotten_tomatoes", None,                "text",    "test"),   # short film reviews
    ("yelp_polarity",   None,                "text",    "test"),   # business reviews
    ("amazon_polarity", None,                "content", "test"),   # product reviews
]
from datasets import load_dataset

# ---------------------------------------------------------------------
# Model + LoRA (loaded once; reused for both experiments)
# ---------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None: tok.pad_token = tok.eos_token
if LOAD_4BIT:
    from transformers import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4")
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    model = get_peft_model(base, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                                            target_modules=LORA_TARGETS, lora_dropout=0.0))
else:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(dev)
    model = get_peft_model(base, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                                            target_modules=LORA_TARGETS, lora_dropout=0.0)).to(dev)
model.config.use_cache = False
model.print_trainable_parameters()
EMB_DEV = model.get_input_embeddings().weight.device     # where inputs go

lora_layers = {n: m for n, m in model.named_modules()
               if n.endswith("lora_A.default") and isinstance(m, nn.Linear)}
in_dim = next(iter(lora_layers.values())).in_features
DEVOF = {n: m.weight.device for n, m in lora_layers.items()}   # per-layer device (multi-GPU safe)
print(f"{len(lora_layers)} LoRA-A layers, input dim {in_dim}, devices "
      f"{sorted({str(d) for d in DEVOF.values()})}")

captured = {}
for name, mod in lora_layers.items():
    mod.register_forward_pre_hook(
        lambda _, inp, nm=name: captured.__setitem__(
            nm, (inp[0] if isinstance(inp, tuple) else inp).detach().reshape(-1, in_dim).float()))

C_in   = {n: torch.zeros(in_dim, in_dim, device=DEVOF[n]) for n in lora_layers}   # recursive input cov
prot_U = {}

def eig_subspace(C, k=RANK_KEEP, tol=1e-2):
    w, V = torch.linalg.eigh(C); keep = w > tol * w.max().clamp(min=1e-12)
    idx = torch.where(keep)[0]
    return (V[:, idx[-k:]] if len(idx) else None)

# ---------------------------------------------------------------------
# Data and loss
# ---------------------------------------------------------------------
def _texts(name, config, field, split, n):
    ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    col = ds.select(range(min(len(ds), n * 12)))[field]
    return [t for t in col if isinstance(t, str) and len(t.strip()) > 64][:n]

def build_data(sources):
    def make(idx, split, n):
        name, config, field, test_split = sources[idx]
        sp = split if split == "train" else test_split
        try:    texts = _texts(name, config, field, sp, n)
        except Exception: texts = _texts(name, config, field, "train", n)
        enc = tok(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=SEQ_LEN)
        return enc["input_ids"], enc["attention_mask"]
    return ({t: make(t, "train", 1200) for t in range(N_TASKS)},
            {t: make(t, "test", 400)  for t in range(N_TASKS)})

def lm_loss(b, m):                                   # causal-LM loss, padding ignored, device-safe
    logits = model(input_ids=b, attention_mask=m).logits
    lab = b.clone(); lab[m == 0] = -100; lab = lab.to(logits.device)
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                           lab[:, 1:].reshape(-1), ignore_index=-100)

@torch.no_grad()
def eval_loss(test, t):
    ids, am = test[t]; tot, n = 0.0, 0
    for i in range(0, len(ids), 16):
        b = ids[i:i+16].to(EMB_DEV); m = am[i:i+16].to(EMB_DEV)
        tot += lm_loss(b, m).item() * len(b); n += len(b)
    return tot / n

@torch.no_grad()
def task_basis(train, t, n_batches=4):               # top-r input directions a task excites
    ids, am = train[t]; tmp = {nm: torch.zeros(in_dim, in_dim, device=DEVOF[nm]) for nm in lora_layers}
    for i in range(0, min(n_batches * 16, len(ids)), 16):
        model(input_ids=ids[i:i+16].to(EMB_DEV), attention_mask=am[i:i+16].to(EMB_DEV))
        for nm, x in captured.items():
            tmp[nm] += (x.T @ x) / max(1, len(x))
    return {nm: eig_subspace(tmp[nm]) for nm in lora_layers}

def proj_hard(mod, U):
    g = mod.weight.grad; Ug = U.to(g.dtype)
    mod.weight.grad = g - (g @ Ug) @ Ug.T

def reset_lora():
    for n, p in model.named_parameters():
        if "lora" in n: nn.init.zeros_(p) if "lora_B" in n else nn.init.normal_(p, std=0.02)

# ---------------------------------------------------------------------
# One method on one task stream
# ---------------------------------------------------------------------
def run(method, train, test, tag):
    reset_lora()
    for nm in C_in: C_in[nm].zero_()
    occupied = {nm: torch.zeros(in_dim, 0, device=DEVOF[nm]) for nm in lora_layers}
    opt = torch.optim.Adam([p for n, p in model.named_parameters() if p.requires_grad], lr=LR)
    hist = []
    for t in range(N_TASKS):
        if method == "ogd":
            for nm in lora_layers:
                prot_U[nm] = occupied[nm] if occupied[nm].shape[1] else None
        elif method == "igfa":
            geom = task_basis(train, t)
            for nm in lora_layers:
                U = eig_subspace(C_in[nm])
                if U is None or geom[nm] is None:
                    prot_U[nm] = None; continue
                overlap = (U.T @ geom[nm]).pow(2).sum().item() / U.shape[1]
                prot_U[nm] = None if overlap >= SSTAR else U
        else:
            for nm in lora_layers: prot_U[nm] = None
        ids, am = train[t]; model.train()
        for step in tqdm(range(STEPS), desc=f"{tag} | {method} | task {t}", leave=False):
            i = (step * 16) % (len(ids) - 16)
            b = ids[i:i+16].to(EMB_DEV); m = am[i:i+16].to(EMB_DEV)
            opt.zero_grad(); lm_loss(b, m).backward()
            for nm, mod in lora_layers.items():
                if mod.weight.grad is None: continue
                if method in ("ogd", "igfa"):
                    x = captured.get(nm)
                    if x is not None: C_in[nm] = GAMMA * C_in[nm] + (x.T @ x) / x.shape[0]
                    if prot_U[nm] is not None: proj_hard(mod, prot_U[nm])
            opt.step()
        if method == "ogd":
            geom = task_basis(train, t)
            for nm in lora_layers:
                if geom[nm] is None: continue
                occupied[nm] = torch.linalg.qr(torch.cat([occupied[nm], geom[nm]], 1)).Q
        model.eval(); hist.append([eval_loss(test, j) for j in range(N_TASKS)])
    A = np.array(hist); final = A[-1]; learned = np.array([A[t, t] for t in range(N_TASKS)])
    return float(final.mean()), float(np.mean(final[:-1] - learned[:-1]))

# ---------------------------------------------------------------------
# Run BOTH experiments
# ---------------------------------------------------------------------
for title, sources in [("DISSIMILAR tasks (gate should reduce to OGD)", DISSIMILAR),
                       ("SIMILAR tasks (gate can share and beat OGD)",   SIMILAR)]:
    print(f"\n========== {title} ==========")
    train, test = build_data(sources)
    rows = {}
    for m in ["naive", "ogd", "igfa"]:
        rows[m] = run(m, train, test, title.split()[0])
    print(f"{'method':12s}{'avg loss':>10s}{'forgetting':>12s}")
    for m in ["naive", "ogd", "igfa"]:
        print(f"{m:12s}{rows[m][0]:>10.3f}{rows[m][1]:>12.3f}")
# Read: DISSIMILAR -> OGD ~ IGFA < naive (no sharing to gain).
#       SIMILAR     -> IGFA <= OGD if the gate shares the structure OGD blocks.
