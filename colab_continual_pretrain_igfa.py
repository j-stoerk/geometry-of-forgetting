# =====================================================================
# Overlay-native CONTINUAL PRETRAINING with IGFA: the parts that need scale.
#
# Paste this whole file into a single Kaggle/Colab GPU cell and run it. It streams
# several text domains through a frozen LM + LoRA WITHOUT task boundaries and tests
# the framework's three "needs-scale" pieces in one shot:
#   #1  cheap online curvature  -> the protected subspace is tracked by FREQUENT
#                                  DIRECTIONS (O(l*d) memory, no d x d eig);
#   #2  overlay-native pretrain -> replay-free retention vs naive and vs small replay,
#                                  measuring where the capacity knee makes replay needed;
#   #3  pre-flight diagnostic   -> a frozen-feature domain-probe estimates I(T;phi)
#                                  before training and forecasts solvability.
#
# Defaults: EleutherAI/pythia-1b in 4-bit, device_map="auto" (uses all GPUs), tqdm,
# auto-install. Fast mode: MODEL_ID="gpt2", LOAD_4BIT=False. ~45-90 min on a T4.
# =====================================================================

import subprocess, sys
def _pip(*p): subprocess.run([sys.executable, "-m", "pip", "install", "-q", *p], check=True)
try:    import transformers, peft, datasets, sklearn, tqdm  # noqa: F401
except ImportError: _pip("transformers", "peft", "datasets", "accelerate", "scikit-learn", "tqdm")

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression

MODEL_ID  = "EleutherAI/pythia-1b"
LOAD_4BIT = True
SEQ_LEN, STEPS_PER_DOMAIN, LR = 128, 150, 2e-3
LORA_R, LORA_ALPHA = 16, 32
RANK_KEEP, FD_ELL, UPDATE_EVERY = 24, 48, 25   # protected rank, FD sketch rows, U refresh period
REPLAY_PER_DOMAIN = 16
torch.manual_seed(0); np.random.seed(0)

DOMAINS = [   # streamed in order, no boundaries given to the algorithm
    ("ag_news",         None,                "text",    "test"),
    ("imdb",            None,                "text",    "test"),
    ("wikitext",        "wikitext-2-raw-v1", "text",    "test"),
    ("tweet_eval",      "sentiment",         "text",    "test"),
    ("yelp_polarity",   None,                "text",    "test"),
    ("dbpedia_14",      None,                "content", "test"),
]
N = len(DOMAINS)

def lora_targets(mid):
    mid = mid.lower()
    if "gpt2" in mid:                     return ["c_attn"]
    if "pythia" in mid or "neox" in mid:  return ["query_key_value"]
    return ["q_proj", "v_proj"]
if LOAD_4BIT:
    try: import bitsandbytes  # noqa
    except ImportError: _pip("bitsandbytes")

# --------------------------- model + LoRA ---------------------------
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None: tok.pad_token = tok.eos_token
if LOAD_4BIT:
    from transformers import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    model = get_peft_model(base, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                                            target_modules=lora_targets(MODEL_ID), lora_dropout=0.0))
else:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(dev)
    model = get_peft_model(base, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                                            target_modules=lora_targets(MODEL_ID), lora_dropout=0.0)).to(dev)
model.config.use_cache = False
EMB_DEV = model.get_input_embeddings().weight.device
lora_layers = {n: m for n, m in model.named_modules()
               if n.endswith("lora_A.default") and isinstance(m, nn.Linear)}
in_dim = next(iter(lora_layers.values())).in_features
DEVOF = {n: m.weight.device for n, m in lora_layers.items()}
print(f"{len(lora_layers)} LoRA-A layers, dim {in_dim}, devices {sorted({str(d) for d in DEVOF.values()})}")

captured = {}
for name, mod in lora_layers.items():
    mod.register_forward_pre_hook(lambda _, inp, nm=name: captured.__setitem__(
        nm, (inp[0] if isinstance(inp, tuple) else inp).detach().reshape(-1, in_dim).float()))

# --------------------------- data + loss ---------------------------
def _texts(name, cfg, field, split, n):
    ds = load_dataset(name, cfg, split=split) if cfg else load_dataset(name, split=split)
    col = ds.select(range(min(len(ds), n * 12)))[field]
    return [t for t in col if isinstance(t, str) and len(t.strip()) > 64][:n]

def make(idx, split, n):
    name, cfg, field, ts = DOMAINS[idx]; sp = split if split == "train" else ts
    try:    txt = _texts(name, cfg, field, sp, n)
    except Exception: txt = _texts(name, cfg, field, "train", n)
    enc = tok(txt, return_tensors="pt", padding="max_length", truncation=True, max_length=SEQ_LEN)
    return enc["input_ids"], enc["attention_mask"]
train = {t: make(t, "train", 1000) for t in range(N)}
test  = {t: make(t, "test", 300)  for t in range(N)}

def lm_loss(b, m):
    logits = model(input_ids=b, attention_mask=m).logits
    lab = b.clone(); lab[m == 0] = -100; lab = lab.to(logits.device)
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), lab[:, 1:].reshape(-1), ignore_index=-100)

@torch.no_grad()
def eval_loss(t):
    ids, am = test[t]; tot, n = 0.0, 0
    for i in range(0, len(ids), 16):
        b = ids[i:i+16].to(EMB_DEV); m = am[i:i+16].to(EMB_DEV)
        tot += lm_loss(b, m).item() * len(b); n += len(b)
    return tot / n

# --------------------------- #3 pre-flight diagnostic ---------------------------
@torch.no_grad()
def feasibility_diagnostic(per=200):
    feats, labs = [], []
    for t in range(N):
        ids, am = train[t]
        for i in range(0, min(per, len(ids)), 16):
            b = ids[i:i+16].to(EMB_DEV); m = am[i:i+16].to(EMB_DEV)
            h = model.get_input_embeddings()(b)                      # cheap frozen feature: mean token embedding
            feats.append((h * m.unsqueeze(-1)).sum(1).div(m.sum(1, keepdim=True)).float().cpu().numpy())
            labs.append(np.full(len(b), t))
    X = np.concatenate(feats); y = np.concatenate(labs)
    idx = np.random.permutation(len(X)); tr, te = idx[:len(X)//2], idx[len(X)//2:]
    acc = LogisticRegression(max_iter=300, multi_class="multinomial").fit(X[tr], y[tr]).score(X[te], y[te])
    # multiclass MI estimate via Fano-type relation (nats)
    I_est = np.log(N) * acc            # crude: fraction of max task-entropy recovered
    print(f"[diagnostic] {N}-way domain-probe accuracy {acc:.3f}  ->  I_est ~ {I_est:.3f} / {np.log(N):.3f} nats "
          f"({'high -> tasks separable, replay-free feasible' if acc > 0.6 else 'low -> hard, expect a floor'})")
    return acc

# --------------------------- #1 Frequent Directions subspace ---------------------------
FD = {nm: torch.zeros(FD_ELL, in_dim, device=DEVOF[nm]) for nm in lora_layers}   # l x d sketch
prot_U = {nm: None for nm in lora_layers}

def fd_insert(nm, rows):                 # add a few activation rows; densify when full
    B = FD[nm]
    for x in rows:
        z = (~B.any(dim=1)).nonzero()
        if len(z) == 0:
            U, s, Vt = torch.linalg.svd(B, full_matrices=False)
            delta = s[FD_ELL // 2] ** 2
            B = torch.diag(torch.sqrt(torch.clamp(s ** 2 - delta, min=0))) @ Vt
            FD[nm] = B; z = (~B.any(dim=1)).nonzero()
        B[z[0, 0]] = x

def refresh_U():
    for nm in lora_layers:
        Vt = torch.linalg.svd(FD[nm], full_matrices=False)[2]
        prot_U[nm] = Vt[:RANK_KEEP].T.contiguous()

def proj_hard(mod, U):
    g = mod.weight.grad; Ug = U.to(g.dtype); mod.weight.grad = g - (g @ Ug) @ Ug.T

def reset_lora():
    for n, p in model.named_parameters():
        if "lora" in n: nn.init.zeros_(p) if "lora_B" in n else nn.init.normal_(p, std=0.02)

# --------------------------- stream training ---------------------------
def run(method):     # naive | replay | igfa  (igfa = FD-tracked recursive projection, replay-free)
    reset_lora()
    for nm in FD: FD[nm].zero_(); prot_U[nm] = None
    bufX, bufY = [], []
    opt = torch.optim.Adam([p for n, p in model.named_parameters() if p.requires_grad], lr=LR)
    forget0 = []                                          # loss on domain 0 as the stream grows
    base0 = None
    for t in range(N):
        ids, am = train[t]; model.train()
        for step in tqdm(range(STEPS_PER_DOMAIN), desc=f"{method} | domain {t}", leave=False):
            i = (step * 16) % (len(ids) - 16)
            if method == "replay" and bufX:
                bx = torch.cat([ids[i:i+16]] + [b for b in bufX]); bm = torch.cat([am[i:i+16]] + [b for b in bufY])
                sel = torch.randperm(len(bx))[:16]; b = bx[sel].to(EMB_DEV); m = bm[sel].to(EMB_DEV)
            else:
                b = ids[i:i+16].to(EMB_DEV); m = am[i:i+16].to(EMB_DEV)
            opt.zero_grad(); lm_loss(b, m).backward()
            if method == "igfa":
                for nm, mod in lora_layers.items():
                    x = captured.get(nm)
                    if x is not None: fd_insert(nm, x[torch.randperm(len(x))[:4]])   # FD streaming update
                    if prot_U[nm] is not None and mod.weight.grad is not None: proj_hard(mod, prot_U[nm])
                if step % UPDATE_EVERY == 0: refresh_U()
            opt.step()
        if method == "replay":
            sel = torch.randperm(len(ids))[:REPLAY_PER_DOMAIN]; bufX.append(ids[sel]); bufY.append(am[sel])
        model.eval()
        l0 = eval_loss(0)
        if base0 is None: base0 = l0
        forget0.append(l0 - base0)                        # rise in domain-0 loss since it was learned
    final = np.array([eval_loss(t) for t in range(N)])
    return float(final.mean()), forget0

# --------------------------- run everything ---------------------------
feasibility_diagnostic()
print(f"\n{'method':8s}{'avg final loss':>16s}{'domain-0 forgetting curve (per domain seen)':>48s}")
for m in ["naive", "replay", "igfa"]:
    avg, f0 = run(m)
    print(f"{m:8s}{avg:>16.3f}   " + " ".join(f"{v:+.2f}" for v in f0))
# Read: igfa (replay-free, FD-tracked subspace) should keep domain-0 forgetting low
# without a buffer, until the capacity knee; replay trades a buffer for retention;
# naive forgets fastest. The diagnostic predicts whether the stream is separable.
