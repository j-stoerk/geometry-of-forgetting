# =====================================================================
#  kaggle_interference_optimizer.py (v3) -- Q12 at scale: is the practical
#  successor to the igfa gate an OPTIMIZER?
#
#  LESSONS FROM v1/v2 (kept for honesty): single-hop text fine-tuning at
#  LoRA scale does NOT forget -- both an AG-News topic pair and a
#  news->IMDB register clash produced forgetting <= 0 (pure backward
#  transfer), making any controller comparison vacuous.  The regime the
#  paper measured as forgetting (Table S3: 0.64-1.21) is the FOUR-domain
#  sequential stream under Adam.  v3 benchmarks there.
#
#  Design: GPT-2 + manual LoRA, sequential stream news -> imdb -> wikitext
#  -> tweets.  Arm 1: tuned Adam schedules (kind x peak grid).  Arm 2: the
#  SAME Adam at fixed lr plus the interference penalty IN THE LOSS,
#  L_t + mu/2 sum_{u<t} ||V_u (theta - theta_u)||^2_{s2u}, with per-domain
#  Frequent-Directions sketches of LoRA gradients (so any base optimizer
#  works).  Pre-flight verifies the stream actually forgets before the
#  sweep.  Frontier = (forgetting after full stream, avg final loss);
#  compared on common support, both axes.
#
#  Kaggle 2x T4, ~1 h with SEEDS=[0,1].  Progress -> file (never prints
#  from worker threads); results dumped incrementally.
# =====================================================================
import os, json, time
import numpy as np
import torch, torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, wait as fwait

SEEDS      = [0, 1]
STEPS      = 250              # per domain per configuration
BS, SEQ    = 8, 128
RANK_LORA  = 8
ELL        = 16               # FD sketch rows per (param, domain)
MU_GRID    = [0.3, 3.0, 30.0, 300.0]
SCHEDULES  = [(k, p) for k in ["constant", "cosine"] for p in [1e-4, 3e-4, 1e-3]]
PEN_SCHEDULES = [(k, p) for k in ["constant", "cosine"] for p in [3e-4, 1e-3]]
BASE_LR    = 3e-4             # fixed lr for the interference arm
DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
def tlog(m):
    with open("interference_opt_progress.log", "a") as f:
        f.write(f"[{time.time()-T0:7.1f}s] {m}\n")


class InterferenceSGD(torch.optim.Optimizer):
    """Reference artifact: SGD with the sketched interference penalty applied in
    the update (see paper Sec. 'Memory systems and control').  The benchmark
    below applies the identical penalty in the LOSS instead, so that any base
    optimizer (here Adam) can carry it; the two are equivalent for SGD."""
    def __init__(self, params, lr, mu=0.0):
        super().__init__(params, dict(lr=lr, mu=mu))
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                g = p.grad
                st = self.state.get(p, {})
                if group["mu"] > 0 and "V" in st:
                    dp = (p - st["ref"]).flatten()
                    g = g + group["mu"] * (st["V"].T @ (st["s2"] * (st["V"] @ dp))).view_as(p)
                p.add_(g, alpha=-group["lr"])
    def set_metric(self, p, V, s2, ref):
        self.state[p] = dict(V=V, s2=s2, ref=ref.detach().clone())


def fd_sketch(G, ell):
    sk = np.zeros((ell, G.shape[1]), dtype=np.float32)
    for i in range(0, len(G), ell):
        B = np.vstack([sk, G[i:i + ell]])
        _, s, Vt = np.linalg.svd(B, full_matrices=False)
        delta = s[ell - 1] ** 2 if len(s) >= ell else 0.0
        sk = (np.sqrt(np.maximum(s[:ell] ** 2 - delta, 0))[:, None] * Vt[:ell]).astype(np.float32)
    _, s, Vt = np.linalg.svd(sk, full_matrices=False)
    return Vt[:ell].astype(np.float32), (s[:ell] ** 2).astype(np.float32)


class LoRA(nn.Module):
    def __init__(self, base, d_in, d_out, r, device):
        super().__init__(); self.base, self.s = base, 2.0 / r
        self.A = nn.Parameter(torch.zeros(d_in, r, device=device))
        self.B = nn.Parameter(torch.zeros(r, d_out, device=device))
        nn.init.normal_(self.A, std=0.02)
    def forward(self, x): return self.base(x) + (x @ self.A) @ self.B * self.s

def build_model(device):
    from transformers import GPT2LMHeadModel
    m = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    for p in m.parameters(): p.requires_grad_(False)
    loras = []
    for blk in m.transformer.h:
        l = LoRA(blk.attn.c_attn, 768, 2304, RANK_LORA, device)
        blk.attn.c_attn = l; loras.append(l)
    return m, loras

def get_domains(tok):
    """news -> imdb -> wikitext -> tweets: the Table-S3 regime that forgets."""
    from datasets import load_dataset
    chunks = []
    news = load_dataset("ag_news", split="train")
    texts = ["\n\n".join([x["text"] for x in news if x["label"] == 0][:1500])]
    imdb = load_dataset("imdb", split="train")
    texts.append("\n\n".join([x["text"] for x in imdb][:600]))
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts.append("\n\n".join([x["text"] for x in wiki if len(x["text"]) > 200][:1200]))
    tw = load_dataset("tweet_eval", "sentiment", split="train")
    texts.append("\n\n".join([x["text"] for x in tw][:4000]))
    for t in texts:
        enc = tok(t, return_tensors="pt").input_ids[0]
        ch = enc[: (len(enc) // SEQ) * SEQ].view(-1, SEQ)
        chunks.append((ch[:110], ch[110:135]))
    return chunks

def lm_loss(model, ids, device): return model(ids.to(device), labels=ids.to(device)).loss

@torch.no_grad()
def heldout(model, chunks, device):
    model.eval()
    v = float(np.mean([lm_loss(model, chunks[i:i+BS], device).item()
                       for i in range(0, len(chunks), BS)]))
    model.train(); return v


def run_seed(seed, device, doms):
    torch.manual_seed(seed)
    model, loras = build_model(device)
    params = [p for l in loras for p in (l.A, l.B)]

    def train_domain(tr, steps, lr_fn, penalty=None):
        opt = torch.optim.Adam(params, lr=lr_fn(0))
        for k in range(steps):
            for gset in opt.param_groups: gset["lr"] = lr_fn(k)
            i = torch.randint(0, len(tr) - BS, (1,)).item()
            loss = lm_loss(model, tr[i:i + BS], device)
            if penalty is not None: loss = loss + penalty()
            opt.zero_grad(); loss.backward(); opt.step()

    def stream(lr_fn, mu=0.0):
        """Run the full 4-domain stream; returns (forgetting, avg final loss)."""
        # domain 0 is always plain (same for both arms)
        for p, s0 in zip(params, snap_d0):
            with torch.no_grad(): p.copy_(s0)
        sketches = dict(d0_metrics)                            # {(param_idx, dom): (V, s2, ref)}
        diag = [held_d0]
        for t in range(1, 4):
            if mu > 0:
                mats = [(pi, dom, torch.tensor(V, device=device), torch.tensor(s2, device=device), ref.to(device))
                        for (pi, dom), (V, s2, ref) in sketches.items()]
                def penalty():
                    tot = 0.0
                    for pi, dom, V, s2, ref in mats:
                        dp = (params[pi] - ref).flatten()
                        tot = tot + 0.5 * mu * torch.sum(s2 * (V @ dp) ** 2)
                    return tot
            else:
                penalty = None
            train_domain(doms[t][0], STEPS, lr_fn, penalty)
            diag.append(heldout(model, doms[t][1], device))
            if mu > 0 and t < 3:                # sketches are only consumed by the penalty arm
                grads = {pi: [] for pi in range(len(params))}
                for k in range(24):
                    i = torch.randint(0, len(doms[t][0]) - BS, (1,)).item()
                    loss = lm_loss(model, doms[t][0][i:i+BS], device)
                    model.zero_grad(); loss.backward()
                    for pi, p in enumerate(params):
                        grads[pi].append(p.grad.detach().flatten().cpu().numpy())
                for pi in grads:
                    V, s2 = fd_sketch(np.stack(grads[pi]), ELL)
                    sketches[(pi, t)] = (V, s2, params[pi].detach().clone())
        finals = [heldout(model, doms[i][1], device) for i in range(4)]
        fgt = float(np.mean([finals[i] - diag[i] for i in range(3)]))
        return fgt, float(np.mean(finals))

    # ---- shared domain-0 training + its sketch ----
    train_domain(doms[0][0], STEPS, lambda k: 3e-4)
    held_d0 = heldout(model, doms[0][1], device)
    snap_d0 = [p.detach().clone() for p in params]
    grads = {pi: [] for pi in range(len(params))}
    for k in range(24):
        i = torch.randint(0, len(doms[0][0]) - BS, (1,)).item()
        loss = lm_loss(model, doms[0][0][i:i+BS], device)
        model.zero_grad(); loss.backward()
        for pi, p in enumerate(params):
            grads[pi].append(p.grad.detach().flatten().cpu().numpy())
    d0_metrics = {}
    for pi in grads:
        V, s2 = fd_sketch(np.stack(grads[pi]), ELL)
        d0_metrics[(pi, 0)] = (V, s2, params[pi].detach().clone())
    # ---- pre-flight: does this stream forget at all? ----
    fgt0, _ = stream(lambda k: 1e-3, mu=0.0)
    tlog(f"seed{seed} PRE-FLIGHT naive Adam@1e-3 stream forgetting: {fgt0:+.3f}")
    if fgt0 < 0.05:
        tlog(f"seed{seed} WARNING: stream shows no interference; comparison will be vacuous.")
    # ---- arm 1: tuned schedules (dict entries, same schema as arm 2) ----
    frontier = {"schedule": [], "interference": []}
    def mk_lr(kind, peak):
        return ((lambda k: peak) if kind == "constant"
                else (lambda k: peak * 0.5 * (1 + np.cos(np.pi * k / STEPS))))
    for kind, peak in SCHEDULES:
        forget, avg = stream(mk_lr(kind, peak), mu=0.0)
        frontier["schedule"].append(dict(schedule=kind, peak_lr=float(peak), mu=0.0,
                                         forget=float(forget), avg=float(avg)))
        tlog(f"seed{seed} schedule {kind}/{peak:g}: forget {forget:+.3f} avg {avg:.3f}")
    # ---- arm 2: interference penalty, base lr tuned symmetrically ----
    for kind, peak in PEN_SCHEDULES:
        for mu in MU_GRID:
            forget, avg = stream(mk_lr(kind, peak), mu=mu)
            frontier["interference"].append(dict(schedule=kind, peak_lr=float(peak),
                                                 mu=float(mu), forget=float(forget),
                                                 avg=float(avg)))
            tlog(f"seed{seed} penalty {kind}/{peak:g} mu={mu:g}: forget {forget:+.3f} avg {avg:.3f}")
    del model; torch.cuda.empty_cache()
    return frontier


if __name__ == "__main__":
    log(f"devices: {DEVICES}")
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    doms = get_domains(tok)
    log("domains ready: news -> imdb -> wikitext -> tweets")
    per = {}
    with ThreadPoolExecutor(len(DEVICES)) as ex:
        futs = {ex.submit(run_seed, s, DEVICES[i % len(DEVICES)], doms): s
                for i, s in enumerate(SEEDS)}
        pending = set(futs)
        while pending:
            done, pending = fwait(pending, timeout=120)
            for f in done: per[futs[f]] = f.result()
            log(f"{len(SEEDS)-len(pending)}/{len(SEEDS)} seeds done "
                f"(details: interference_opt_progress.log)")
            with open("interference_opt_results.json", "w") as f:
                json.dump({str(k): v for k, v in per.items()}, f, indent=1)
    # ---------------- inlined shape-agnostic report (self-contained) ----------------
    def to_xy(entries):
        return np.array(sorted((float(e["forget"]), float(e["avg"])) for e in entries))
    def pareto(pts):
        keep = []
        for i, (f, a) in enumerate(pts):
            if not any((pts[j, 0] <= f and pts[j, 1] <= a and j != i
                        and (pts[j, 0] < f or pts[j, 1] < a)) for j in range(len(pts))):
                keep.append((f, a))
        return np.array(sorted(keep))
    print("\n===== frontier comparison (common support, both axes) =====")
    for s in SEEDS:
        fs, fi = to_xy(per[s]["schedule"]), to_xy(per[s]["interference"])
        print(f"  seed {s}: schedule forgetting [{fs[:,0].min():+.3f},{fs[:,0].max():+.3f}] "
              f"avg [{fs[:,1].min():.3f},{fs[:,1].max():.3f}] | penalty forgetting "
              f"[{fi[:,0].min():+.3f},{fi[:,0].max():+.3f}] avg [{fi[:,1].min():.3f},{fi[:,1].max():.3f}]")
    print("  --- avg final loss at matched forgetting (percent of common range) ---")
    for frac in [0.25, 0.5, 0.75]:
        ls, li = [], []
        for s in SEEDS:
            fs, fi = to_xy(per[s]["schedule"]), to_xy(per[s]["interference"])
            lo = max(fs[:, 0].min(), fi[:, 0].min()); hi = min(fs[:, 0].max(), fi[:, 0].max())
            if hi <= lo: continue
            f = lo + frac * (hi - lo)
            ls.append(np.interp(f, fs[:, 0], fs[:, 1])); li.append(np.interp(f, fi[:, 0], fi[:, 1]))
        if ls:
            print(f"    at {int(frac*100)}%: tuned-Adam {np.mean(ls):.3f}   penalty "
                  f"{np.mean(li):.3f}   ({100*(np.mean(ls)-np.mean(li))/np.mean(ls):+.2f}%, n={len(ls)})")
        else:
            print(f"    at {int(frac*100)}%: no common support")
    print("  --- forgetting at matched avg final loss ---")
    for off in [0.02, 0.05, 0.1]:
        gs, gi = [], []
        for s in SEEDS:
            fs, fi = to_xy(per[s]["schedule"]), to_xy(per[s]["interference"])
            fs = fs[np.argsort(fs[:, 1])]; fi = fi[np.argsort(fi[:, 1])]
            b = max(fs[:, 1].min(), fi[:, 1].min()) + off
            if b > min(fs[:, 1].max(), fi[:, 1].max()): continue
            gs.append(np.interp(b, fs[:, 1], fs[:, 0])); gi.append(np.interp(b, fi[:, 1], fi[:, 0]))
        if gs:
            print(f"    at min avg-loss+{off:.2f} (n={len(gs)}): tuned-Adam {np.mean(gs):+.3f}   "
                  f"penalty {np.mean(gi):+.3f}")
    print("  --- per-seed Pareto fronts (forgetting, avg loss) ---")
    for s in SEEDS:
        print(f"    seed {s} schedule : " + "  ".join(
            f"({f:+.3f},{a:.3f})" for f, a in pareto(to_xy(per[s]["schedule"]))))
        print(f"    seed {s} penalty  : " + "  ".join(
            f"({f:+.3f},{a:.3f})" for f, a in pareto(to_xy(per[s]["interference"]))))
        best = min(per[s]["interference"], key=lambda e: e["avg"] + max(e["forget"], 0))
        print(f"    seed {s} best penalty config: {best['schedule']}/{best['peak_lr']:g} "
              f"mu={best['mu']:g} -> forget {best['forget']:+.3f} avg {best['avg']:.3f}")
    log("ALL DONE -> interference_opt_results.json")