# =====================================================================
#  kaggle_jspace_interference.py -- BRIDGING EXPERIMENT: does measured
#  forgetting concentrate in the Jacobian-lens J-space?
#
#  Two averaged-Jacobian theories meet here.  The J-lens of Gurnee,
#  Lindsey et al. (2026) reads a residual-stream activation by its
#  context-AVERAGED forward Jacobian  J_l = E[ d h_final / d h_l ]
#  composed with the unembedding, and calls the low-rank span of the
#  active readout directions the J-space (the "verbalizable" subspace).
#  This paper's interference geometry reads a task by its data-AVERAGED
#  Gauss-Newton curvature Sigma_A, and calls range(Sigma_A) the
#  directions whose perturbation costs forgetting (ker(Sigma_A) free).
#
#  HYPOTHESIS: these are the same functional subspace.  Then forgetting
#  of task A caused by learning B should live in J-space:
#    (H1 decomposition) the readout change on A projected onto J-space,
#        P_J . Delta f, should recover ~all of the measured forgetting,
#        while the orthogonal part (I - P_J).Delta f contributes ~0.
#    (H2 intervention)  protecting J-space -- projecting the LoRA update
#        so the induced Delta f stays orthogonal to A's J-space -- should
#        cut forgetting nearly as much as protecting Sigma_A's own top
#        subspace, and far more than naive; a random subspace of equal
#        rank should NOT (specificity control).
#
#  Setup: frozen pythia-410m; the J-lens supplies P_J at the readout
#  layer; a LoRA head learns the 4-domain stream; forgetting on earlier
#  domains is decomposed and then gated by P_J.  ~40 min on 1x T4.
#
#  Needs the released lens package:  pip install jlens  (github
#  anthropics/jacobian-lens); falls back to fitting a lens from ~200
#  corpus prompts if no pretrained lens for the model is available.
# =====================================================================
import json, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL = "EleutherAI/pythia-410m"
LAYER = None            # readout layer for the lens/decomposition; None -> ~2/3 depth
RANK_J = 64             # dimension of the J-space / protected subspaces compared
STEPS, BS, SEQ, LR = 250, 8, 128, 1e-4
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
if DEV.startswith("cuda") and torch.cuda.get_device_capability(0) < (7, 0):
    raise SystemExit("Unsupported GPU (sm<70): set Accelerator to 'GPU T4 x2'.")
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


# ---- J-space basis from the released Jacobian lens -------------------
def jspace_basis(model, tok, layer, contexts, rank):
    """Top-`rank` principal subspace of the J-lens readout map active on
    `contexts`.  Returns an orthonormal (d_model x rank) basis P_J whose
    span is the residual-stream subspace that the averaged forward
    Jacobian maps to output tokens actually predicted on these contexts."""
    try:
        import jlens
    except Exception as e:
        raise SystemExit("pip install jlens  (github anthropics/jacobian-lens)") from e
    jm = jlens.from_hf(model, tok)
    try:                                                   # a pretrained lens if one exists
        lens = jlens.JacobianLens.from_pretrained("anthropics/jacobian-lens",
                                                   filename=f"{MODEL.split('/')[-1]}/lens.pt")
    except Exception:
        log("no pretrained lens for this model; fitting one (~200 prompts)")
        lens = jlens.fit(jm, prompts=contexts[:200])
    Jl = lens.jacobian(layer).to(DEV).float()              # (d_model x d_model) averaged Jacobian
    WU = model.get_output_embeddings().weight.detach().float()   # (V x d_model)
    # readout map on this layer:  logits ~ WU @ (Jl @ h).  Collect the rows
    # of (WU @ Jl) for the tokens actually predicted on the contexts, and
    # take their top-`rank` right-singular directions = the active J-space.
    with torch.no_grad():
        active = set()
        for ids in contexts[:64]:
            h = hidden_at(model, ids.to(DEV), layer)       # (T x d_model)
            top = (h @ (WU @ Jl).T).topk(5, -1).indices    # top predicted tokens per pos
            active.update(top.flatten().tolist())
        rows = (WU[list(active)] @ Jl)                     # (|active| x d_model)
        Vt = torch.linalg.svd(rows, full_matrices=False)[2]
    return Vt[:rank].T.contiguous()                        # (d_model x rank)


# ---- model plumbing --------------------------------------------------
def build(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(DEV)
    m.config.use_cache = False
    for p in m.parameters(): p.requires_grad_(False)
    return m, tok


def hidden_at(model, ids, layer):
    ids = ids.view(1, -1) if ids.ndim == 1 else ids
    hs = model(ids, attention_mask=torch.ones_like(ids), output_hidden_states=True).hidden_states
    return hs[layer][0]


class Head(nn.Module):
    """Trainable low-rank readout perturbation at `layer`: the free
    parameters whose movement we decompose against P_J.  Delta f is the
    induced change in the layer-`layer` residual stream."""
    def __init__(self, d, r=16):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(d, r)); self.B = nn.Parameter(torch.randn(r, d) * 1e-3)
    def delta(self, h): return (h @ self.A) @ self.B


def get_domains(tok):
    from datasets import load_dataset
    out = []
    for name, ds, filt, n in [
        ("news", load_dataset("ag_news", split="train"), lambda x: x["label"] == 0, 1500),
        ("imdb", load_dataset("imdb", split="train"), lambda x: True, 700),
        ("wiki", load_dataset("wikitext", "wikitext-2-raw-v1", split="train"),
         lambda x: len(x["text"]) > 200, 1400),
        ("tweets", load_dataset("tweet_eval", "sentiment", split="train"), lambda x: True, 4500)]:
        ids = tok("\n\n".join([x["text"] for x in ds if filt(x)][:n]),
                  return_tensors="pt").input_ids[0]
        ch = ids[: (len(ids) // SEQ) * SEQ].view(-1, SEQ)
        out.append((name, ch[:110], ch[110:134]))
    return out


if __name__ == "__main__":
    model, tok = build(MODEL)
    d = model.config.hidden_size
    layer = LAYER or (model.config.num_hidden_layers * 2 // 3)
    doms = get_domains(tok)
    log(f"model {MODEL}, readout layer {layer}, d={d}")

    # J-space basis on task A (domain 0) held-out contexts
    A_ctx = [doms[0][2][i] for i in range(len(doms[0][2]))]
    P_J = jspace_basis(model, tok, layer, A_ctx, RANK_J)
    log(f"J-space basis: {P_J.shape}")

    # Sigma_A top-RANK_J subspace (this paper's native protected subspace) on A
    with torch.no_grad():
        H = torch.cat([hidden_at(model, doms[0][2][i].to(DEV), layer) for i in range(24)])
        C = H.T @ H / len(H)
        P_S = torch.linalg.eigh(C)[1][:, -RANK_J:]         # top-RANK_J eigenvectors
    P_R = torch.linalg.qr(torch.randn(d, RANK_J, device=DEV))[0]   # random control
    overlap = float((P_J.T @ P_S).square().sum() / RANK_J)
    log(f"principal overlap  J-space vs Sigma_A-subspace = {overlap:.3f} "
        f"(random baseline ~ {RANK_J/d:.3f})")

    # ---- H1: decompose measured forgetting by J-space projection ------
    # Learn domains 1..3 with a naive head; measure Delta f on A at the
    # readout layer and split energy into P_J vs orthogonal.
    def readout_ce(head, dom):
        te = dom[1] if len(dom) == 2 else dom[2]
        tot = 0.0
        for i in range(0, len(te), 4):
            ids = te[i:i+4].to(DEV)
            hs = model(ids, attention_mask=torch.ones_like(ids),
                       output_hidden_states=True).hidden_states
            h = hs[layer] + (head.delta(hs[layer]) if head else 0)
            # cheap readout proxy: re-inject and run remaining layers is costly;
            # use the model's own head on the perturbed final state approximation
            logits = model.get_output_embeddings()(model.gpt_neox.final_layer_norm(h))
            tot += float(F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                         ids[:, 1:].reshape(-1))) * len(ids)
        return tot / len(te)

    head = Head(d).to(DEV)
    opt = torch.optim.Adam([head.A, head.B], lr=LR)
    base_A = readout_ce(head, doms[0])
    with torch.no_grad():
        f0 = torch.cat([head.delta(hidden_at(model, doms[0][2][i].to(DEV), layer))
                        for i in range(24)])
    for t in (1, 2, 3):
        for _ in range(STEPS):
            ids = doms[t][1][random.randint(0, len(doms[t][1]) - BS):][:BS].to(DEV)
            hs = model(ids, attention_mask=torch.ones_like(ids),
                       output_hidden_states=True).hidden_states
            h = hs[layer] + head.delta(hs[layer])
            logits = model.get_output_embeddings()(model.gpt_neox.final_layer_norm(h))
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                   ids[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        f1 = torch.cat([head.delta(hidden_at(model, doms[0][2][i].to(DEV), layer))
                        for i in range(24)])
        df = f1 - f0
        dfJ = df @ P_J @ P_J.T
        e_tot = float(df.square().sum()); e_J = float(dfJ.square().sum())
    fgt = readout_ce(head, doms[0]) - base_A
    log(f"H1: forgetting on A = {fgt:+.4f} nats;  ||Delta f||^2 in J-space "
        f"= {100*e_J/max(e_tot,1e-12):.0f}%  (random subspace ~ {100*RANK_J/d:.0f}%)")

    # ---- H2: protect J-space vs Sigma-space vs random vs naive --------
    def run(P):
        h2 = Head(d).to(DEV); op = torch.optim.Adam([h2.A, h2.B], lr=LR)
        b0 = readout_ce(h2, doms[0])
        for t in (1, 2, 3):
            for _ in range(STEPS):
                ids = doms[t][1][random.randint(0, len(doms[t][1]) - BS):][:BS].to(DEV)
                hs = model(ids, attention_mask=torch.ones_like(ids),
                           output_hidden_states=True).hidden_states
                h = hs[layer] + h2.delta(hs[layer])
                logits = model.get_output_embeddings()(model.gpt_neox.final_layer_norm(h))
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                       ids[:, 1:].reshape(-1))
                op.zero_grad(); loss.backward()
                if P is not None:                          # project B's row space off P
                    with torch.no_grad():
                        h2.B.grad -= (h2.B.grad @ P) @ P.T
                op.step()
        return readout_ce(h2, doms[0]) - b0
    res = {name: run(P) for name, P in
           [("naive", None), ("protect-Jspace", P_J), ("protect-Sigma", P_S),
            ("protect-random", P_R)]}
    log("H2 forgetting on A:  " + "  ".join(f"{k}={v:+.4f}" for k, v in res.items()))

    out = dict(model=MODEL, layer=layer, rank=RANK_J, overlap_J_Sigma=overlap,
               H1_forgetting=fgt, H1_jspace_energy_frac=e_J/max(e_tot,1e-12),
               H1_random_frac=RANK_J/d, H2=res)
    json.dump(out, open("jspace_interference_results.json", "w"), indent=1)
    log("ALL DONE -> jspace_interference_results.json")
    print("\nReading: H1 confirms if the J-space energy fraction >> rank/d "
          "(forgetting is verbalizable-subspace-mediated); H2 confirms if "
          "protect-Jspace ~ protect-Sigma << naive while protect-random ~ naive "
          "(J-space IS the interference-carrying subspace, not just any subspace).")
