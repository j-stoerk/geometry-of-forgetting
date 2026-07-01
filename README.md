# Forgetting as Interference in Continual Learning

Code, figures, and results for the paper *"Forgetting as Interference in Continual
Learning: Optimal Merging and Interference-Gated Allocation"* (Julius Störk,
VARTA Microbattery / TU Braunschweig). The compiled manuscript is `main.pdf`;
the source is `main.tex` (pdfLaTeX, run twice, bibliography embedded).

**Core claim.** In the frozen-feature regime, the forgetting of task A caused by a later
update Δ equals the interference energy ½ ΔᵀΣ_A Δ. This exact identity yields a
removability dichotomy, an irreducible distortion floor tied to task inference, an
optimal-merge identity (Σ-orthogonalization), and igfa — a replay-free, Fisher-free
allocation rule that shares capacity when tasks align and orthogonalizes under conflict.

## Reproducing the results

All simulation figures and numbers regenerate from fixed seeds:

| Script | Produces |
|---|---|
| `experiments.py` | interference identity, lossless/relocation, sign-change s*, capacity floor |
| `experiments_rev.py` | curvature-drift breakdown + segment-averaged fix, Split-Digits benchmark |
| `variance_report.py` | multi-seed mean ± CI and paired significance for the benchmark tables |
| `experiments_compare.py` | nine-method comparison (Fig. 1) and offline-merge residual-D ablation |
| `verify_mi_identity.py` | information–estimation identity and multiclass constants |
| `experiments_drift.py` | s* ablation, graceful degradation, drift/stale-subspace figures |
| `experiments_extra.py` | self-calibrating s*, tight multiclass constants |
| `experiments_roadmap.py` | feasibility diagnostic, Frequent Directions, soft knee, per-direction gate |
| `evaluate_hypotheses.py` | extensions: federated Σ-merge, OOD signal, task-free igfa, multimodal blocks |
| `evaluate_applications.py` | extensions: PINN interference, Fisher-metric policy gradients, meta-learned floor (Fig. S9) |
| `fig_theorems.py`, `fig_concept.py` | theorem schematics |

GPU notebooks (Colab/Kaggle) for the real-backbone experiments:
`colab_split_cifar100_igfa.py`, `colab_lora_llm_igfa.py`,
`colab_lora_llm_igfa_3approaches.py`, `colab_continual_pretrain_igfa.py`,
`kaggle_pending_experiments.py`, `kaggle_H2_H3_H12.py`.

Result logs are the `*.json` files; figures are in `figures/`.
`mpl_style.py` provides the shared matplotlib style.
