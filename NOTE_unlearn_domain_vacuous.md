# imdb-domain unlearning: a vacuous run (archived for the record)

`unlearn_domain_results.json` (kaggle_unlearn_domain.py, pythia-410m + 1b).
**The test was vacuous: there was almost nothing to unlearn.**

The imdb LoRA adapter added only **+0.029 nats/token** of held-out imdb
knowledge over the base model at 410m, and **-0.009** at 1b (i.e. within
noise the joint model was no better than base on imdb anchors -- so the
stopping rule fired at step 20). imdb is ordinary movie-review English that
pythia already models well, so the adapter's imdb-specific contribution is
negligible. With so little to remove, the ascent moved only ~4% of that gap
and **both arms tied at a ~0 retained certificate** (410m: gated 2.4e-4 vs
naive 3.1e-4 -- directionally correct but negligible; 1b: both 3e-8).

This neither confirms nor refutes the exact-regime dichotomy
(`evaluate_unlearning.py`); it is the pre-flight lesson a fourth time --
**measure how much domain-specific knowledge exists (trained-vs-reference
gap) before attempting to unlearn it.**

Fix: `kaggle_unlearn_canary.py` injects a memorized canary (a fixed novel
string repeated in training) -- a large, cleanly-owned target that maps to
the "memorized content unlearns free" branch, so the gate-vs-naive
retained-certificate contrast becomes measurable.
