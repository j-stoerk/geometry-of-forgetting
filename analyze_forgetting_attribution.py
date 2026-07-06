# =====================================================================
#  analyze_forgetting_attribution.py -- assemble the ATTRIBUTED
#  FORGETTING tables from forgetting_attribution_v2_results.json
#  (produced on Kaggle; archived copy in this repo).
#
#  Decomposition per model, per earlier domain, at deployed rank r=8:
#      final_seq - single_r8  =  [joint_r8 - single_r8]  +  [final_seq - joint_r8]
#             excess                  CAPACITY                  CONTROL
#  with FLOOR bounded by any domain-classifier's cross-entropy
#  (CE >= H(T|X) >= I(T;Y|X)); the v2 probe bound is LOOSE (see
#  kaggle_floor_certificate.py for the tight generative certificate).
#  kl_to_single additionally instruments the Bregman identity with the
#  single-domain models as anchors: CE-excess = residual + E[KL].
# =====================================================================
import json, sys
import numpy as np
from scipy import stats

PATH = sys.argv[1] if len(sys.argv) > 1 else "forgetting_attribution_v2_results.json"
DOMS = ["news", "imdb", "wiki", "tweets"]
EARLIER = ["news", "imdb", "wiki"]
RANKS = [2, 8, 32]

d = json.load(open(PATH))
for model in d:
    R = d[model]
    print("=" * 74); print(f"ATTRIBUTED FORGETTING -- {model}  (nats/token)"); print("=" * 74)
    single = {r: {DOMS[i]: np.mean([v["eval_ce"] for v in R[f"single/r{r}/d{i}"].values()])
                  for i in range(4)} for r in RANKS}
    cap = {}
    for r in RANKS:
        per_seed = [np.mean([v["eval_ce"][m] - single[r][m] for m in DOMS])
                    for v in R[f"joint/r{r}"].values()]
        cap[r] = (np.mean(per_seed), np.std(per_seed))
    def arm(name):
        runs = list(R[f"seq/{name}/r8"].values())
        fgt = np.array([np.mean([v["final_ce"][m] - v["diag_ce"][m] for m in EARLIER]) for v in runs])
        exc = np.array([np.mean([v["final_ce"][m] - single[8][m] for m in EARLIER]) for v in runs])
        kls = np.array([v["kl_to_single"]["mean"] for v in runs])
        return fgt, exc, kls, runs
    fgt_n, exc_n, kl_n, runs_n = arm("naive")
    fgt_g, exc_g, kl_g, _ = arm("func-active")
    joint8 = {m: np.mean([v["eval_ce"][m] for v in R["joint/r8"].values()]) for m in DOMS}
    ctl_n = np.mean([np.mean([v["final_ce"][m] - joint8[m] for m in EARLIER]) for v in runs_n])
    ctl_g = np.mean([np.mean([v["final_ce"][m] - joint8[m] for m in EARLIER])
                     for v in R["seq/func-active/r8"].values()])
    cap8e = np.mean([joint8[m] - single[8][m] for m in EARLIER])
    diag_gap = np.mean([np.mean([v["diag_ce"][m] - single[8][m] for m in DOMS]) for v in runs_n])
    p = stats.ttest_rel(fgt_n, fgt_g).pvalue
    print(f"  forgetting naive {fgt_n.mean():.3f}+/-{fgt_n.std():.3f}   "
          f"gate {fgt_g.mean():.3f}+/-{fgt_g.std():.3f}   (paired p={p:.4f})")
    print(f"  FLOOR (probe bound, LOOSE)     : <= {R['probe']['probe_ce']:.3f}"
          f"  (acc {R['probe']['probe_acc']:.2f}; trivial ln4={np.log(4):.3f})")
    print(f"  CAPACITY(r)                    : " +
          "  ".join(f"r={r}: {cap[r][0]:+.3f}+/-{cap[r][1]:.3f}" for r in RANKS)
          + f"   [earlier-only r=8: {cap8e:+.3f}]")
    print(f"  CONTROL naive {ctl_n:+.3f}  gate {ctl_g:+.3f}"
          f"   ({100*(1-ctl_g/max(ctl_n,1e-9)):.0f}% removed)")
    print(f"  plasticity check (diag-single) : {diag_gap:+.3f}")
    print(f"  KL instrument: naive E[KL]={kl_n.mean():.3f} vs excess {exc_n.mean():.3f}"
          f" (residual {exc_n.mean()-kl_n.mean():+.3f});"
          f"  gate E[KL]={kl_g.mean():.3f} vs {exc_g.mean():.3f}"
          f" (residual {exc_g.mean()-kl_g.mean():+.3f})")
print("""
Reading: control dominates capacity at BOTH scales (2-4x) -- most measured
forgetting is a POLICY problem, not a width problem -- and the active-set gate
removes a large fraction of exactly that term.  The rank sweep leaves capacity
small and roughly rank-insensitive at this budget (the joint surrogate includes
joint-optimization error, so capacity is an over-estimate).  The nonzero KL
residual is the held-out-anchor effect: single-domain models are converged on
their TRAIN data, not on the test anchor, so forgetting != pure KL by exactly
the Bregman residual term.  Floor certification pending the generative
classifier (kaggle_floor_certificate.py).  Caveat: LR fixed at 1e-4 across
scales; the 410m->1b comparison is at fixed hyperparameters, not per-scale
tuned.""")


# ------------------------- KL view (the v2 protocol's primary observable) ----
print("\n" + "#" * 74)
print("# KL ATTRIBUTION (anchored per-token KL to single-domain references)")
print("#" * 74)
for model in d:
    R = d[model]
    jkl = {r: np.array([v["kl_to_single"]["mean"] for v in R[f"joint/r{r}"].values()])
           for r in RANKS}
    nkl = np.array([v["kl_to_single"]["mean"] for v in R["seq/naive/r8"].values()])
    gkl = np.array([v["kl_to_single"]["mean"] for v in R["seq/func-active/r8"].values()])
    p = stats.ttest_rel(nkl, gkl).pvalue
    ctl_n, ctl_g = nkl.mean() - jkl[8].mean(), gkl.mean() - jkl[8].mean()
    print(f"\n== {model}")
    print(f"  naive KL {nkl.mean():.4f}+/-{nkl.std():.4f}   gate {gkl.mean():.4f}+/-"
          f"{gkl.std():.4f}  (paired p={p:.4f}, removed {nkl.mean()-gkl.mean():+.4f})")
    print(f"  KL_joint(r): " + "  ".join(f"r={r}: {jkl[r].mean():.4f}" for r in RANKS)
          + "   [FLAT in r -> substantially a two-models baseline, not capacity;")
    print(f"   calibrate with the seed-to-seed KL of kaggle_floor_certificate.py]")
    print(f"  control naive {ctl_n:+.4f}  gate {ctl_g:+.4f}"
          f"  ({100*(1-ctl_g/max(ctl_n,1e-9)):.0f}% removed)")


# ------------- FINAL netted attribution (with the certificate results) -------
import os
if os.path.exists("floor_certificate_results.json"):
    cert = json.load(open("floor_certificate_results.json"))
    # the floor is a DATA property: the best classifier over all models bounds
    # it for every model (I(T;Y|X) <= H(T|X) <= min over classifiers of CE)
    FLOOR = min(v["classifier_ce"] for v in cert.values())
    best = min(cert, key=lambda k: cert[k]["classifier_ce"])
    print("\n" + "#" * 74)
    print("# FINAL ATTRIBUTION -- floor certified, KL terms netted of seed baseline")
    print("#" * 74)
    print(f"  CERTIFIED FLOOR of the stream: I(T;Y|X) <= {FLOOR:.4f} nats/token")
    print(f"  (generative classifier of {best}: acc "
          f"{cert[best]['classifier_acc']:.3f}, CE {FLOOR:.4f}; data property ->")
    print(f"   applies to every model's attribution)")
    for model in d:
        R = d[model]; b = cert[model]["seed_baseline_kl"]
        nkl = np.mean([v["kl_to_single"]["mean"] for v in R["seq/naive/r8"].values()])
        gkl = np.mean([v["kl_to_single"]["mean"] for v in R["seq/func-active/r8"].values()])
        jkl = np.mean([v["kl_to_single"]["mean"] for v in R["joint/r8"].values()])
        fn = np.mean([np.mean([v["final_ce"][m] - v["diag_ce"][m] for m in EARLIER])
                      for v in R["seq/naive/r8"].values()])
        print(f"\n== {model}   (seed-to-seed KL baseline {b:.4f})")
        print(f"  net KL forgetting  naive {nkl - b:.3f}   gate {gkl - b:.3f}")
        print(f"  net class term     {jkl - b:.3f}   net control naive {nkl - jkl:.3f}"
              f"  gate {gkl - jkl:.3f}")
        print(f"  CE forgetting {fn:.3f}  vs certified floor {FLOOR:.4f}"
              f"  ->  >= {100 * (1 - FLOOR / fn):.1f}% AVOIDABLE IN PRINCIPLE (certified)")
    print("\n  Note: the seed baseline (~0.02-0.03) explains only ~15% of the flat")
    print("  KL_joint, so the class term is real but rank-insensitive at this")
    print("  budget: interference inside the deployed class sits in the shared")
    print("  frozen geometry / joint optimization, not in adapter rank.")
