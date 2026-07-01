# =====================================================================
# Variance reporting for the two locally-runnable real benchmarks:
#   Split-Digits   (Table 1, dissimilar tasks)
#   Rotated-Digits (Table 2, similar tasks)
# Runs N seeds, reports mean +/- 95% CI (t-distribution) for accuracy and
# forgetting, and a paired significance test (IGFA vs OGD, the matched
# structural baseline; and IGFA vs the best non-IGFA baseline). Regenerates
# fig_benchmark.pdf / fig_benchmark2.pdf with mean +/- CI error bars so the
# figures and tables share the multi-seed numbers.
#
# Run:  python variance_report.py        (writes variance_results.json + figs)
# =====================================================================
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
try:
    import mpl_style  # shared publication style
except Exception:
    pass
import matplotlib.pyplot as plt
from scipy import stats

import experiments_rev as E

FIGDIR = "figures"
GREY, ORANGE, BLUE, PURPLE, GREEN = "#7f8c8d", "#d68910", "#2b6cb0", "#8854d0", "#27875a"
COLS = [GREY, ORANGE, BLUE, PURPLE, GREEN]
METHODS = ["naive", "ewc", "ogd", "replay", "igfa"]

N_SEEDS_DIGITS = 10
N_SEEDS_ROT = 5
ROT_CACHE = "rot_cache.npz"


def rotations_cached(angles=(0, 12, 24, 36, 48)):
    """Compute the slow scipy rotations once and cache to disk for reuse."""
    import os
    if os.path.exists(ROT_CACHE):
        z = np.load(ROT_CACHE)
        return {int(k): z[k] for k in z.files}
    rot = E.rotate_digits_once(angles)
    np.savez_compressed(ROT_CACHE, **{str(a): rot[a] for a in angles})
    return rot


def ci95(x):
    """mean and half-width of the 95% CI (Student-t)."""
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    if n < 2:
        return m, 0.0
    se = x.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return float(m), float(h)


def paired_p(a, b):
    """Two-sided paired t-test p-value between two per-seed arrays."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return 1.0
    return float(stats.ttest_rel(a, b).pvalue)


def aggregate(runs):
    """runs: list of {method: (acc, forget)} -> per-method arrays."""
    acc = {m: np.array([r[m][0] for r in runs]) for m in METHODS}
    frg = {m: np.array([r[m][1] for r in runs]) for m in METHODS}
    return acc, frg


def report(name, acc, frg):
    print(f"\n===== {name}  (n={len(acc['igfa'])} seeds) =====")
    print(f"{'method':10s} {'acc mean+/-CI95':>20s} {'forget mean+/-CI95':>22s}")
    for m in METHODS:
        am, ah = ci95(acc[m]); fm, fh = ci95(frg[m])
        print(f"{m:10s} {am:.3f} +/- {ah:.3f}      {fm:+.3f} +/- {fh:.3f}")
    # significance: IGFA vs OGD (matched structural baseline)
    p_acc_ogd = paired_p(acc["igfa"], acc["ogd"])
    p_frg_ogd = paired_p(frg["igfa"], frg["ogd"])
    # IGFA vs best non-IGFA baseline on accuracy
    base = [m for m in METHODS if m != "igfa"]
    best = max(base, key=lambda m: acc[m].mean())
    p_acc_best = paired_p(acc["igfa"], acc[best])
    print(f"  paired t-test  IGFA vs OGD : acc p={p_acc_ogd:.3g}, forget p={p_frg_ogd:.3g}")
    print(f"  paired t-test  IGFA vs {best:6s}: acc p={p_acc_best:.3g}")
    return {
        "n_seeds": len(acc["igfa"]),
        "acc": {m: list(map(float, acc[m])) for m in METHODS},
        "forget": {m: list(map(float, frg[m])) for m in METHODS},
        "acc_mean_ci": {m: ci95(acc[m]) for m in METHODS},
        "forget_mean_ci": {m: ci95(frg[m]) for m in METHODS},
        "p_igfa_vs_ogd_acc": p_acc_ogd,
        "p_igfa_vs_ogd_forget": p_frg_ogd,
        "best_baseline": best, "p_igfa_vs_best_acc": p_acc_best,
    }


def make_fig(acc, frg, titles, ylabels, fname):
    labels = {"naive": "Naive", "ewc": "EWC", "ogd": "OGD/GPM",
              "replay": "Replay", "igfa": "igfa (ours)"}
    xs = np.arange(len(METHODS))
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for k, (data, ttl, yl) in enumerate([(acc, titles[0], ylabels[0]),
                                         (frg, titles[1], ylabels[1])]):
        mu = np.array([ci95(data[m])[0] for m in METHODS])
        hi = np.array([ci95(data[m])[1] for m in METHODS])
        ax[k].bar(xs, mu, yerr=hi, color=COLS, capsize=3,
                  error_kw=dict(ecolor="#333333", lw=1.1))
        ax[k].set_xticks(xs)
        ax[k].set_xticklabels([labels[m] for m in METHODS], rotation=25, ha="right", fontsize=8)
        ax[k].set_ylabel(yl); ax[k].set_title(ttl, fontsize=9.5)
        for i, m in enumerate(METHODS):
            off = hi[i] + (0.02 if k == 0 else 0.006)
            ax[k].text(i, mu[i] + off, f"{mu[i]:.2f}", ha="center", fontsize=7.5)
    ax[0].set_ylim(0, 1.04)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/{fname}"); plt.close(fig)


if __name__ == "__main__":
    out = {}

    # ---- Split-Digits (dissimilar) ----
    print("Running Split-Digits across seeds ...")
    runs = [E.exp_benchmark(seed=s, make_fig=False) for s in range(N_SEEDS_DIGITS)]
    acc, frg = aggregate(runs)
    out["split_digits"] = report("Split-Digits", acc, frg)
    make_fig(acc, frg,
             ["Retention across 5 tasks (mean $\\pm$ 95% CI)",
              "Forgetting (frozen-feature, A1 exact)"],
             ["avg. final accuracy", "avg. forgetting (lower better)"],
             "fig_benchmark.pdf")

    # ---- Rotated-Digits (similar) ----  rotate once (cached), then sweep seeds
    print("Loading/Pre-computing rotations (cached) ...")
    rot_imgs = rotations_cached()
    print("Running Rotated-Digits across seeds ...")
    runs2 = [E.exp_benchmark2(seed=s, make_fig=False, rot_imgs=rot_imgs)
             for s in range(N_SEEDS_ROT)]
    acc2, frg2 = aggregate(runs2)
    out["rotated_digits"] = report("Rotated-Digits", acc2, frg2)
    make_fig(acc2, frg2,
             ["Rotated-Digits (similar): retention",
              "igfa shares aligned tasks"],
             ["avg. final accuracy", "avg. forgetting (lower better)"],
             "fig_benchmark2.pdf")

    with open("variance_results.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else o)
    print("\nwrote variance_results.json and regenerated fig_benchmark*.pdf")
