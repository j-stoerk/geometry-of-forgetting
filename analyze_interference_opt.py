# =====================================================================
#  analyze_interference_opt.py -- shape-agnostic analysis of the
#  InterferenceSGD benchmark results.  Handles every schema the script
#  has emitted ((forget, avg) tuples/lists and config dicts), so it also
#  runs on the completed 2.2h v3 JSON whose in-notebook report crashed.
#
#  Usage:  python analyze_interference_opt.py [path/to/interference_opt_results.json]
# =====================================================================
import json, sys
import numpy as np

def to_xy(entries):
    """Normalize a frontier list to an (n,2) array of (forgetting, avg loss)."""
    pts = []
    for e in entries:
        if isinstance(e, dict):
            pts.append((float(e["forget"]), float(e["avg"])))
        else:
            pts.append((float(e[0]), float(e[1])))
    return np.array(sorted(pts))

def pareto(pts):
    """Non-dominated subset (lower forgetting AND lower avg loss are better)."""
    keep = []
    for i, (f, a) in enumerate(pts):
        if not any((pts[j, 0] <= f and pts[j, 1] <= a and j != i
                    and (pts[j, 0] < f or pts[j, 1] < a)) for j in range(len(pts))):
            keep.append((f, a))
    return np.array(sorted(keep))

def report(per):
    seeds = sorted(per.keys())
    print("\n===== frontier comparison (common support, both axes) =====")
    for s in seeds:
        fs, fi = to_xy(per[s]["schedule"]), to_xy(per[s]["interference"])
        print(f"  seed {s}: schedule forgetting [{fs[:,0].min():+.3f},{fs[:,0].max():+.3f}] "
              f"avg [{fs[:,1].min():.3f},{fs[:,1].max():.3f}] | penalty forgetting "
              f"[{fi[:,0].min():+.3f},{fi[:,0].max():+.3f}] avg [{fi[:,1].min():.3f},{fi[:,1].max():.3f}]")
    # matched-forgetting comparison on each seed's common support
    print("  --- avg final loss at matched forgetting ---")
    any_support = False
    for frac in [0.25, 0.5, 0.75]:
        ls, li = [], []
        for s in seeds:
            fs, fi = to_xy(per[s]["schedule"]), to_xy(per[s]["interference"])
            lo = max(fs[:, 0].min(), fi[:, 0].min()); hi = min(fs[:, 0].max(), fi[:, 0].max())
            if hi <= lo: continue
            f = lo + frac * (hi - lo)
            ls.append(np.interp(f, fs[:, 0], fs[:, 1])); li.append(np.interp(f, fi[:, 0], fi[:, 1]))
        if ls:
            any_support = True
            print(f"    at {int(frac*100)}% of common range (n={len(ls)}): "
                  f"tuned-Adam {np.mean(ls):.3f}   penalty {np.mean(li):.3f}   "
                  f"({100*(np.mean(ls)-np.mean(li))/np.mean(ls):+.2f}%)")
    if not any_support:
        print("    no common support -- the arms occupy disjoint frontier regions")
    # matched-loss comparison
    print("  --- forgetting at matched avg final loss ---")
    for off in [0.02, 0.05, 0.1]:
        gs, gi = [], []
        for s in seeds:
            fs = to_xy(per[s]["schedule"]); fi = to_xy(per[s]["interference"])
            fs = fs[np.argsort(fs[:, 1])]; fi = fi[np.argsort(fi[:, 1])]
            b = max(fs[:, 1].min(), fi[:, 1].min()) + off
            if b > min(fs[:, 1].max(), fi[:, 1].max()): continue
            gs.append(np.interp(b, fs[:, 1], fs[:, 0])); gi.append(np.interp(b, fi[:, 1], fi[:, 0]))
        if gs:
            print(f"    at min avg-loss+{off:.2f} (n={len(gs)}): "
                  f"tuned-Adam forgetting {np.mean(gs):+.3f}   penalty {np.mean(gi):+.3f}")
    # Pareto view + best configs
    print("  --- per-seed Pareto fronts (forgetting, avg loss) ---")
    for s in seeds:
        print(f"    seed {s} schedule : " + "  ".join(f"({f:+.3f},{a:.3f})" for f, a in pareto(to_xy(per[s]['schedule']))))
        print(f"    seed {s} penalty  : " + "  ".join(f"({f:+.3f},{a:.3f})" for f, a in pareto(to_xy(per[s]['interference']))))
    for s in seeds:
        ent = [e for e in per[s]["interference"] if isinstance(e, dict)]
        if ent:
            best = min(ent, key=lambda e: e["avg"] + max(e["forget"], 0))
            print(f"    seed {s} best penalty config: {best['schedule']}/{best['peak_lr']:g} "
                  f"mu={best['mu']:g} -> forget {best['forget']:+.3f} avg {best['avg']:.3f}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "interference_opt_results.json"
    report(json.load(open(path)))
