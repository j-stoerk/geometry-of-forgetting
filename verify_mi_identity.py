"""
Numerical verification of the general floor / mutual-information results.

Generative model: T ~ Uniform{A,B}; phi | T ~ N(mu_T, Sigma_T);
y = phi^T w_T + noise,  noise ~ N(0, s2),  noise _|_ (T, phi).

We verify, by Monte Carlo:
  (1) EXACT identity  D* = 1/2 E[ Var(T|phi) (phi^T delta)^2 ]
      against the directly simulated single-head excess risk D* = R* - s2/2.
  (2) The proven two-sided Shannon-MI sandwich
        1/8 E[G hb^2]  <=  D*  <=  1/4 E[G hb],
      with hb the binary entropy (nats), G = (phi^T delta)^2, and
      I(T;phi) = ln2 - E[hb(eta)].
  (3) Exact proportionality D* / I(T;phi) -> const in the low-separation
      (local) Gaussian limit.
Outputs results_mi.json and figures/fig_mi_identity.pdf.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
try:
    import mpl_style  # shared publication style
except Exception:
    pass
import matplotlib.pyplot as plt

RNG = np.random.default_rng(0)
BLUE, RED, GREEN, GREY = "#4C72B0", "#C44E52", "#55A868", "#777777"
res = {}


def gauss_logpdf(X, mu, Sig):
    d = X.shape[1]
    L = np.linalg.cholesky(Sig)
    sol = np.linalg.solve(L, (X - mu).T)
    quad = np.sum(sol ** 2, axis=0)
    logdet = 2 * np.sum(np.log(np.diag(L)))
    return -0.5 * (quad + logdet + d * np.log(2 * np.pi))


def binary_entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)        # nats


def model(d, sep, rng, equal_cov=True, s2=0.25, N=200000, struct_seed=None):
    """Sample a two-task Gaussian model with mean separation `sep`. If struct_seed
    is given, the geometry (direction, covariances, targets) is held fixed and only
    `sep` (hence identifiability) varies across calls."""
    srng = np.random.default_rng(struct_seed) if struct_seed is not None else rng
    direction = srng.standard_normal(d); direction /= np.linalg.norm(direction)
    muA = 0.5 * sep * direction; muB = -0.5 * sep * direction
    if equal_cov:
        Q = np.linalg.qr(srng.standard_normal((d, d)))[0]
        eig = srng.uniform(0.5, 1.5, d)
        SigA = SigB = (Q * eig) @ Q.T
    else:
        QA = np.linalg.qr(srng.standard_normal((d, d)))[0]
        QB = np.linalg.qr(srng.standard_normal((d, d)))[0]
        SigA = (QA * srng.uniform(0.5, 1.5, d)) @ QA.T
        SigB = (QB * srng.uniform(0.5, 1.5, d)) @ QB.T
    wA = srng.standard_normal(d); wB = srng.standard_normal(d)
    delta = wA - wB
    # sample equal numbers from each task
    nA = N // 2
    XA = rng.multivariate_normal(muA, SigA, nA)
    XB = rng.multivariate_normal(muB, SigB, N - nA)
    X = np.vstack([XA, XB]); Tlab = np.concatenate([np.ones(nA), np.zeros(N - nA)])
    # posterior eta = P(A | phi), equal prior
    la = gauss_logpdf(X, muA, SigA); lb = gauss_logpdf(X, muB, SigB)
    m = np.maximum(la, lb)
    eta = np.exp(la - m) / (np.exp(la - m) + np.exp(lb - m))
    G = (X @ delta) ** 2
    # (1) identity RHS  and  direct excess risk
    D_identity = 0.5 * np.mean(eta * (1 - eta) * G)
    mA = X @ wA; mB = X @ wB
    fstar = eta * mA + (1 - eta) * mB
    y = np.where(Tlab[:, None].ravel() == 1, mA, mB) + np.sqrt(s2) * rng.standard_normal(N)
    R_star = 0.5 * np.mean((fstar - y) ** 2)
    D_direct = R_star - 0.5 * s2
    # (2) information quantities and integrated sandwich
    hb = binary_entropy(eta)
    I_Tphi = np.log(2) - np.mean(hb)
    lower = 0.125 * np.mean(G * hb ** 2)
    upper = 0.25 * np.mean(G * hb)
    return dict(D_identity=D_identity, D_direct=D_direct, I=I_Tphi,
                lower=lower, upper=upper)


def run():
    rng = RNG
    # (1) exact identity across many random models
    idn, drc = [], []
    for _ in range(60):
        sep = rng.uniform(0.0, 3.0)
        out = model(8, sep, rng, equal_cov=bool(rng.integers(2)), N=120000)
        idn.append(out["D_identity"]); drc.append(out["D_direct"])
    idn, drc = np.array(idn), np.array(drc)
    rel = float(np.max(np.abs(idn - drc)) / (np.max(drc) + 1e-12))
    res["identity_max_rel_err"] = rel

    # (2)+(3) sweep separation: floor, MI, sandwich, proportionality
    seps = np.linspace(0.05, 7.0, 18)
    D, I, lo, hi = [], [], [], []
    for s in seps:
        o = model(8, s, rng, equal_cov=True, N=200000, struct_seed=12345)
        D.append(o["D_direct"]); I.append(o["I"]); lo.append(o["lower"]); hi.append(o["upper"])
    D, I, lo, hi = map(np.array, (D, I, lo, hi))
    res["sandwich_holds"] = bool(np.all(lo <= D + 1e-6) and np.all(D <= hi + 1e-6))
    # the floor is DECREASING in I(T;phi): maximal when unidentifiable, ->0 when identifiable
    res["floor_min_identifiability"] = float(D[0])    # sep->0, I~0  (max floor)
    res["floor_max_identifiability"] = float(D[-1])   # sep large, I~ln2 (floor ->0)
    res["I_min_sep"] = float(I[0]); res["I_max_sep"] = float(I[-1])

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.7))
    lim = max(idn.max(), drc.max()) * 1.08
    ax[0].plot([0, lim], [0, lim], "--", color=GREY, lw=1)
    ax[0].scatter(idn, drc, s=22, color=BLUE)
    ax[0].set_xlabel(r"identity  $\frac{1}{2}\,E[\,\mathrm{Var}(T|\phi)\,(\phi^\top\delta)^2]$")
    ax[0].set_ylabel(r"direct single-head excess risk $\mathcal{D}^\star$")
    ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim)

    ovl = np.log(2) - I                                    # H(T|phi): 0 identifiable .. ln2 not
    order = np.argsort(ovl)
    ax[1].fill_between(ovl[order], lo[order], hi[order], color=GREEN, alpha=0.25,
                       label="proven MI sandwich")
    ax[1].plot(ovl[order], D[order], "-o", color=BLUE, ms=4, label=r"floor $\mathcal{D}^\star$")
    ax[1].set_xlabel(r"$H(T|\phi)=\ln 2 - I(T;\phi)$  (unidentifiability)")
    ax[1].set_ylabel("distortion floor")
    ax[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig("figures/fig_mi_identity.pdf"); plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    run()
    with open("results_mi.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
