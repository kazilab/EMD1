"""Hierarchical Bayesian fit of the decay / reader sub-model.

Why only this sub-model
-----------------------
The full ODE carries ~40 parameters against roughly 8-10 constrained
quantities, and most of what constrains it are *figure-digitised point
estimates* with no recoverable replicate spread. A full-model posterior would
be prior-dominated almost everywhere and the sampling time would go into
non-identifiability pathology rather than science.

The decay sub-model is different: it has genuine per-replicate data
(648 chase points from the arsenic redox work, plus the NEDD4L chases from the
FTO/autophagy paper), it is identifiable, and it carries the claim the whole
EMD1 argument rests on -- that one m6A change moves two transcripts in opposite
directions because different readers bind them.

Model
-----
An actinomycin chase is first-order decay, so on the log scale it is linear:

    log y = -k * t + eps

with log k decomposed into a baseline plus arm effects. Effects are on the log
scale, so exp(effect) reads directly as a **decay-rate ratio** against the
arm's own control. A ratio > 1 means that arm decays faster, i.e. the
perturbation destabilised the transcript.

Antioxidant genes get partial pooling: both a gene-level baseline and a
gene-level deviation on each arm effect, so "is this a property of the module
or of one or two genes" is answered by the pooled effect and its spread rather
than by a tally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# numpyro is chatty about platform on import
import numpyro
import numpyro.distributions as dist
from jax import random
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS

numpyro.set_host_device_count(1)

from .zhao import parse_fig5b

XLSX_NC = None  # resolved lazily in load_nedd4l_decay


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_antioxidant_decay() -> pd.DataFrame:
    """Zhao chase. t=0 is exactly 1.000 by normalisation, so it carries no
    information and would only shrink the noise term -- dropped."""
    d = parse_fig5b()
    return d[d.hour > 0].reset_index(drop=True)


def load_nedd4l_decay() -> pd.DataFrame:
    """NEDD4L chases from the FTO/autophagy source data.

    Three blocks: 3i (Control vs As-T), 3m (WT vs FTO KO) and 5a, a 2x2 of
    FTO KO x shMETTL14. Unlike the Zhao chase these are not renormalised to
    exactly 1 at t=0, so the t=0 points carry real noise and are kept.
    """
    import openpyxl
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "data" / "papers" / \
        "NatCommun2021_NEDD4L_SourceData.xlsx"
    wb = openpyxl.load_workbook(p, data_only=True)
    rows = []
    for sheet, panel, arms in [
            ("Fig.3", "3i", ["Control", "As-T"]),
            ("Fig.3", "3m", ["WT", "FTO KO"]),
            ("Fig.5", "5a", ["WT+shNC", "FTO KO+shNC", "WT+shMETTL14", "FTO KO+shMETTL14"])]:
        g = [list(r) for r in wb[sheet].iter_rows(values_only=True)]
        start = next(i for i, r in enumerate(g)
                     if any(isinstance(c, str) and c == panel for c in r))
        hour, rep = None, 0
        for r in g[start + 1:]:
            if not r or all(c is None for c in r):
                break
            lab = next((c for c in r[:4] if isinstance(c, str) and c.endswith("h")), None)
            if lab is not None:
                hour, rep = float(lab[:-1]), 0
            if hour is None:
                continue
            vals = [c for c in r if isinstance(c, (int, float))]
            if len(vals) < len(arms):
                continue
            rep += 1
            for a, v in zip(arms, vals[-len(arms):]):
                rows.append(dict(panel=panel, arm=a, hour=hour, rep=rep, frac=float(v)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
def _antiox_model(t, y, gene, stage, arm, n_gene, n_stage, n_arm):
    mu = numpyro.sample("mu", dist.Normal(np.log(0.12), 1.0))
    tau_g = numpyro.sample("tau_gene", dist.HalfNormal(0.5))
    tau_s = numpyro.sample("tau_stage", dist.HalfNormal(0.3))
    # non-centred: avoids the funnel these hierarchies otherwise produce
    zg = numpyro.sample("z_gene", dist.Normal(0, 1).expand([n_gene]))
    zs = numpyro.sample("z_stage", dist.Normal(0, 1).expand([n_stage]))

    # arm 0 is siControl and is the reference, so only n_arm-1 effects
    delta = numpyro.sample("delta", dist.Normal(0, 1.0).expand([n_arm - 1]))
    tau_a = numpyro.sample("tau_arm", dist.HalfNormal(0.5).expand([n_arm - 1]))
    za = numpyro.sample("z_arm", dist.Normal(0, 1).expand([n_gene, n_arm - 1]))

    eff = jnp.concatenate([jnp.zeros((n_gene, 1)),
                           delta[None, :] + tau_a[None, :] * za], axis=1)
    log_k = mu + tau_g * zg[gene] + tau_s * zs[stage] + eff[gene, arm]
    k = numpyro.deterministic("k", jnp.exp(log_k))
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.3))
    numpyro.sample("obs", dist.Normal(-k * t, sigma), obs=jnp.log(y))


def _nedd4l_model(t, y, arm, n_arm):
    mu = numpyro.sample("mu", dist.Normal(np.log(0.10), 1.0))
    delta = numpyro.sample("delta", dist.Normal(0, 1.0).expand([n_arm - 1]))
    eff = jnp.concatenate([jnp.zeros(1), delta])
    k = numpyro.deterministic("k", jnp.exp(mu + eff[arm]))
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.3))
    numpyro.sample("obs", dist.Normal(-k * t, sigma), obs=jnp.log(y))


def _run(model, seed=0, warmup=1000, samples=2000, chains=4, **kw):
    """Four sequential chains so r_hat is meaningful; one chain cannot detect
    the multimodality that non-identifiable hierarchies are prone to."""
    mcmc = MCMC(NUTS(model, target_accept_prob=0.95), num_warmup=warmup,
                num_samples=samples, num_chains=chains,
                chain_method="sequential", progress_bar=False)
    mcmc.run(random.PRNGKey(seed), extra_fields=("diverging",), **kw)
    return mcmc


def diagnostics(mcmc, names=("mu", "delta", "sigma")) -> str:
    """Worst r_hat and smallest effective sample size across the named sites."""
    from numpyro.diagnostics import summary
    st = summary(mcmc.get_samples(group_by_chain=True), prob=0.9)
    worst_r, min_n = 1.0, np.inf
    for k, v in st.items():
        if not any(k.startswith(n) for n in names):
            continue
        worst_r = max(worst_r, float(np.nanmax(v["r_hat"])))
        min_n = min(min_n, float(np.nanmin(v["n_eff"])))
    nd = int(mcmc.get_extra_fields().get("diverging", np.zeros(1)).sum()) \
        if "diverging" in mcmc.get_extra_fields() else -1
    return (f"max r_hat = {worst_r:.4f}   min ESS = {min_n:.0f}"
            + (f"   divergences = {nd}" if nd >= 0 else ""))


def _q(x, lo=5, hi=95):
    return np.percentile(np.asarray(x), [lo, 50, hi])


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
ARMS_A = ["siControl", "siMETTL3", "siFTO"]

# NEDD4L arms, grouped by the panel they belong to. Each panel has its own
# control, so panel baselines are estimated separately and every effect is
# read against that panel's own reference.
PANELS_N = {"3i": ["Control", "As-T"],
            "3m": ["WT", "FTO KO"],
            "5a": ["WT+shNC", "FTO KO+shNC", "WT+shMETTL14", "FTO KO+shMETTL14"]}
REF_N = {v[0] for v in PANELS_N.values()}


def _nedd4l_model2(t, y, panel, eff_idx, n_panel, n_eff):
    mu = numpyro.sample("mu", dist.Normal(np.log(0.10), 1.0))
    b_panel = numpyro.sample("b_panel", dist.Normal(0, 0.5).expand([n_panel - 1]))
    bp = jnp.concatenate([jnp.zeros(1), b_panel])
    delta = numpyro.sample("delta", dist.Normal(0, 1.0).expand([n_eff]))
    eff = jnp.concatenate([jnp.zeros(1), delta])      # index 0 = "is a control arm"
    k = numpyro.deterministic("k", jnp.exp(mu + bp[panel] + eff[eff_idx]))
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.3))
    numpyro.sample("obs", dist.Normal(-k * t, sigma), obs=jnp.log(y))


def fit_antioxidant(seed=0):
    d = load_antioxidant_decay()
    genes = sorted(d.gene.unique()); stages = sorted(d.stage.unique())
    gi = d.gene.map({g: i for i, g in enumerate(genes)}).to_numpy()
    si = d.stage.map({s: i for i, s in enumerate(stages)}).to_numpy()
    ai = d.sirna.map({a: i for i, a in enumerate(ARMS_A)}).to_numpy()
    mcmc = _run(_antiox_model, seed=seed,
                t=jnp.array(d.hour.to_numpy()), y=jnp.array(d.frac.to_numpy()),
                gene=jnp.array(gi), stage=jnp.array(si), arm=jnp.array(ai),
                n_gene=len(genes), n_stage=len(stages), n_arm=len(ARMS_A))
    return mcmc, genes


def fit_nedd4l(seed=0):
    d = load_nedd4l_decay()
    panels = list(PANELS_N)
    effects = [a for p in panels for a in PANELS_N[p] if a not in REF_N]
    pi = d.panel.map({p: i for i, p in enumerate(panels)}).to_numpy()
    ei = d.arm.map(lambda a: 0 if a in REF_N else effects.index(a) + 1).to_numpy()
    mcmc = _run(_nedd4l_model2, seed=seed,
                t=jnp.array(d.hour.to_numpy()), y=jnp.array(d.frac.to_numpy()),
                panel=jnp.array(pi), eff_idx=jnp.array(ei),
                n_panel=len(panels), n_eff=len(effects))
    return mcmc, effects


def main() -> int:
    print("=" * 84)
    print("Hierarchical Bayesian fit of the decay / reader sub-model")
    print("Effects are decay-RATE RATIOS vs that arm's own control.")
    print("  ratio > 1  = perturbation destabilised the transcript (faster decay)")
    print("  ratio < 1  = perturbation stabilised it")
    print("=" * 84)

    # ---- antioxidant module ------------------------------------------
    mcmc, genes = fit_antioxidant()
    s = mcmc.get_samples()
    print("\nANTIOXIDANT MODULE  (6 genes, 3 stages, 486 chase points)")
    print(f"  convergence: {diagnostics(mcmc)}")
    print(f"  {'arm':12s}{'ratio (median)':>16s}{'90% CrI':>20s}   P(ratio>1)")
    for j, arm in enumerate(ARMS_A[1:]):
        r = np.exp(np.asarray(s["delta"][:, j]))
        lo, md, hi = _q(r)
        print(f"  {arm:12s}{md:16.2f}   [{lo:5.2f}, {hi:5.2f}]      {np.mean(r > 1):.3f}")
    ta = np.exp(np.asarray(s["tau_arm"]))
    print(f"\n  between-gene spread of the arm effects (tau, log scale):")
    for j, arm in enumerate(ARMS_A[1:]):
        lo, md, hi = _q(np.asarray(s["tau_arm"][:, j]))
        print(f"    {arm:12s} {md:.2f}  [{lo:.2f}, {hi:.2f}]")
    print(f"  residual sigma: {np.median(np.asarray(s['sigma'])):.3f}")

    # per-gene effect, to see whether the module moves together
    print(f"\n  per-gene siMETTL3 ratio:")
    for i, g in enumerate(genes):
        r = np.exp(np.asarray(s["delta"][:, 0] + s["tau_arm"][:, 0] * s["z_arm"][:, i, 0]))
        lo, md, hi = _q(r)
        print(f"    {g:7s} {md:5.2f}  [{lo:4.2f}, {hi:5.2f}]   P(>1) = {np.mean(r > 1):.3f}")

    # ---- NEDD4L -------------------------------------------------------
    mcmc2, effects = fit_nedd4l()
    s2 = mcmc2.get_samples()
    print("\n\nNEDD4L  (72 chase points, 3 panels, each vs its own control)")
    print(f"  convergence: {diagnostics(mcmc2)}")
    print(f"  {'arm':20s}{'ratio (median)':>16s}{'90% CrI':>20s}   P(ratio>1)")
    idx = {a: j for j, a in enumerate(effects)}
    for a in effects:
        r = np.exp(np.asarray(s2["delta"][:, idx[a]]))
        lo, md, hi = _q(r)
        print(f"  {a:20s}{md:16.2f}   [{lo:5.2f}, {hi:5.2f}]      {np.mean(r > 1):.3f}")

    # 2x2 epistasis: is FTO's stabilisation abolished when the writer is down?
    d_fto = np.asarray(s2["delta"][:, idx["FTO KO+shNC"]])
    d_m14 = np.asarray(s2["delta"][:, idx["WT+shMETTL14"]])
    d_both = np.asarray(s2["delta"][:, idx["FTO KO+shMETTL14"]])
    inter = d_both - d_fto - d_m14
    lo, md, hi = _q(np.exp(inter))
    print(f"\n  2x2 interaction (FTO KO x shMETTL14), as a ratio: "
          f"{md:.2f}  [{lo:.2f}, {hi:.2f}]")
    print(f"  P(interaction > 0, i.e. FTO's effect needs the writer) = {np.mean(inter > 0):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
