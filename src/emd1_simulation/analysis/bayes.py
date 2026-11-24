"""Hierarchical Bayesian fits of first-order mRNA decay in chase assays.

The response follows log(y) = -k*t + error. Exponentiating a log-rate
contrast gives the decay-rate ratio relative to its control. A ratio above
one means faster decay.

The six-gene antioxidant model partially pools gene baselines and intervention
effects. Baselines vary by exposure stage, while the intervention effect for
each gene is shared across stages. The ODE represents five writer-associated
genes as one pool and omits the PRDX5 branch. The chase fit does not calibrate
that pool. NEDD4L contrasts use the control from their own source panel.
"""

from __future__ import annotations

import argparse
import numpy as np

from ..processing.decay import load_antioxidant_decay, load_nedd4l_decay
from .bayesian_diagnostics import format_diagnostics

_BACKEND_LOADED = False


def _load_backend() -> None:
    """Load optional samplers only when fitting is explicitly requested."""
    global _BACKEND_LOADED, numpyro, dist, random, jnp, MCMC, NUTS
    if _BACKEND_LOADED:
        return
    try:
        import numpyro
        import numpyro.distributions as dist
        from jax import random
        import jax.numpy as jnp
        from numpyro.infer import MCMC, NUTS
    except ImportError as exc:
        raise ImportError(
            "Optional Bayesian analysis requires JAX and NumPyro. From the EMD1 "
            "directory, run pip install -r src/emd1_simulation/requirements-bayes.txt. "
            "The offline simulation does not need these packages."
        ) from exc
    numpyro.set_host_device_count(1)
    _BACKEND_LOADED = True


def _antiox_model(t, y, gene, stage, arm, n_gene, n_stage, n_arm):
    _load_backend()
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
    _load_backend()
    mu = numpyro.sample("mu", dist.Normal(np.log(0.10), 1.0))
    delta = numpyro.sample("delta", dist.Normal(0, 1.0).expand([n_arm - 1]))
    eff = jnp.concatenate([jnp.zeros(1), delta])
    k = numpyro.deterministic("k", jnp.exp(mu + eff[arm]))
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.3))
    numpyro.sample("obs", dist.Normal(-k * t, sigma), obs=jnp.log(y))


def _run(model, seed=0, warmup=1000, samples=2000, chains=4, **kw):
    """Sample sequential chains for between-chain convergence diagnostics."""
    _load_backend()
    mcmc = MCMC(NUTS(model, target_accept_prob=0.95), num_warmup=warmup,
                num_samples=samples, num_chains=chains,
                chain_method="sequential", progress_bar=False)
    mcmc.run(random.PRNGKey(seed), extra_fields=("diverging",), **kw)
    return mcmc


def diagnostics(mcmc, names=None) -> str:
    """Summarise convergence for all returned sites or selected name prefixes."""
    _load_backend()
    from numpyro.diagnostics import summary

    statistics = summary(mcmc.get_samples(group_by_chain=True), prob=0.9)
    extra_fields = mcmc.get_extra_fields()
    divergences = int(np.asarray(extra_fields["diverging"]).sum()) if "diverging" in extra_fields else None
    return format_diagnostics(statistics, names=names, divergences=divergences)


def _q(x, lo=5, hi=95):
    return np.percentile(np.asarray(x), [lo, 50, hi])


def factorial_contrasts(delta_fto, delta_writer, delta_both, *,
                       equivalence_log_margin: float | None = None,
                       equivalence_justification: str | None = None) -> dict:
    """Summarize nonadditivity and the conditional FTO effect separately.

    Effects are posterior draws of log decay-rate ratios relative to the same
    reference arm. A positive interaction does not establish that the FTO
    effect disappears under writer knockdown. Practical equivalence is left
    unassessed by default; a caller must supply and justify a positive margin
    on the log-ratio scale to report its posterior mass.
    """
    arrays = [np.atleast_1d(np.asarray(v, dtype=float))
              for v in (delta_fto, delta_writer, delta_both)]
    if any(v.shape != arrays[0].shape for v in arrays):
        raise ValueError("Factorial effects must contain corresponding draws with identical shapes")
    if not arrays[0].size or any(not np.isfinite(v).all() for v in arrays):
        raise ValueError("Factorial effects require nonempty finite posterior draws")
    if equivalence_log_margin is None:
        if equivalence_justification is not None:
            raise ValueError("An equivalence justification requires an explicit log-ratio margin")
    elif (not np.isfinite(equivalence_log_margin) or equivalence_log_margin <= 0
          or not isinstance(equivalence_justification, str)
          or not equivalence_justification.strip()):
        raise ValueError("Equivalence requires a finite positive log-ratio margin and its justification")
    fto, writer, both = arrays
    interaction = both - fto - writer
    conditional = both - writer
    equivalence = {"status": "not assessed; no justified equivalence margin was supplied"}
    if equivalence_log_margin is not None:
        equivalence = {
            "status": "posterior mass within a caller-specified equivalence range; no automatic decision",
            "log_ratio_margin": float(equivalence_log_margin),
            "justification": equivalence_justification.strip(),
            "conditional_fto_probability_within_margin": float(
                np.mean(np.abs(conditional) <= equivalence_log_margin)),
        }
    return {
        "interpretation": (
            "Interaction is nonadditivity on the log decay-rate scale. Conditional FTO "
            "effect is delta_both minus delta_writer. Neither an interaction sign nor "
            "its posterior probability establishes mechanistic dependence or abolition."),
        "interaction_log_p5_p50_p95": list(map(float, _q(interaction))),
        "interaction_ratio_p5_p50_p95": list(map(float, _q(np.exp(interaction)))),
        "probability_interaction_positive": float(np.mean(interaction > 0)),
        "conditional_fto_log_p5_p50_p95": list(map(float, _q(conditional))),
        "conditional_fto_ratio_p5_p50_p95": list(map(float, _q(np.exp(conditional)))),
        "probability_conditional_fto_negative": float(np.mean(conditional < 0)),
        "probability_conditional_fto_positive": float(np.mean(conditional > 0)),
        "equivalence": equivalence,
    }


ARMS_A = ["siControl", "siMETTL3", "siFTO"]

# The first arm in each source panel is its reference.
PANELS_N = {"3i": ["Control", "As-T"],
            "3m": ["WT", "FTO KO"],
            "5a": ["WT+shNC", "FTO KO+shNC", "WT+shMETTL14", "FTO KO+shMETTL14"]}
REF_N = {v[0] for v in PANELS_N.values()}


def _nedd4l_model2(t, y, panel, eff_idx, n_panel, n_eff):
    _load_backend()
    mu = numpyro.sample("mu", dist.Normal(np.log(0.10), 1.0))
    b_panel = numpyro.sample("b_panel", dist.Normal(0, 0.5).expand([n_panel - 1]))
    bp = jnp.concatenate([jnp.zeros(1), b_panel])
    delta = numpyro.sample("delta", dist.Normal(0, 1.0).expand([n_eff]))
    eff = jnp.concatenate([jnp.zeros(1), delta])      # index 0 = "is a control arm"
    k = numpyro.deterministic("k", jnp.exp(mu + bp[panel] + eff[eff_idx]))
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.3))
    numpyro.sample("obs", dist.Normal(-k * t, sigma), obs=jnp.log(y))


def fit_antioxidant(seed=0):
    _load_backend()
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
    _load_backend()
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--equivalence-log-margin", type=float, default=None,
                        help="Optional positive margin for the conditional FTO log decay-rate ratio")
    parser.add_argument("--equivalence-justification", default=None,
                        help="Scientific justification required when an equivalence margin is supplied")
    args = parser.parse_args()
    # Validate the optional decision specification before loading data/samplers.
    factorial_contrasts([0.], [0.], [0.],
                        equivalence_log_margin=args.equivalence_log_margin,
                        equivalence_justification=args.equivalence_justification)
    print("Bayesian chase fits")
    print("A decay-rate ratio above one indicates faster decay than in the matched control.")

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

    # Nonadditivity and conditional effect answer different scientific questions.
    d_fto = np.asarray(s2["delta"][:, idx["FTO KO+shNC"]])
    d_m14 = np.asarray(s2["delta"][:, idx["WT+shMETTL14"]])
    d_both = np.asarray(s2["delta"][:, idx["FTO KO+shMETTL14"]])
    contrasts = factorial_contrasts(
        d_fto, d_m14, d_both,
        equivalence_log_margin=args.equivalence_log_margin,
        equivalence_justification=args.equivalence_justification)
    lo, md, hi = contrasts["interaction_ratio_p5_p50_p95"]
    print(f"\n  2x2 interaction (FTO KO x shMETTL14), as a ratio: "
          f"{md:.2f}  [{lo:.2f}, {hi:.2f}]")
    print("  P(log interaction > 0) = "
          f"{contrasts['probability_interaction_positive']:.3f} (nonadditivity only)")
    lo, md, hi = contrasts["conditional_fto_ratio_p5_p50_p95"]
    print(f"  Conditional FTO KO effect under writer knockdown, rate ratio: "
          f"{md:.2f}  [{lo:.2f}, {hi:.2f}]")
    print(f"  Equivalence: {contrasts['equivalence']['status']}")
    if args.equivalence_log_margin is not None:
        eq = contrasts["equivalence"]
        print(f"    log-ratio margin: +/-{eq['log_ratio_margin']:g}")
        print(f"    justification: {eq['justification']}")
        print("    P(conditional FTO effect within margin) = "
              f"{eq['conditional_fto_probability_within_margin']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
