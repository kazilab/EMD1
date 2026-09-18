"""Run the EMD1 simulation: nominal trajectories, validation battery, ensemble.

    python -m emd1_simulation.run [--n-ensemble 400] [--no-figure]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dataclasses import replace

from .conditions import (CHECKS, CHECK_ROLES, CONDITIONS, DOSE, OUT_OF_CALIBRATION_CHECK_KEYS, PLOTTED, REPAIR_DDB2,
                         REPAIR_ARSENIC_WRITER_INPUT, REPAIR_FEATURE_ACCESSION,
                         REPAIR_FEATURE_CHROM, REPAIR_FEATURE_EXONS_0BASED,
                         REPAIR_FEATURE_STRAND, REPAIR_INPUT_QC_BIN_NT,
                         REPAIR_MIN_CONTROL_INPUT_BINS, REPAIR_MIN_CONTROL_INPUT_DENSITY,
                         REPAIR_N_BACKGROUND, REPAIR_NULL_BAND,
                         REPAIR_NULL_BY_CONTRAST,
                         REPAIR_N_QUANTIFIED, REPAIR_N_REPAIR_QUANTIFIED,
                         REPAIR_XPC, Check, check_repair_edge)
from .ensemble import run_ensemble
from .model import (Intervention, Params, control_steady_state, normalise_params,
                    simulate, simulate_protocol, trace)
from .scientific_audit import (antioxidant_branch_reconciliation, evidence_audit,
                               population_summary, repair_scenarios)

T_END = 45.0
N_T = 451
OBSERVABLES = ["R", "Fo", "Ab", "M_a3b", "M_ned", "M_rep", "M_aox", "M_mean4",
               "A3B", "NEDD4L", "A", "Q", "WNT", "L", "N", "lnP"]


def run_nominal(p: Params, t: np.ndarray, intervention_start: float = 0.0) -> dict:
    p = normalise_params(p)
    y0 = control_steady_state(p)
    runs = {}
    for c in CONDITIONS:
        y = simulate_protocol(p, c.iv, c.E, t, intervention_start, y0)
        tr = trace(y, p, c.iv)
        if intervention_start > 0:
            before = t < intervention_start
            pre = trace(y[:, before], p, Intervention())
            for key in tr:
                tr[key][before] = pre[key]
        runs[c.key] = tr
    return runs


def _cf_fold(q: Params, t: np.ndarray) -> dict:
    """One counterfactual draw, as fold change over its OWN control.

    The arm changes Params, not just intervention handles, so it cannot share
    the nominal control fixed point: it gets its own burn-in and normalisation.
    """
    y0 = control_steady_state(q)
    ctl = trace(simulate(q, Intervention(), 0.0, t, y0=y0), q, Intervention())
    ars = trace(simulate(q, Intervention(), DOSE, t, y0=y0), q, Intervention())
    out = {}
    for o in ("M_rep", "Q", "N"):
        # N is a cumulative counter and is 0 at t = 0; the limit there is the
        # ratio of rates, which is 1 because both arms start from the same
        # fixed point. Same guard as figure._ratio.
        r = np.ones_like(ars[o], dtype=float)
        np.divide(ars[o], ctl[o], out=r, where=np.abs(ctl[o]) > 1e-12)
        out[o] = r
    return out


def run_counterfactual(p: Params, t: np.ndarray, n: int = 0, seed: int = 20260825) -> dict:
    """"What if arsenic DID drive the repair site?" -- one parameter changed.

    Reported with its own uncertainty band, not as a single line. Under the
    model's own stress-test scheme the distribution overlaps the measured
    cross-transcript null, so a nominal trace alone would overstate how cleanly
    the alternative is excluded. This is a heuristic sensitivity band, not a
    confidence or credible interval.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}")
    p = normalise_params(p)
    out = _cf_fold(replace(p, **REPAIR_ARSENIC_WRITER_INPUT), t)
    if n:
        from .ensemble import sample_params
        rng = np.random.default_rng(seed)
        draws = {o: [] for o in ("M_rep", "Q")}
        for _ in range(n):
            try:
                d = _cf_fold(replace(sample_params(rng, base=p),
                                     **REPAIR_ARSENIC_WRITER_INPUT), t)
            except Exception:
                continue
            for o in draws:
                draws[o].append(d[o])
        for o, v in draws.items():
            if not v:
                raise RuntimeError("no valid counterfactual draws")
            a = np.asarray(v)
            out[o + "_lo"], out[o + "_hi"] = np.percentile(a, [5, 95], axis=0)
            out[o + "_n"] = len(v)
    return out


def validate(runs: dict) -> list[tuple]:
    endpoints = {k: {o: float(v[-1]) for o, v in tr.items()} for k, tr in runs.items()}
    results = []
    for chk in CHECKS:
        try:
            passed = bool(chk.test(endpoints))
        except Exception as exc:                      # a malformed check must not
            passed = False                            # masquerade as a failed model
            print(f"  !! {chk.cid} raised: {exc}")
        results.append((chk, passed))
    return results


def v10_ensemble_sensitivity(ens: dict) -> dict:
    """Audit the three-part V10 claim across already-normalised ensemble draws.

    Fold normalisation preserves the two abundance comparisons, and subtracting
    each draw's control lnP preserves the clonal-expansion ordering.  This is a
    sensitivity diagnostic, not a validation probability.
    """
    as_a = np.asarray(ens["as"]["A"][:, -1])
    kd_a = np.asarray(ens["as_m3kd"]["A"][:, -1])
    as_p = np.asarray(ens["as"]["lnP"][:, -1])
    kd_p = np.asarray(ens["as_m3kd"]["lnP"][:, -1])
    adaptation = as_a > 1.05
    writer_lowers_a = kd_a < 0.95 * as_a
    writer_lowers_p = kd_p < as_p
    passed = adaptation & writer_lowers_a & writer_lowers_p
    difference = kd_p - as_p
    return {
        "interpretation": (
            "Heuristic parameter-stress-test frequency, not a probability of truth or "
            "an additional validation."),
        "draws": int(passed.size),
        "passes": int(passed.sum()),
        "pass_fraction": float(passed.mean()),
        "arsenic_adaptation_pass_fraction": float(adaptation.mean()),
        "writer_kd_lowers_antioxidant_pass_fraction": float(writer_lowers_a.mean()),
        "writer_kd_lowers_expansion_vs_arsenic_pass_fraction": float(writer_lowers_p.mean()),
        "writer_kd_log2_expansion_p5_p50_p95": list(map(
            float, np.percentile(kd_p, [5, 50, 95]))),
        "writer_kd_minus_arsenic_log2_expansion_p5_p50_p95": list(map(
            float, np.percentile(difference, [5, 50, 95]))),
        "robust_at_5th_95th_sensitivity_level": bool(np.percentile(difference, 95) < 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ensemble", type=int, default=400)
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--outdir", default="figures/output")
    ap.add_argument("--strict-evidence", action="store_true",
                    help="exit nonzero while the scientific evidence audit remains unresolved")
    args = ap.parse_args()
    if args.n_ensemble <= 0:
        ap.error("--n-ensemble must be a positive integer")

    p = Params()
    t = np.linspace(0.0, T_END, N_T)

    print("=" * 78)
    print("EMD1 mechanistic simulation   KCC5 -> EMD1/KCC4 -> KCC2, KCC3, KCC10")
    print("public-data-constrained proof of concept; arsenic as exemplar exposure")
    print("=" * 78)

    runs = run_nominal(p, t)
    ctl = {o: float(runs["control"][o][-1]) for o in OBSERVABLES}

    # --- endpoint table --------------------------------------------------
    print(f"\nEndpoints at nominal model day {T_END:.0f}, as fold change over the control arm")
    print("(KCC10 row is log2 relative clonal expansion, so control = 0.00)\n")
    cols = ["R", "Fo", "M_a3b", "M_ned", "M_rep", "M_aox", "M_mean4",
            "A3B", "NEDD4L", "A", "Q", "WNT", "N", "lnP"]
    hdr = f"{'condition':28s}" + "".join(f"{c:>9s}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for c in CONDITIONS:
        row = f"{c.label:28s}"
        for o in cols:
            v = float(runs[c.key][o][-1])
            if o == "lnP":
                row += f"{(v - runs['control']['lnP'][-1]) / np.log(2):>9.2f}"
            else:
                row += f"{v / ctl[o]:>9.2f}"
        if c.plotted:
            suffix = ""
        elif c.key in OUT_OF_CALIBRATION_CHECK_KEYS:
            suffix = "   (held out of numerical calibration; qualitative sign check)"
        else:
            suffix = "   (held out; unvalidated projection)"
        print(row + suffix)

    # --- model checks ----------------------------------------------------
    print("\n" + "=" * 78)
    print("Model-check register (roles distinguish calibration from validation)")
    print("Conditions marked (held out) above are not used to set any parameter.")
    print("Two have out-of-calibration sign checks; ALKBH5 and antioxidant-reader knockdown are")
    print("unvalidated projections.")
    print("=" * 78)
    results = validate(runs)
    for chk, passed in results:
        print(f"\n[{'PASS' if passed else 'FAIL'}] {chk.cid}  [{CHECK_ROLES[chk.cid]}]  {chk.edge}")
        print(f"       {chk.statement}")
        print(f"       basis: {chk.source}")
    # --- V13: live DDB2/NER edge; nominal arsenic writer input is zero ----
    cf = run_counterfactual(p, t, n=args.n_ensemble)
    m_rep_as = float(runs["as"]["M_rep"][-1]) / ctl["M_rep"]
    q_as = float(runs["as"]["Q"][-1]) / ctl["Q"]
    m_rep_wkd = float(runs["as_m3kd"]["M_rep"][-1]) / ctl["M_rep"]
    q_wkd = float(runs["as_m3kd"]["Q"][-1]) / ctl["Q"]
    v13, v13_msg = check_repair_edge(m_rep_as, q_as, m_rep_wkd, q_wkd)
    basis = (f"Measured alongside (an observation, not a test of this model): "
             f"annotation-fixed first-three-exon DDB2 is {REPAIR_DDB2['writer_kd'][0]:.2f}x under "
             f"writer knockdown and {REPAIR_DDB2['arsenic'][0]:.2f}x under arsenic, "
             f"against {REPAIR_N_QUANTIFIED['writer_kd']} and "
             f"{REPAIR_N_QUANTIFIED['arsenic']} quantified background transcripts. "
             f"XPC moves {REPAIR_XPC['writer_kd']:.2f}x and "
             f"{REPAIR_XPC['arsenic']:.2f}x. Both DDB2 contrasts remain inside their "
             f"cross-transcript nulls; this track reanalysis is not a positive control "
             f"for the modelled writer response. See fit/repair_edge.py")
    results.append((Check(
        "V13", "EMD1 -> KCC3  (EDGE LIVE; ARSENIC INPUT ASSUMED ZERO)",
        v13_msg, None, basis), v13))
    print(f"\n[{'PASS' if v13 else 'FAIL'}] V13  [{CHECK_ROLES['V13']}]  "
          "EMD1 -> KCC3 (EDGE LIVE; ARSENIC INPUT ASSUMED ZERO)")
    print("       Structural check. The DDB2/NER proxy responds to writer perturbation;")
    print("       the nominal arsenic arm encodes zero writer input to that site.")
    print("       The measured cross-transcript null is too wide to exclude a moderate effect:")
    print(f"       {v13_msg}")
    print(f"       basis: {basis}")
    tr, tr_pct = REPAIR_DDB2["transformed"]
    print(f"       note: in the arsenic-TRANSFORMED line DDB2 falls to {tr:.2f}x "
          f"(background percentile {tr_pct:.1f}), below its contrast-specific null.")
    print("             This is exploratory and outside the 45-day window modelled here.")

    n_pass = sum(ok for _, ok in results)
    heldout = [(chk, ok) for chk, ok in results
               if CHECK_ROLES[chk.cid] == "out-of-calibration qualitative check"]
    print(f"\n  {n_pass}/{len(results)} model checks passed")
    print(f"  {sum(ok for _, ok in heldout)}/{len(heldout)} out-of-calibration "
          "qualitative checks passed (not blinded prospective validations)")

    # --- ensemble --------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"Heuristic sensitivity propagation ({args.n_ensemble} independent log-normal draws)")
    print("=" * 78)
    ens, n_ok, check_sensitivity = run_ensemble(
        CONDITIONS, OBSERVABLES, t, n=args.n_ensemble, audit_checks=True)
    print(f"  {n_ok}/{args.n_ensemble} draws integrated successfully\n")
    v10_sensitivity = v10_ensemble_sensitivity(ens)
    print("  V10 writer-disruption sensitivity: "
          f"{v10_sensitivity['passes']}/{v10_sensitivity['draws']} draws retain all "
          "three nominal signs")
    if v10_sensitivity["robust_at_5th_95th_sensitivity_level"]:
        print("  Writer-KD expansion suppression persists across the 5th-95th sensitivity range.")
    else:
        print("  Writer-KD expansion suppression is not robust at the 5th-95th sensitivity level.")
    print("  Relative expansion does not establish net cell loss or measured apoptosis.\n")
    print("  Per-check stress-test retention (not validation probabilities):")
    for cid, item in check_sensitivity["checks"].items():
        print(f"    {cid:>3s}: {item['passes']:>3d}/{check_sensitivity['draws']} "
              f"({100 * item['pass_fraction']:5.1f}%)")
    print()

    # Observables held exactly at the null by structural choices (arsenic-null
    # repair edge, ALKBH5 flat) are not "non-robust signs" — they are pinned.
    NULL_BY_DESIGN = {"M_rep", "Q", "Ab"}
    print(f"  {'observable':12s} {'arsenic arm, day 45':>34s}   sign robust?")
    print(f"  {'':12s} {'median [5th, 95th pct]':>34s}")
    for o in ["R", "M_a3b", "M_ned", "M_rep", "M_aox", "M_mean4",
              "A3B", "NEDD4L", "A", "Q", "N", "lnP"]:
        arr = ens["as"][o][:, -1]
        lo, med, hi = np.percentile(arr, [5, 50, 95])
        null = 0.0 if o == "lnP" else 1.0
        if o in NULL_BY_DESIGN:
            robust = "null (by design)"
        elif lo > null or hi < null:
            robust = "yes"
        else:
            robust = "NO"
        print(f"  {o:12s} {med:12.2f}  [{lo:7.2f}, {hi:7.2f}]   {robust}")

    # --- persist ---------------------------------------------------------
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    scientific = evidence_audit(p, runs)
    population = population_summary(runs, t)
    scenarios = repair_scenarios(p, t)
    recon = antioxidant_branch_reconciliation(p, runs, T_END)
    scientific["antioxidant_branch_reconciliation"] = recon
    delayed = run_nominal(p, t, intervention_start=30.0)
    print("\nScientific evidence audit: unresolved; nominal checks are not quantitative validation.")
    print("  FTO antioxidant decay, model ratio "
          f"{recon['model_ftoi_over_arsenic']:.3f} against the chase, by exposure stage")
    print("  (writer branch = SOD1/SOD2/CAT/TXN/GPX1; PRDX5 is on the eraser branch):")
    for stage, d in recon["by_stage"].items():
        mark = " <- nearest the model horizon" if stage == recon["nearest_chase_stage"] else ""
        print(f"    {stage:7s} median {d['writer_branch_median_siFTO']:.2f}x  "
              f"mean log-ratio {d['writer_branch_mean_log_ratio']:+.3f}  "
              f"p={d['writer_branch_p_vs_no_effect']:.3f}  "
              f"({d['writer_branch_n_faster']}/{d['writer_branch_n']} faster)"
              f"  PRDX5 {d['eraser_branch_siFTO']['PRDX5']:.2f}x{mark}")
    print(f"    pooled median over all six genes and all stages: "
          f"{recon['pooled_median_all_genes_all_stages']:.3f}x -- not a reliable summary")
    print(f"    matched stage contradicts the model: {recon['matched_stage_contradicts_model']}")
    print("  Writer-KD population fold vs start: "
          f"{population['as_m3kd']['population_fold_vs_start']:.3f}.")
    summary = {
        "schema_version": 2,
        "time_interpretation": "nominal model days; illustrative, not an experimentally calibrated protocol",
        "scientific_evidence_audit": scientific,
        "population_diagnostics": population,
        "repair_structural_scenarios": scenarios,
        "delayed_intervention_scenario": {
            "intervention_start_nominal_day": 30.0,
            "interpretation": "Illustrative delayed intervention, not a matched experimental protocol",
            "population_diagnostics": population_summary(delayed, t)},
        "t_end_days": T_END,
        "endpoints_fold_vs_control": {
            c.key: {o: (float(runs[c.key][o][-1] - runs["control"]["lnP"][-1]) / np.log(2)
                        if o == "lnP" else float(runs[c.key][o][-1]) / ctl[o])
                    for o in cols}
            for c in CONDITIONS},
        "endpoint_units": {o: ("log2 fold vs control" if o == "lnP" else "fold vs control")
                           for o in cols},
        "model_checks": {chk.cid: {"role": CHECK_ROLES[chk.cid], "edge": chk.edge,
                                    "statement": chk.statement, "source": chk.source,
                                    "passed": ok}
                         for chk, ok in results},
        "model_checks_passed": n_pass,
        "out_of_calibration_checks_passed": sum(ok for _, ok in heldout),
        "out_of_calibration_checks_total": len(heldout),
        "ensemble": {o: dict(zip(("p5", "p50", "p95"),
                                 map(float, np.percentile(ens["as"][o][:, -1], [5, 50, 95]))))
                     for o in OBSERVABLES},
        "ensemble_draws_ok": n_ok,
        "ensemble_interpretation": (
            "Independent log-normal parameter stress test; heuristic 5th-95th "
            "sensitivity bands, not confidence intervals, credible intervals, or "
            "a calibration-conditioned posterior."),
        "ensemble_check_sensitivity": {"V10": v10_sensitivity},
        "ensemble_model_check_sensitivity": check_sensitivity,
        "repair_edge": {
            "model_arsenic": {"M_rep_fold": m_rep_as, "Q_fold": q_as},
            "model_writer_kd": {"M_rep_fold": m_rep_wkd, "Q_fold": q_wkd},
            "counterfactual_params": REPAIR_ARSENIC_WRITER_INPUT,
            "counterfactual_M_rep_fold": float(cf["M_rep"][-1]),
            "counterfactual_Q_fold": float(cf["Q"][-1]),
            "counterfactual_N_fold": float(cf["N"][-1]),
            "counterfactual_draws_ok": int(cf.get("M_rep_n", 0)),
            "counterfactual_Q_p5_p95": (
                [float(cf["Q_lo"][-1]), float(cf["Q_hi"][-1])] if "Q_lo" in cf else None),
            "counterfactual_M_rep_p5_p95": (
                [float(cf["M_rep_lo"][-1]), float(cf["M_rep_hi"][-1])] if "M_rep_lo" in cf else None),
            "measured_ddb2_first_three_exons": {
                k: {"effect": v[0], "background_pct": v[1]}
                for k, v in REPAIR_DDB2.items()},
            "measured_xpc_same_pathway_comparator": REPAIR_XPC,
            "cross_transcript_null_5_95": list(REPAIR_NULL_BAND),
            "cross_transcript_null_by_contrast": {k: list(v) for k, v in REPAIR_NULL_BY_CONTRAST.items()},
            "n_background_sampled": REPAIR_N_BACKGROUND,
            "n_background_quantified": REPAIR_N_QUANTIFIED,
            "n_repair_quantified": REPAIR_N_REPAIR_QUANTIFIED,
            "repair_feature_definition": (
                "annotation-only, strand-aware first three exons of the longest "
                "spliced hg19 RefSeq NM_ isoform; identical across conditions and genes"),
            "ddb2_feature": {
                "accession": REPAIR_FEATURE_ACCESSION,
                "chrom": REPAIR_FEATURE_CHROM,
                "strand": REPAIR_FEATURE_STRAND,
                "exon_intervals_0based_half_open": [list(x) for x in REPAIR_FEATURE_EXONS_0BASED],
                "exonic_nt": sum(end - start for start, end in REPAIR_FEATURE_EXONS_0BASED),
            },
            "control_input_qc": {
                "fixed_bin_nt": REPAIR_INPUT_QC_BIN_NT,
                "minimum_passing_bins_per_control_replicate": REPAIR_MIN_CONTROL_INPUT_BINS,
                "minimum_library_normalised_density": REPAIR_MIN_CONTROL_INPUT_DENSITY,
                "eligibility_uses_treatment_tracks": False,
            },
            "writer_kd_track_reanalysis_supports_decrease": False,
            "edge_live_arsenic_input_assumed_zero": v13,
        },
    }
    (outdir / "emd1_simulation_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {outdir / 'emd1_simulation_summary.json'}")

    if not args.no_figure:
        from .figure import build_figure
        paths = build_figure(runs, ens, t, outdir, cf=cf, n_ensemble=n_ok)
        for q in paths:
            print(f"  wrote {q}")

    if n_pass != len(results):
        return 1
    return 2 if args.strict_evidence and not scientific["quantitatively_validated"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
