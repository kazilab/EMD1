"""Check the direction and sensitivity of responses in the assumed model graph."""
from dataclasses import dataclass
import numpy as np
from ..simulation.model import Params

# Conventional implementation threshold, not a fitted assay acceptance limit.
AOX_DECAY_RATIO_MIN = 1.5

def antioxidant_decay_rate(M_aox: float, p: Params | None = None,
                           aox_reader: float = 1.0) -> float:
    """Effective first-order decay constant for the antioxidant pool."""
    p = p or Params()
    return p.d_aox / (1.0 + p.gam_ia * aox_reader * M_aox)

# Each check compares endpoint observables on their native scales.

@dataclass(frozen=True)
class Check:
    cid: str
    edge: str
    statement: str
    test: object
    source: str


CHECK_ROLES = {
    **{f"V{i}": "encoding consistency" for i in (1, 2, 3, 4, 7, 10, 11)},
    "V5": "qualitative assumed-graph consistency",
    "V6": "structural regression",
    "V8": "qualitative source concordance",
    "V9": "indirect pathway support",
    "V12": "qualitative source concordance",
    "V13": "structural regression",
}


def _gt(a, b, margin=1.05):
    return a > b * margin


def _lt(a, b, margin=0.95):
    return a < b * margin


CHECKS = [
    Check("V1", "KCC5 -> EMD1",
          "Arsenic raises ROS and FTO activity",
          lambda e: _gt(e["as"]["R"], e["control"]["R"]) and _gt(e["as"]["Fo"], e["control"]["Fo"]),
          "As-induced ROS; FTO protein stabilised by impaired autophagy (NC 6, JBC 3A)"),

    Check("V2", "EMD1 (site-specific)",
          "Arsenic lowers m6A occupancy on APOBEC3B and NEDD4L and raises "
          "occupancy in the antioxidant pool at the illustrative parameters",
          lambda e: (_lt(e["as"]["M_a3b"], e["control"]["M_a3b"])
                     and _lt(e["as"]["M_ned"], e["control"]["M_ned"])
                     and _gt(e["as"]["M_aox"], e["control"]["M_aox"])),
          "Published NEDD4L and antioxidant data motivate these directions. "
          "The NEDD4L interval is quarantined pending coordinate verification."),

    Check("V3", "EMD1 -> KCC2",
          "Arsenic raises APOBEC3B and the illustrative cumulative fixation index",
          lambda e: _gt(e["as"]["A3B"], e["control"]["A3B"]) and _gt(e["as"]["N"], e["control"]["N"]),
          "Gao Fig 1 supports increased A3B and TP53 hypermutation qualitatively. Maxima "
          "of substitution-category bars do not calibrate total fixation or mutation burden."),

    Check("V4", "EMD1 -> KCC2 (rescue)",
          "FTO inhibition lowers APOBEC3B and the illustrative fixation index versus arsenic",
          lambda e: _lt(e["as_ftoi"]["A3B"], e["as"]["A3B"]) and _lt(e["as_ftoi"]["N"], e["as"]["N"]),
          "Gao Fig 3 supports the pathway qualitatively. Fig 3K omits arsenic exposure "
          "from its protocol description. Fig 3Q measures mutation categories."),

    Check("V5", "EMD1 -> KCC2 (qualitative perturbation)",
          "APOBEC3B knockdown lowers the fixation index while ROS is unchanged by the "
          "assumed graph; this does not discriminate causal order or test epistasis",
          lambda e: (_lt(e["as_a3bkd"]["N"], e["as"]["N"])
                     and np.isclose(e["as_a3bkd"]["R"], e["as"]["R"], rtol=1e-5, atol=1e-8)),
          "Gao Fig 1 provides qualitative A3B-knockdown evidence. The model contains "
          "no A3B-to-ROS feedback and no factorial FTO-by-A3B rescue test in this check."),

    Check("V6", "EMD1 -> KCC3 (assumed zero arsenic input)",
          "Under zero arsenic input to the repair site, the DDB2/NER proxy remains "
          "within 5% of control",
          lambda e: abs(e["as"]["Q"] / e["control"]["Q"] - 1.0) < 0.05,
          "Under arsenic, DDB2 enrichment across its first three exons was 0.91x "
          "(38th percentile among 336 quantified transcripts). XPC also fell within "
          "the background range. These measurements cannot exclude a moderate effect."),

    Check("V7", "EMD1 -> KCC10",
          "At the illustrative parameters arsenic lowers NEDD4L, de-represses WNT and "
          "raises relative population expansion; the expansion sign is sensitivity-dependent",
          lambda e: (_lt(e["as"]["NEDD4L"], e["control"]["NEDD4L"])
                     and _gt(e["as"]["WNT"], e["control"]["WNT"])
                     and e["as"]["lnP"] > e["control"]["lnP"]),
          "NC 3e: NEDD4L 0.68x; NC 1e: proliferation 1.5x"),

    Check("V8", "EMD1 -> KCC10 (rescue)",
          "FTO inhibition restores NEDD4L and suppresses expansion",
          lambda e: (_gt(e["as_ftoi"]["NEDD4L"], e["as"]["NEDD4L"])
                     and e["as_ftoi"]["lnP"] < e["as"]["lnP"]),
          "NC 3l: NEDD4L 1.66x in FTO KO; NC Fig 2: FTO KO reduces tumour growth"),

    Check("V9", "EMD1 -> KCC10 (indirect pathway support)",
          "Simulated NEDD4L re-expression suppresses expansion at unchanged FTO activity",
          lambda e: e["as_nedresc"]["lnP"] < e["as"]["lnP"],
          "Cui Fig 4 knocks down NEDD4L in FTO-deleted cells and partially reverses "
          "the FTO-deletion phenotype. It does not validate NEDD4L re-expression "
          "with FTO unchanged."),

    Check("V10", "feedback loop",
          "At the illustrative parameters, arsenic increases antioxidant abundance. "
          "Writer knockdown under arsenic lowers abundance and expansion. The direction "
          "of the expansion response varies across parameter draws",
          lambda e: (_gt(e["as"]["A"], e["control"]["A"])
                     and _lt(e["as_m3kd"]["A"], e["as"]["A"])
                     and e["as_m3kd"]["lnP"] < e["as"]["lnP"]),
          "Zhao Fig 5B/S5b/S5c reports antioxidant levels of 0.61–0.65x with "
          "siMETTL3 and 0.58–0.66x with STM2457."),

    Check("V11", "EMD1 -> antioxidant (mechanism)",
          "Writer knockdown raises the writer-associated aggregate decay rate by at least "
          f"{AOX_DECAY_RATIO_MIN:.1f}x, a conventional encoding threshold; "
          "the distinct PRDX5 reader mechanism is not represented",
          lambda e: (
              antioxidant_decay_rate(e["as_m3kd"]["M_aox"])
              / antioxidant_decay_rate(e["as"]["M_aox"])
              >= AOX_DECAY_RATIO_MIN
          ),
          "Zhao Fig 5B supports writer-associated stabilisation for SOD1, SOD2, CAT, "
          "TXN and GPX1. PRDX5 follows a distinct FTO/YTHDF2 pathway."),

    Check("V12", "reader-specific divergence",
          "FTO inhibition raises occupancy on both transcripts. YTHDF2-mediated decay "
          "lowers APOBEC3B, while IGF2BP-mediated stabilisation increases NEDD4L",
          lambda e: ((e["as_ftoi"]["M_a3b"] > e["as"]["M_a3b"])
                     and (e["as_ftoi"]["M_ned"] > e["as"]["M_ned"])
                     and (e["as_ftoi"]["A3B"] < e["as"]["A3B"])
                     and (e["as_ftoi"]["NEDD4L"] > e["as"]["NEDD4L"])),
          "Gao Fig 3E reports faster A3B decay after FTO knockdown; Cui Fig 3m "
          "reports slower NEDD4L decay after FTO knockout in a different biological "
          "background. The gains are not quantitatively calibrated by this check. "
          "Opposite signs do not exclude a shared driver with different readers."),

    # V13 uses control-normalised repair readouts in check_repair_edge.
]


def ordinal(n: float) -> str:
    """Format a rounded percentile as an English ordinal."""
    i = int(round(n))
    suf = "th" if 11 <= i % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(i % 10, "th")
    return f"{i}{suf}"


def check_repair_edge(m_rep_as: float, q_as: float,
                      m_rep_wkd: float, q_wkd: float) -> tuple[bool, str]:
    """Test the repair proxy under the assumed zero arsenic input.

    Writer knockdown must lower both occupancy and the proxy by more than 5%.
    Under arsenic alone both must match control within numerical tolerance.
    """
    present = (m_rep_wkd < 0.95) and (q_wkd < 0.95)
    arsenic_input_assumed_zero = bool(
        np.isclose(m_rep_as, 1.0, rtol=1e-5, atol=1e-8)
        and np.isclose(q_as, 1.0, rtol=1e-5, atol=1e-8))
    return bool(present and arsenic_input_assumed_zero), (
        f"live writer response={bool(present)}: M_rep={m_rep_wkd:.6g}x, Q={q_wkd:.6g}x "
        f"(both must be below 0.95), canonical zero arsenic response="
        f"{arsenic_input_assumed_zero}: M_rep={m_rep_as:.6g}x, Q={q_as:.6g}x "
        f"(both must equal 1 within rtol=1e-5, atol=1e-8).")


def validate(runs: dict, p: Params | None = None) -> list[tuple]:
    endpoints = {k: {o: float(v[-1]) for o, v in tr.items()} for k, tr in runs.items()}
    results = []
    for chk in CHECKS:
        try:
            if chk.cid == "V11" and p is not None:
                passed = bool(antioxidant_decay_rate(endpoints["as_m3kd"]["M_aox"], p)
                              / antioxidant_decay_rate(endpoints["as"]["M_aox"], p)
                              >= AOX_DECAY_RATIO_MIN)
            else:
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


def evaluate_draw_checks(p: Params, runs: dict) -> dict[str, bool]:
    """Score one draw while its native trajectories and parameters are available."""
    scores = {check.cid: passed for check, passed in validate(runs, p)}
    control = runs["control"]
    try:
        scores["V13"], _ = check_repair_edge(
            float(runs["as"]["M_rep"][-1] / control["M_rep"][-1]),
            float(runs["as"]["Q"][-1] / control["Q"][-1]),
            float(runs["as_m3kd"]["M_rep"][-1] / control["M_rep"][-1]),
            float(runs["as_m3kd"]["Q"][-1] / control["Q"][-1]))
    except Exception:
        scores["V13"] = False
    return scores
