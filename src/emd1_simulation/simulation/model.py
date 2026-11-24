"""Model of oxidative stress, transcript-specific m6A and downstream KCC responses.

Published studies motivate the directions of selected interactions. Parameter
values and nominal model days are illustrative. By default, exposure and
intervention start together at time zero.

The model tracks m6A occupancy on four targets. Increased occupancy promotes
APOBEC3B decay through YTHDF2, NEDD4L stabilisation through IGF2BP1–3, repair
protein translation through YTHDF1, and stabilisation of a writer-associated
antioxidant pool. PRDX5 has a distinct FTO/YTHDF2 mechanism outside this pool.
Reader-specific effects can produce opposite abundance changes even under a
shared methylation driver.
"""

from __future__ import annotations

from dataclasses import dataclass, replace, fields

import numpy as np
from scipy.integrate import solve_ivp

# State vector
IDX = {
    "R":     0,   # oxidative stress / ROS                       -> KCC5
    "W":     1,   # METTL3/METTL14 writer activity
    "Fo":    2,   # FTO state before intervention scaling (chronic, slow)
    "Ab":    3,   # ALKBH5 state before intervention scaling (ERK/JNK, fast)
    "M_a3b": 4,   # m6A occupancy, APOBEC3B                      -.
    "M_ned": 5,   # m6A occupancy, NEDD4L                          |-> EMD1 / KCC4
    "M_rep": 6,   # m6A occupancy, DNA-repair module               |
    "M_aox": 7,   # m6A occupancy, antioxidant module            -'
    "T_a3b": 8,   # APOBEC3B abundance
    "T_ned": 9,   # NEDD4L abundance
    "T_rep": 10,  # repair-module transcript abundance
    "T_aox": 11,  # antioxidant-module transcript abundance
    "L":     12,  # unrepaired lesion load (gammaH2AX analogue); feeds N
    "N":     13,  # cumulative fixation index (arbitrary units)  -> KCC2
    "lp":    14,  # log relative population expansion            -> KCC10
}
N_STATE = len(IDX)

ANTIOXIDANT_SCOPE = {
    "representation": "writer_associated_stabilising_aggregate",
    "source_branch_genes": ["SOD1", "SOD2", "CAT", "TXN", "GPX1"],
    "unrepresented_branch_genes": ["PRDX5"],
    "branch_specific_predictions_supported": False,
    "eraser_blend_interpretation": (
        "Assumed pooled eraser sensitivity; it does not implement a separate "
        "PRDX5/FTO/YTHDF2 decay branch."),
    "calibration_status": "illustrative magnitude; no matched aggregate-assay calibration",
}


@dataclass(frozen=True)
class Params:
    """Illustrative relative parameters and control normalisations."""

    # --- exposure -> ROS (KCC5) -----------------------------------------
    r_basal: float = 0.90      # basal ROS production
    k_E: float = 3.00          # arsenic -> ROS gain
    k_R: float = 1.00          # constitutive ROS clearance
    k_A: float = 2.00          # antioxidant-dependent ROS clearance (the feedback)

    # --- ROS -> writer / eraser effectors --------------------------------
    R_ref: float = 0.30        # resting ROS, derived for each draw as r_basal / (k_R + k_A)
    a_W: float = 1.41          # illustrative METTL3/14 induction amplitude
    K_W: float = 0.25
    tau_W: float = 1.50
    a_F: float = 3.50          # assumed FTO accumulation amplitude
    K_F: float = 0.20
    n_F: float = 2.00
    tau_F: float = 8.00        # slow FTO accumulation through impaired autophagic degradation
    a_B: float = 0.00          # assumed zero activity response under chronic arsenic
    K_B: float = 0.18
    tau_B: float = 0.50        # fast: post-translational (ERK/JNK) modification

    # --- site-specific m6A occupancy -------------------------------------
    tau_M: float = 0.50        # m6A write/erase turnover timescale
    M0_a3b: float = 0.35       # control occupancies
    M0_ned: float = 0.40
    M0_rep: float = 0.30
    M0_aox: float = 0.30
    w_aox_fto: float = 0.15    # assumed FTO share of aggregate eraser input
    w_rep_writer: float = 0.00 # assumed share of arsenic-induced writer activity at the repair site
    w_rep_fto: float = 0.00    # assumed FTO share of repair-site erasure
    repair_proxy_weight: float = 0.00  # assumed transfer from DDB2 to aggregate lesion repair

    # --- reader-specific coupling gains ----------------------------------
    gam_d2: float = 10.00      # assumed YTHDF2-dependent decay gain
    gam_i2: float = 5.00       # assumed IGF2BP stabilisation gain; NEDD4L
    gam_f1r: float = 2.00      # assumed DDB2 translation gain through YTHDF1
                               # Perturbation and rescue support this coupling (PMID 34452996).
    gam_ia: float = 15.0       # assumed aggregate stabilisation gain;
                               # source decay contrasts are not a matched fit

    # --- transcript turnover ---------------------------------------------
    s_a3b: float = 4.500       # d_a3b * (1 + gam_d2 * M0_a3b); control abundance 1
    d_a3b: float = 1.00
    s_ned: float = 1.0 / 3.0   # d_ned / (1 + gam_i2 * M0_ned); control abundance 1
    d_ned: float = 1.00
    s_rep: float = 1.00
    d_rep: float = 1.00
    s_aox: float = 1.0 / 5.5   # d_aox / (1 + gam_ia * M0_aox); control abundance 1
    d_aox: float = 1.00
    nrf2: float = 0.270        # assumed m6A-independent ROS induction of aggregate

    # --- KCC2 / KCC3: lesion load and fixed mutations ---------------------
    # These rates are illustrative. Empirical calibration requires aggregate
    # mutation counts, assay denominators and a mapping to the fixation index.
    k_ROS: float = 1.0672985348329618  # direct oxidative lesion rate
    k_A3B: float = 1.3956264414264556  # APOBEC3B-attributable lesion rate
    k_rep: float = 2.7159806534354018  # repair flux (saturable, scaled by Q_lesion)
    K_L: float = 0.5768266984210220
    k_fix: float = 0.2136534839667953  # persistent lesions -> fixed mutations.
                               # It affects ratios but is not empirically identified.

    # --- KCC10: WNT / proliferation-survival ------------------------------
    K_N: float = 0.80          # NEDD4L -| WNT/beta-catenin
    h_N: float = 2.50
    wnt_min: float = 0.00      # lower bound on WNT activity
    k_prol: float = 0.0373
    d0: float = 0.0061         # basal turnover
    k_apo: float = 0.0853      # assumed redox-imbalance death gain
    K_Rd: float = 1.4036       # redox-balance (R/A) tolerance threshold
    n_Rd: float = 5.2018       # threshold, not a gradient
    k_dam: float = 0.0019      # assumed contribution of lesion load to the death rate
    K_Ld: float = 2.20
    n_Ld: float = 3.00

    def __post_init__(self):
        for name in ("w_rep_writer", "w_rep_fto", "repair_proxy_weight"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a fraction in [0, 1]")


def normalise_params(p: Params) -> Params:
    """Derive resting ROS and the synthesis rates that give unit control abundance."""
    return replace(
        p, R_ref=p.r_basal / (p.k_R + p.k_A),
        s_a3b=p.d_a3b * (1.0 + p.gam_d2 * p.M0_a3b),
        s_ned=p.d_ned / (1.0 + p.gam_i2 * p.M0_ned),
        s_rep=p.d_rep,
        s_aox=p.d_aox / (1.0 + p.gam_ia * p.M0_aox),
    )


def apobec3b_bounds(p: Params, reference_occupancy: float | None = None,
                    synthesis_multiplier: float = 1.0) -> dict:
    """Exact steady-state bounds for this fixed-synthesis, unit-reader mechanism.

    FTO loss changes occupancy within [0, 1]. With reader activity fixed at one,
    its largest possible decay-rate ratio is (1+gamma)/(1+gamma*M_reference).
    The reciprocal, multiplied by a specified synthesis change, bounds abundance.
    The default reference is the assumed unexposed equilibrium, not an observed
    methylation stoichiometry. These are model constraints, not biological limits.
    """
    m = p.M0_a3b if reference_occupancy is None else float(reference_occupancy)
    if not np.isfinite(m) or not 0 <= m <= 1:
        raise ValueError("reference_occupancy must be finite and in [0, 1]")
    if not np.isfinite(p.gam_d2) or p.gam_d2 < 0:
        raise ValueError("gam_d2 must be finite and non-negative")
    if not np.isfinite(synthesis_multiplier) or synthesis_multiplier < 0:
        raise ValueError("synthesis_multiplier must be finite and non-negative")
    upper = (1.0 + p.gam_d2) / (1.0 + p.gam_d2 * m)
    return {
        "reference_occupancy": float(m),
        "decay_ratio_upper": float(upper),
        "abundance_ratio_lower": float(synthesis_multiplier / upper),
        "synthesis_multiplier": float(synthesis_multiplier),
        "scope": "steady-state bound of the specified occupancy-dependent decay model",
        "assumptions": ["unit YTHDF2 reader activity", "occupancy in [0, 1]",
                        "same basal decay constant", "specified synthesis multiplier"],
    }


@dataclass(frozen=True)
class Intervention:
    """Multiplicative perturbations; 1.0 = untouched.

    These abstract handles are not automatically matched to a published assay's
    cell type, exposure, duration, efficiency or molecular specificity. Source
    comparisons must state that protocol mapping explicitly.
    """

    fto: float = 1.0
    mettl3: float = 1.0
    alkbh5: float = 1.0
    ythdf1: float = 1.0
    ythdf2: float = 1.0
    igf2bp: float = 1.0
    aox_reader: float = 1.0    # effective pooled stabilising-reader activity;
                               # not an identified single molecular intervention
    a3b_expr: float = 1.0
    ned_expr: float = 1.0


NO_IV = Intervention()


def _hill(x: float, k: float, n: float = 1.0) -> float:
    if x <= 0.0:
        return 0.0
    xn = x ** n
    return xn / (k ** n + xn)


def readouts(y: np.ndarray, p: Params, iv: Intervention) -> dict:
    """Derived quantities: the reader layer plus the three KCC outputs.

    Capacity proxies have a control value of one. Division, death and net-growth
    rates retain their native units; conversion to contrasts happens separately.
    """
    M_rep, M_aox = y[IDX["M_rep"]], y[IDX["M_aox"]]
    T_rep, T_aox, T_ned = y[IDX["T_rep"]], y[IDX["T_aox"]], y[IDX["T_ned"]]

    # m6A affects the antioxidant pool through transcript decay.
    A = T_aox
    # YTHDF1 couples repair-site occupancy to the DDB2/NER translation proxy.
    Q = T_rep * (1 + p.gam_f1r * iv.ythdf1 * M_rep) / (1 + p.gam_f1r * p.M0_rep)

    # NEDD4L inhibits WNT. The control has WNT activity 1 for any allowed floor.
    w_max = p.wnt_min + (1.0 - p.wnt_min) * (1.0 + (1.0 / p.K_N) ** p.h_N)
    wnt = p.wnt_min + ((w_max - p.wnt_min)
                       / (1.0 + (max(T_ned, 0.0) / p.K_N) ** p.h_N))

    # DDB2/NER is not calibrated as repair of all ROS/APOBEC lesions.
    q_lesion = 1.0 + p.repair_proxy_weight * (Q - 1.0)
    growth = p.k_prol * wnt
    redox = y[IDX["R"]] / max(A, 1e-6)
    death = (p.d0 + p.k_apo * _hill(redox, p.K_Rd, p.n_Rd)
             + p.k_dam * _hill(y[IDX["L"]], p.K_Ld, p.n_Ld))
    return {"A": A, "Q": Q, "Q_lesion": q_lesion, "wnt": wnt,
            "division_rate": growth, "death_rate": death,
            "net_growth_rate": growth - death}


def rhs(t: float, y: np.ndarray, p: Params, iv: Intervention, E: float) -> np.ndarray:
    R = y[IDX["R"]]
    W, Fo, Ab = y[IDX["W"]], y[IDX["Fo"]], y[IDX["Ab"]]
    M_a3b, M_ned = y[IDX["M_a3b"]], y[IDX["M_ned"]]
    M_rep, M_aox = y[IDX["M_rep"]], y[IDX["M_aox"]]
    T_a3b, T_ned = y[IDX["T_a3b"]], y[IDX["T_ned"]]
    T_rep, T_aox = y[IDX["T_rep"]], y[IDX["T_aox"]]
    L = y[IDX["L"]]

    der = readouts(y, p, iv)
    A, Q = der["A"], der["Q_lesion"]

    dy = np.zeros(N_STATE)

    # --- KCC5: exposure -> oxidative stress, damped by antioxidant capacity
    dy[IDX["R"]] = p.k_E * E + p.r_basal - p.k_R * R - p.k_A * A * R

    # --- ROS -> writer / eraser effectors --------------------------------
    dR = max(R - p.R_ref, 0.0)
    W_inf = 1.0 + p.a_W * _hill(dR, p.K_W)
    F_inf = 1.0 + p.a_F * _hill(dR, p.K_F, p.n_F)
    B_inf = 1.0 - p.a_B * _hill(dR, p.K_B)          # ROS *inhibits* ALKBH5
    dy[IDX["W"]] = (W_inf - W) / p.tau_W
    dy[IDX["Fo"]] = (F_inf - Fo) / p.tau_F
    dy[IDX["Ab"]] = (B_inf - Ab) / p.tau_B

    # effective activities after intervention
    Feff = Fo * iv.fto
    Beff = Ab * iv.alkbh5

    # --- EMD1 / KCC4: site-specific m6A occupancy ------------------------
    # dM_j/dt = k_w,j W(R) (1 - M_j) - k_e,j F_j(R) M_j
    # The eraser is transcript-specific: FTO for A3B/NEDD4L, ALKBH5 for the
    # repair module (GSE144620), a blend for the antioxidant module.
    def occ(M, M0, eraser, w_scale=1.0):
        k_w = M0 / p.tau_M
        k_e = (1.0 - M0) / p.tau_M
        # Exposure-induced writer activity is site-specific. Direct METTL3
        # perturbation also affects sites with zero exposure responsiveness.
        w = iv.mettl3 * (1.0 + (W - 1.0) * w_scale)
        return k_w * w * (1.0 - M) - k_e * eraser * M

    aox_eraser = p.w_aox_fto * Feff + (1.0 - p.w_aox_fto) * Beff
    dy[IDX["M_a3b"]] = occ(M_a3b, p.M0_a3b, Feff)
    dy[IDX["M_ned"]] = occ(M_ned, p.M0_ned, Feff)
    rep_eraser = p.w_rep_fto * Feff + (1.0 - p.w_rep_fto) * Beff
    dy[IDX["M_rep"]] = occ(M_rep, p.M0_rep, rep_eraser, p.w_rep_writer)
    dy[IDX["M_aox"]] = occ(M_aox, p.M0_aox, aox_eraser)

    # --- reader-specific transcript fate ---------------------------------
    # APOBEC3B: m6A -> YTHDF2 -> decay.  FTO lowers M_a3b, stabilises A3B.
    dy[IDX["T_a3b"]] = (p.s_a3b * iv.a3b_expr
                        - p.d_a3b * (1 + p.gam_d2 * iv.ythdf2 * M_a3b) * T_a3b)
    # NEDD4L: m6A -> IGF2BP1-3 -> stabilisation.  Opposite sign, same FTO event.
    dy[IDX["T_ned"]] = (p.s_ned * iv.ned_expr
                        - p.d_ned / (1 + p.gam_i2 * iv.igf2bp * M_ned) * T_ned)
    dy[IDX["T_rep"]] = p.s_rep - p.d_rep * T_rep
    # m6A stabilises the writer-associated antioxidant pool.
    dy[IDX["T_aox"]] = (p.s_aox * (1.0 + p.nrf2 * _hill(dR, p.K_W))
                        - p.d_aox / (1 + p.gam_ia * iv.aox_reader * M_aox) * T_aox)

    # Fixation transfers repairable lesions (L) into the cumulative index (N).
    # This flux limits lesion accumulation when repair saturates.
    fix = p.k_fix * max(L, 0.0)
    dy[IDX["L"]] = (p.k_ROS * R + p.k_A3B * T_a3b
                    - p.k_rep * Q * L / (p.K_L + max(L, 0.0)) - fix)
    dy[IDX["N"]] = fix

    # Exponential population model over a finite illustrative horizon. Density
    # dependence/passaging must be specified for quantitative culture predictions.
    # R/A is an assumed death function in addition to A-dependent ROS clearance;
    # growth/death rates are not separately identified by relative expansion.
    dy[IDX["lp"]] = der["net_growth_rate"]

    return dy


def control_steady_state(p: Params, iv: Intervention = NO_IV) -> np.ndarray:
    """Burn the unexposed system in to its fixed point.

    The noncumulative control states start at this fixed point. Native values
    need not equal one; relative reporting divides by the appropriate control.
    Cumulative fixation and log population have no fixed points and are reset.
    """
    y0 = np.array([0.10, 1.0, 1.0, 1.0,
                   p.M0_a3b, p.M0_ned, p.M0_rep, p.M0_aox,
                   1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    sol = solve_ivp(rhs, (0.0, 600.0), y0, args=(p, iv, 0.0),
                    method="LSODA", rtol=1e-8, atol=1e-10, dense_output=False)
    if not sol.success:
        raise RuntimeError(f"control burn-in failed: {sol.message}")
    ss = sol.y[:, -1].copy()
    ss[IDX["N"]] = 0.0
    ss[IDX["lp"]] = 0.0
    return ss


def simulate(p: Params, iv: Intervention, E: float, t_eval: np.ndarray,
             y0: np.ndarray | None = None) -> np.ndarray:
    if y0 is None:
        y0 = control_steady_state(p)
    sol = solve_ivp(rhs, (float(t_eval[0]), float(t_eval[-1])), y0,
                    args=(p, iv, E), method="LSODA",
                    t_eval=t_eval, rtol=1e-7, atol=1e-9)
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    return sol.y


def simulate_protocol(p: Params, iv: Intervention, E: float,
                      t_eval: np.ndarray, intervention_start: float = 0.0,
                      y0: np.ndarray | None = None) -> np.ndarray:
    """Exposure starts at zero; switch intervention at an explicit model time.

    Integrate separate segments at the discontinuity, even when the switch is
    absent from the output grid. Positive starts represent delayed treatment;
    pretreatment requires an explicitly supplied initial state.
    """
    t = np.asarray(t_eval, dtype=float)
    if (t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all()
            or t[0] != 0 or np.any(np.diff(t) <= 0)):
        raise ValueError("protocol times must start at zero and strictly increase")
    if not np.isfinite(intervention_start) or intervention_start < 0:
        raise ValueError("intervention_start must be finite and non-negative")
    if intervention_start == 0:
        return simulate(p, iv, E, t, y0)
    if intervention_start >= t[-1]:
        return simulate(p, NO_IV, E, t, y0)
    before = t[t < intervention_start]
    first = simulate(p, NO_IV, E, np.append(before, intervention_start), y0)
    after = t[t > intervention_start]
    second = simulate(p, iv, E, np.insert(after, 0, intervention_start), first[:, -1])
    middle = second[:, :1] if np.any(t == intervention_start) else np.empty((N_STATE, 0))
    return np.concatenate((first[:, :-1], middle, second[:, 1:]), axis=1)


def trace(y: np.ndarray, p: Params, iv: Intervention) -> dict:
    """Native observables, including intervention-scaled effective eraser activities.

    ``Fo`` and ``Ab`` are the underlying states before intervention. The
    corresponding ``*_effective`` arrays are the activities used by the occupancy
    equations. Relative fold reporting is performed by the analysis/export stage.
    For delayed interventions the protocol layer applies the correct pre-switch
    readouts to each array.
    """
    n = y.shape[1]
    A = np.empty(n)
    Q = np.empty(n)
    wnt = np.empty(n)
    rates = {key: np.empty(n) for key in
             ("Q_lesion", "division_rate", "death_rate", "net_growth_rate")}
    for i in range(n):
        d = readouts(y[:, i], p, iv)
        A[i], Q[i], wnt[i] = d["A"], d["Q"], d["wnt"]
        for key in rates:
            rates[key][i] = d[key]
    return {
        "R": y[IDX["R"]], "W": y[IDX["W"]], "Fo": y[IDX["Fo"]], "Ab": y[IDX["Ab"]],
        "Fo_effective": y[IDX["Fo"]] * iv.fto,
        "Ab_effective": y[IDX["Ab"]] * iv.alkbh5,
        "M_a3b": y[IDX["M_a3b"]], "M_ned": y[IDX["M_ned"]],
        "M_rep": y[IDX["M_rep"]], "M_aox": y[IDX["M_aox"]],
        "M_mean4": (y[IDX["M_a3b"]] + y[IDX["M_ned"]]
                     + y[IDX["M_rep"]] + y[IDX["M_aox"]]) / 4.0,
        "A3B": y[IDX["T_a3b"]], "NEDD4L": y[IDX["T_ned"]],
        "A": A, "Q": Q, "WNT": wnt,
        "L": y[IDX["L"]], "N": y[IDX["N"]],
        "lnP": y[IDX["lp"]], "P": np.exp(np.clip(y[IDX["lp"]], -700, 700)),
        **rates,
    }
