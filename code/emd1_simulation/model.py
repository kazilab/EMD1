"""EMD1 mechanistic simulation: KCC5 -> EMD1/KCC4 -> KCC2, KCC3, KCC10.

Public-data-constrained mechanistic proof-of-concept for Figure 2B. This is a
*hypothesis-testing* model expressed in relative (normalised) effect sizes, not
a quantitatively validated carcinogenicity-prediction model: no parameter here
should be read as a physiological rate constant.

Design decision that drives everything else
-------------------------------------------
EMD1 is NOT modelled as one global m6A variable. m6A has no single carcinogenic
direction -- under oxidative stress it is protective on some transcripts and
transforming on others -- so the EMD1 state is a vector of site/transcript
specific occupancies

    M(t) = {M_A3B, M_NEDD4L, M_repair, M_antioxidant}

each coupled to its own *reader*, with the sign of the coupling set by which
reader binds it:

    APOBEC3B   m6A --YTHDF2-->    decay          (m6A up  => A3B down)
    NEDD4L     m6A --IGF2BP1-3--> stabilisation  (m6A up  => NEDD4L up)
    repair     m6A --YTHDF1-->    translation    (m6A up  => repair up)
    antioxid.  m6A --(reader?)-->  stabilisation  (m6A up  => antioxidant up)

Reader-specific response functions can produce opposite outcomes even with a
shared methylation driver. Multiple sites represent differing site responses;
opposite transcript signs alone do not establish that they are necessary.

Time is in nominal model days, not calibrated physiological days. The default
protocol applies exposure and intervention together from t = 0.
"""

from __future__ import annotations

from dataclasses import dataclass, replace, fields

import numpy as np
from scipy.integrate import solve_ivp

# --- state vector layout -------------------------------------------------
# Kept as a flat float array (not a dict) so the ensemble can run thousands of
# integrations without per-step allocation.
IDX = {
    "R":     0,   # oxidative stress / ROS                       -> KCC5
    "W":     1,   # METTL3/METTL14 writer activity
    "Fo":    2,   # FTO eraser activity        (chronic, slow)
    "Ab":    3,   # ALKBH5 eraser activity     (ERK/JNK, fast)
    "M_a3b": 4,   # m6A occupancy, APOBEC3B                      -.
    "M_ned": 5,   # m6A occupancy, NEDD4L                          |-> EMD1 / KCC4
    "M_rep": 6,   # m6A occupancy, DNA-repair module               |
    "M_aox": 7,   # m6A occupancy, antioxidant module            -'
    "T_a3b": 8,   # APOBEC3B abundance
    "T_ned": 9,   # NEDD4L abundance
    "T_rep": 10,  # repair-module transcript abundance
    "T_aox": 11,  # antioxidant-module transcript abundance
    "L":     12,  # unrepaired lesion load (gammaH2AX analogue); feeds N
    "N":     13,  # cumulative fixed mutational burden           -> KCC2
    "lp":    14,  # log relative clonal expansion                -> KCC10
}
N_STATE = len(IDX)


@dataclass(frozen=True)
class Params:
    """Relative effect sizes. Provenance lives in the inline comments and in
    the source comments and archive data/THIRD_PARTY.md; methods and results
    are described in the accompanying manuscript."""

    # --- exposure -> ROS (KCC5) -----------------------------------------
    r_basal: float = 0.90      # basal ROS production
    k_E: float = 3.00          # arsenic -> ROS gain
    k_R: float = 1.00          # constitutive ROS clearance
    k_A: float = 2.00          # antioxidant-dependent ROS clearance (the feedback)

    # --- ROS -> writer / eraser effectors --------------------------------
    R_ref: float = 0.30        # derived control ROS (= r_basal/(k_R+k_A));
                               # effectors respond to (R - R_ref)+. Recomputed
                               # per ensemble draw, not an independent parameter.
    a_W: float = 1.41          # METTL3/14 induction amplitude; with ALKBH5
                               # flat this alone must carry the measured
                               # antioxidant-site rise (~1.28x)
    K_W: float = 0.25
    tau_W: float = 1.50
    a_F: float = 3.50          # FTO accumulation amplitude (chronic arsenic).
                               # Raised alongside a_W: with the writer stronger,
                               # FTO must work harder to hold M_ned at the
                               # measured 0.61 on the prespecified published
                               # exon-1 interval. Protein-level (autophagy), not mRNA.
    K_F: float = 0.20
    n_F: float = 2.00
    tau_F: float = 8.00        # slow: FTO accumulation is a chronic-exposure
                               # phenotype. Impaired autophagic degradation
                               # stabilises FTO protein; this is accumulation,
                               # not transcriptional induction.
    a_B: float = 0.00          # ALKBH5 *inhibition* amplitude. Zero for the
                               # arsenic exemplar: ALKBH5 inhibition is an acute
                               # ERK/JNK-driven peroxide response (NAR), and under
                               # chronic arsenic ALKBH5 *expression* rises
                               # (1.19-2.26x) rather than its activity falling.
                               # Leaving it on made M_rep climb 1.76x, which two
                               # platforms contradict. Set ~0.6 to model acute ROS.
    K_B: float = 0.18
    tau_B: float = 0.50        # fast: post-translational (ERK/JNK) modification

    # --- site-specific m6A occupancy -------------------------------------
    tau_M: float = 0.50        # m6A write/erase turnover timescale
    M0_a3b: float = 0.35       # control occupancies
    M0_ned: float = 0.40
    M0_rep: float = 0.30
    M0_aox: float = 0.30
    w_aox_fto: float = 0.15    # antioxidant eraser: mostly ALKBH5-like; the FTO
                               # share is the minority PRDX5/YTHDF2 branch
    w_rep_writer: float = 0.00 # how much of the ARSENIC-driven writer induction
                               # the DNA-repair site sees. Assumed zero: first-three-exon DDB2
                               # sits at background percentile 38 under chronic
                               # arsenic, with XPC moving identically, so
                               # no site-specific exposure effect is detectable.
                               # This is the whole of the arsenic-null claim --
                               # it does not delete the edge, which stays live
                               # through gam_f1r and responds to direct writer
                               # intervention.
    w_rep_fto: float = 0.00    # assumed FTO share of repair-site erasure;
                               # explored separately as a structural scenario
    repair_proxy_weight: float = 0.00  # uncalibrated DDB2 -> aggregate lesion repair
                               # transfer. Default excludes that extrapolation.

    # --- reader-specific coupling gains ----------------------------------
    gam_d2: float = 10.00       # YTHDF2  -> decay,          A3B
    gam_i2: float = 5.00       # IGF2BP1-3 -> stabilisation, NEDD4L
    gam_f1r: float = 2.00      # YTHDF1 translation coupling, DDB2 repair proxy.
                               # NON-zero: the published METTL14 -> DDB2 ->
                               # YTHDF1 -> global-repair perturbation/rescue
                               # experiments support the edge (PMID 34452996).
                               # The annotation-fixed reanalysis of its deposited
                               # bigWigs does not independently reproduce the
                               # DDB2 decrease, so existence comes from the
                               # published functional evidence and magnitude is
                               # assumed. What arsenic fails to do is drive the
                               # SITE -- that claim belongs on w_rep_writer.
                               # Setting this
                               # to zero deleted the coupling globally, so even
                               # ALKBH5 knockdown left Q pinned at 1.000 while
                               # M_rep moved 1.96x.
    gam_ia: float = 15.0       # m6A -> stabilisation,      antioxidant module
                               # fixed by the measured siMETTL3 decay ratio (2.2x)

    # --- transcript turnover ---------------------------------------------
    s_a3b: float = 4.500       # d_a3b * (1 + gam_d2 * M0_a3b); control abundance 1
    d_a3b: float = 1.00
    s_ned: float = 1.0 / 3.0   # d_ned / (1 + gam_i2 * M0_ned); control abundance 1
    d_ned: float = 1.00
    s_rep: float = 1.00
    d_rep: float = 1.00
    s_aox: float = 1.0 / 5.5   # d_aox / (1 + gam_ia * M0_aox); control abundance 1
    d_aox: float = 1.00
    nrf2: float = 0.270        # direct, m6A-INDEPENDENT ROS induction of antioxidants
                               # set so total induction matches the measured 1.44x

    # --- KCC2 / KCC3: lesion load and fixed mutations ---------------------
    # Refitted against TWO digitised JBC mutation ratios only: 1G 2 -> 8 (4.0x)
    # and 3Q 9 -> 4 (0.44x of As). The A3B-knockdown ratio (1G 8 -> 3) was
    # deliberately EXCLUDED so that arm stays held out and V5 remains a genuine
    # prediction; fitting to it, as an earlier pass did, quietly turned V5 into
    # a restatement of the fit. The model now predicts 0.24x there against a
    # measured 0.375x -- correct sign, magnitude soft.
    k_ROS: float = 1.0672985348329618  # direct oxidative lesion rate
    k_A3B: float = 1.3956264414264556  # APOBEC3B-attributable lesion rate
    k_rep: float = 2.7159806534354018  # repair flux (saturable, scaled by Q_lesion)
    K_L: float = 0.5768266984210220
    k_fix: float = 0.2136534839667953  # persistent lesions -> fixed mutations.
                               # With the sink restored it enters the N ratios;
                               # nevertheless these five parameters remain
                               # jointly nonidentified by only two targets.

    # --- KCC10: WNT / proliferation-survival ------------------------------
    K_N: float = 0.80          # NEDD4L -| WNT/beta-catenin
    h_N: float = 2.50
    wnt_min: float = 0.00      # floor on WNT activity. Zero for the arsenic
                               # exemplar, but carried explicitly so the
                               # published equation is the bounded general form
                               # W = W_min + (W_max - W_min)/(1 + (T/K_N)^n)
                               # rather than a special case of it.
    k_prol: float = 0.0373
    d0: float = 0.0061         # basal turnover
    k_apo: float = 0.0853      # redox-imbalance death (the dominant channel)
    K_Rd: float = 1.4036       # redox-balance (R/A) tolerance threshold
    n_Rd: float = 5.2018       # threshold, not a gradient
    k_dam: float = 0.0019      # small in the nominal expansion calibration;
                               # not evidence identifying a biological death cause
    K_Ld: float = 2.20
    n_Ld: float = 3.00

    def __post_init__(self):
        for name in ("w_rep_writer", "w_rep_fto", "repair_proxy_weight"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a fraction in [0, 1]")


def normalise_params(p: Params) -> Params:
    """Derive baseline synthesis and resting ROS from independent parameters.

    Use this constructor for parameter exploration. Keeping this in the model
    gives the nominal run, sensitivity analysis, and ensemble the same
    parameterization. Params defaults match these derived values.
    """
    return replace(
        p, R_ref=p.r_basal / (p.k_R + p.k_A),
        s_a3b=p.d_a3b * (1.0 + p.gam_d2 * p.M0_a3b),
        s_ned=p.d_ned / (1.0 + p.gam_i2 * p.M0_ned),
        s_rep=p.d_rep,
        s_aox=p.d_aox / (1.0 + p.gam_ia * p.M0_aox),
    )


@dataclass(frozen=True)
class Intervention:
    """Multiplicative perturbations; 1.0 = untouched.

    These are the handles the published arsenic experiments actually pull
    (FTO knockdown / FB23-2 inhibition, METTL3 or ALKBH5 knockdown, YTHDF1/2
    knockdown, APOBEC3B knockdown, NEDD4L rescue), so predicted interventions
    can be compared against reported ones without refitting.
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

    Each is normalised by its control-occupancy value so that an unperturbed
    cell sits at 1.0 and every trajectory reads as a fold change.
    """
    M_rep, M_aox = y[IDX["M_rep"]], y[IDX["M_aox"]]
    T_rep, T_aox, T_ned = y[IDX["T_rep"]], y[IDX["T_aox"]], y[IDX["T_ned"]]

    # Antioxidant capacity is transcript abundance: the m6A dependence sits in
    # the decay term (see dT_aox/dt), because the actinomycin chase in the
    # arsenic redox work shows METTL3 knockdown roughly doubling the decay
    # constant. It is NOT a translation gain, which is how this was first built.
    A = T_aox
    # DDB2/NER proxy. The coupling is live (gam_f1r > 0): the
    # published METTL14 -> DDB2 -> YTHDF1 -> global genome repair experiments
    # support a live edge, so the model responds to a writer perturbation at
    # this site. The deposited-track reanalysis does not calibrate its size. The
    # arsenic-specific null lives upstream, in w_rep_writer = 0, so this proxy is flat
    # under arsenic by assumption while still moving under writer or eraser intervention.
    Q = T_rep * (1 + p.gam_f1r * iv.ythdf1 * M_rep) / (1 + p.gam_f1r * p.M0_rep)

    # NEDD4L is a negative regulator of WNT/beta-catenin, so the sign flips.
    # W_max is set by the normalisation WNT(T_ned = 1) = 1, so the curve is
    # pinned to the control fixed point for any floor.
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
        # Two separate things, and they must not be folded together:
        #   w_scale   how much of the EXPOSURE-driven writer induction this site
        #             sees. Zero at the repair site in the nominal scenario (assumed).
        #   iv.mettl3 a direct writer INTERVENTION, which every site feels
        #             whatever its exposure responsiveness.
        # Scaling the whole deviation by w_scale (the previous form) silently
        # made METTL3 knockdown a no-op wherever w_scale = 0: M_rep stayed at
        # 0.300000 even for a 99% knockdown. That is algebraically wrong for a
        # direct writer intervention and incompatible with retaining the
        # published writer-responsive repair mechanism.
        # For sites with w_scale = 1 this is algebraically identical to before.
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
    # Antioxidant module: m6A stabilises, mirroring NEDD4L rather than acting
    # through translation. Measured: siMETTL3 -> ~2.2x faster decay.
    dy[IDX["T_aox"]] = (p.s_aox * (1.0 + p.nrf2 * _hill(dR, p.K_W))
                        - p.d_aox / (1 + p.gam_ia * iv.aox_reader * M_aox) * T_aox)

    # --- KCC2 / KCC3: lesion load, then fixed mutations -------------------
    # Two variables, not one. The sketch's single D with zeroth-order removal
    # (-k_repair*Q) can run negative, and it conflates a repairable lesion load
    # with an irreversible mutational burden -- which matters here, because
    # arsenic-transformed cells survive *while* accumulating mutations. L is the
    # gammaH2AX-like repairable load (feeds N; the k_dam death term on L is
    # fitted near zero). N is the monotone mutation endpoint and drives nothing.
    #
    # Fixation is a SINK on L as well as the source of N: a lesion that has been
    # fixed as a mutation is no longer a repairable lesion, and counting it in
    # both pools double-counts it. It also matters numerically -- repair
    # saturates at k_rep*Q_lesion, so with no linear sink L grows without bound in any
    # arm whose lesion production exceeds that ceiling (the METTL3-knockdown arm
    # does: 3.4 vs 3.0 per day, giving L = 26 at day 45 and 177 at one year).
    # The sink bounds L for any production rate.
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

    Every condition starts here, and every reported trajectory is normalised to
    it, so the control arm is flat at 1.0 by construction and no parameter has
    to be hand-solved to place the baseline.
    """
    y0 = np.array([0.10, 1.0, 1.0, 1.0,
                   p.M0_a3b, p.M0_ned, p.M0_rep, p.M0_aox,
                   1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    sol = solve_ivp(rhs, (0.0, 600.0), y0, args=(p, iv, 0.0),
                    method="LSODA", rtol=1e-8, atol=1e-10, dense_output=False)
    if not sol.success:
        raise RuntimeError(f"control burn-in failed: {sol.message}")
    ss = sol.y[:, -1].copy()
    # N and lp are cumulative counters, not states with fixed points
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
    """Full observable set for a solved trajectory, as fold change vs control."""
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
