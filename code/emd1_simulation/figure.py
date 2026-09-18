"""Figure 2B: the EMD1 cascade, simulated.

Editable twin: ``figures/regenerate_figures.ipynb`` § Figure 2B (drawing cell).
Promote notebook layout edits back here so ``python -m emd1_simulation.run``
stays in sync. Follows the conventions already established in that notebook:
exact 183 mm double-column canvas, one shared type scale with a 6 pt floor,
Type 42 embedded fonts, colour assigned by role, solid tints rather than alpha
(EPS has no real transparency), vector + 600 dpi raster export.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

# Force Agg only for non-notebook CLI runs. The interactive macOS backend snaps
# figure width to whole screen pixels (silently costing ~0.34 pt and breaking
# the "PDF measures exactly 183 mm" guarantee). Inside IPython/Jupyter leave
# the kernel backend alone so regenerate_figures.ipynb can display inline.
try:
    get_ipython()  # noqa: F821
except NameError:
    mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .conditions import (PLOTTED, REPAIR_DDB2, REPAIR_NULL_BY_CONTRAST,
                         REPAIR_N_QUANTIFIED)

MM = 1 / 25.4
W_DOUBLE = 183 * MM

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
    # Route mathtext through the same stack; the default fontset would embed
    # DejaVu alongside Arial just for the superscripts.
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
    "mathtext.default": "regular",
    "figure.dpi": 160, "savefig.dpi": 600,
    "savefig.bbox": None, "savefig.pad_inches": 0,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.transparent": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
})

INK, INK2, MUTED, RULE = "#12253a", "#3d4a58", "#6b7785", "#c9d1da"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
TS = {"fignote": 6.6, "body": 7.2, "label": 7.6, "head": 8.6, "title": 9.4}

COND_COLOR = {"control": MUTED, "as": CAT[0], "as_ftoi": CAT[1], "as_m3kd": CAT[2]}


def tint(hex_color: str, frac: float) -> str:
    """Blend toward white, returning a SOLID hex (EPS-safe; no alpha)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (round(c + (255 - c) * (1 - frac)) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _style(ax, title, ylabel, tag):
    ax.set_title(title, fontsize=TS["label"], color=INK, pad=3.5, loc="left")
    ax.set_ylabel(ylabel, fontsize=TS["fignote"], color=INK2, labelpad=2)
    ax.set_xlim(0, 45)
    ax.set_xticks([0, 15, 30, 45])
    ax.tick_params(labelsize=TS["fignote"], colors=INK2, pad=1.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.text(-0.20, 1.14, tag, transform=ax.transAxes, fontsize=TS["head"],
            fontweight="bold", color=INK, va="top", ha="left")


def _ord(n: float) -> str:
    """1 -> 1st, 42 -> 42nd. Plain %dth reads as '3th' and '42th'."""
    i = int(round(n))
    suf = "th" if 11 <= i % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(i % 10, "th")
    return f"{i}{suf}"


def _ratio(num, den):
    """num/den elementwise; where den == 0 (t = 0 for cumulative counters) the
    limit is the ratio of rates, which is 1 because every arm starts from the
    same control fixed point."""
    out = np.ones_like(num, dtype=float)
    np.divide(num, den, out=out, where=np.abs(den) > 1e-12)
    return out


def _band(ax, t, arr, color):
    """5th-95th percentile of the ensemble, as a solid tint behind the lines."""
    lo, hi = np.percentile(arr, [5, 95], axis=0)
    ax.fill_between(t, lo, hi, color=tint(color, 0.16), linewidth=0, zorder=1)


def _ensemble_n(ens) -> int:
    """Number of successful draws actually present in the ensemble arrays."""
    if not ens:
        return 0
    for cond in ens.values():
        for arr in cond.values():
            return int(np.asarray(arr).shape[0])
    return 0


def build_figure(runs, ens, t, outdir: Path, cf=None, n_ensemble: int | None = None) -> list[Path]:
    fig = plt.figure(figsize=(W_DOUBLE, 7.15))
    gs = fig.add_gridspec(
        3, 4, left=0.062, right=0.988, top=0.765, bottom=0.095,
        hspace=0.62, wspace=0.34)

    ctl = {o: float(v[-1]) for o, v in runs["control"].items()}
    n_draws = int(n_ensemble) if n_ensemble is not None else _ensemble_n(ens)

    def draw(ax, obs, *, mode="fold", ref=None, band_keys=("as",)):
        """mode: fold (vs control), abs (raw), log2 (relative clonal expansion)."""
        if mode == "fold":
            # Elementwise against the control TRAJECTORY, not its endpoint: N is
            # a cumulative counter, so dividing by a scalar would drag its curve
            # down to zero at t=0 instead of the correct ratio-of-rates limit.
            series = lambda tr: _ratio(tr[obs], runs["control"][obs])
            band = lambda a: a
        elif mode == "abs":
            series = lambda tr: tr[obs]
            band = None
        else:
            series = lambda tr: (tr["lnP"] - runs["control"]["lnP"]) / np.log(2.0)
            band = lambda a: a
        if band is not None:
            for key in band_keys:
                if key in ens:
                    _band(ax, t, ens[key][obs], COND_COLOR[key])
        for c in PLOTTED:
            ax.plot(t, series(runs[c.key]), color=COND_COLOR[c.key],
                    lw=1.25, zorder=3, solid_capstyle="round")
        if ref is not None:
            ax.axhline(ref, color=RULE, lw=0.6, ls=(0, (2.5, 2.0)), zorder=0)

    # ---- row 1: upstream (KCC5) and the four-site mean (not transcriptome-wide m6A)
    ax = fig.add_subplot(gs[0, 0])
    draw(ax, "R", ref=1.0)
    _style(ax, "Oxidative stress", "fold vs control", "a")
    ax.text(0.97, 0.06, "KCC5", transform=ax.transAxes, fontsize=TS["fignote"],
            color=CAT[0], ha="right", fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    for key, lab, ls, ypos in (("Fo", "FTO", "-", 0.90),
                               ("W", "METTL3/14", (0, (1.1, 1.3)), 0.52),
                               ("Ab", "ALKBH5", (0, (3, 1.6)), 0.05)):
        ax.plot(t, runs["as"][key] / ctl[key], color=CAT[0], lw=1.25, ls=ls, zorder=3)
        ax.text(0.96, ypos, lab, transform=ax.transAxes, fontsize=TS["fignote"],
                color=INK2, ha="right", va="center")
    ax.axhline(1.0, color=RULE, lw=0.6, ls=(0, (2.5, 2.0)), zorder=0)
    # Must clear the FTO plateau (~4.2x); the old 3.7 cap ran that trace off
    # the top of the panel.
    top = max(float(np.max(runs[c.key][k] / ctl[k]))
              for c in PLOTTED for k in ("Fo", "W", "Ab"))
    ax.set_ylim(0, top * 1.10)
    _style(ax, "Writer / eraser response", "fold vs control", "b")

    ax = fig.add_subplot(gs[0, 2])
    draw(ax, "A", ref=1.0, band_keys=("as", "as_m3kd"))
    _style(ax, "Antioxidant capacity", "fold vs control", "c")
    ax.set_ylim(0.55, 2.05)
    ax.text(0.96, 0.93, "m$^6$A-dependent\nadaptation damps ROS", transform=ax.transAxes,
            fontsize=TS["fignote"], color=MUTED, va="top", ha="right", linespacing=1.3)

    ax = fig.add_subplot(gs[0, 3])
    draw(ax, "M_mean4", ref=1.0)
    _style(ax, "Mean m$^6$A (four sites)", "fold vs control", "d")
    ax.set_ylim(0.35, 1.95)
    ax.text(0.96, 0.93,
            "leans down, but averages\naway the opposing site\nsigns in (e) and (h)",
            transform=ax.transAxes, fontsize=TS["fignote"], color=MUTED,
            va="top", ha="right", linespacing=1.3)

    # ---- row 2: EMD1 / KCC4 as a vector of site-specific states
    # All four sites use fold-vs-own-control. The ensemble stores normalized
    # effects, not raw occupancies; multiplying those effects by the nominal M0
    # used to collapse the e/f band to a false point at t=0 while labelling it
    # absolute occupancy. Fold space is both honest and directly comparable
    # across sites whose basal stoichiometries are not measured.
    specs = [("M_a3b", "APOBEC3B site", "e", "eraser: FTO"),
             ("M_ned", "NEDD4L site", "f", "eraser: FTO"),
             ("M_rep", "DNA-repair site (DDB2)", "g", ""),
             ("M_aox", "Antioxidant module", "h", "eraser: mostly ALKBH5")]
    for col, (obs, title, tag, note) in enumerate(specs):
        ax = fig.add_subplot(gs[1, col])
        if obs == "M_rep":
            # This is the ARSENIC contrast's cross-transcript specificity band.
            # Draw it first and below every model sensitivity band.
            lo, hi = REPAIR_NULL_BY_CONTRAST["arsenic"]
            ax.fill_between(t, lo, hi, color=tint(INK2, 0.11),
                            linewidth=0, zorder=0.5)
        _band(ax, t, ens["as"][obs], CAT[0])
        # The arsenic M_rep band has zero width by construction; show uncertainty
        # where that model actually moves, in the METTL3-knockdown arm.
        if obs == "M_rep" and "as_m3kd" in ens:
            _band(ax, t, ens["as_m3kd"][obs], CAT[2])
        if obs == "M_aox" and "as_m3kd" in ens:
            _band(ax, t, ens["as_m3kd"][obs], CAT[2])
        for c in PLOTTED:
            ax.plot(t, runs[c.key][obs] / ctl[obs], color=COND_COLOR[c.key],
                    lw=1.25, zorder=3)
        ax.axhline(1.0, color=RULE, lw=0.6, ls=(0, (2.5, 2.0)), zorder=0)
        _style(ax, title, "m$^6$A, fold vs control", tag)
        ax.set_ylim(0.15, 2.25)
        if note:
            ax.text(0.03, 0.94, note, transform=ax.transAxes,
                    fontsize=TS["fignote"], color=MUTED, va="top")
        if obs == "M_rep":
            # Everything here is in fold-vs-control, so the measured DDB2
            # ratios plot natively -- no assumed occupancy in the way. They are
            # cross-sectional endpoint contrasts, not trajectories, so they are
            # drawn as reference LEVELS rather than at some invented timepoint.
            # The grey band behind the model bands is the ARSENIC contrast's
            # null. The measured writer-knockdown point is labelled with its
            # own contrast-specific null instead of being judged against grey.
            lo, hi = REPAIR_NULL_BY_CONTRAST["arsenic"]
            ars, _ = REPAIR_DDB2["arsenic"]
            kd, _ = REPAIR_DDB2["writer_kd"]
            for lvl, col in ((ars, CAT[0]), (kd, INK)):
                ax.plot([0, 45], [lvl, lvl], color=col, lw=0.8,
                        ls=(0, (1.2, 1.4)), zorder=4)
            if cf is not None:
                # Same observable as the empirical DDB2 comparison above. This
                # is the one-parameter w_rep_writer counterfactual; its Q
                # consequence is shown separately in (k).
                if "M_rep_lo" in cf:
                    ax.fill_between(t, cf["M_rep_lo"], cf["M_rep_hi"],
                                    color=tint(INK2, 0.22), linewidth=0,
                                    zorder=2)
                ax.plot(t, cf["M_rep"], color=INK2, lw=1.0,
                        ls=(0, (3, 1.6)), zorder=4)
            # Keep this unusually information-dense panel readable by placing
            # labels beside their endpoint levels instead of stacking prose in
            # the upper half. The two measured levels are dotted; the model
            # counterfactual is dashed. Neither measured contrast is an outlier
            # in the annotation-fixed first-three-exon reanalysis.
            label_box = dict(facecolor="white", edgecolor="none", pad=0.35)
            ax.text(0.05, 0.975,
                    f"grey: As specificity null {lo:.2f}-{hi:.2f}",
                    transform=ax.transAxes, fontsize=TS["fignote"],
                    color=MUTED, ha="left", va="top", bbox=label_box)
            ax.text(1.5, ars - 0.035, f"DDB2 As {ars:.2f}",
                    fontsize=TS["fignote"], color=CAT[0], ha="left", va="top",
                    bbox=label_box)
            ax.text(1.5, kd + 0.035, f"DDB2 METTL14 KD {kd:.2f}",
                    fontsize=TS["fignote"], color=INK, ha="left", va="bottom",
                    bbox=label_box)
            if cf is not None:
                ax.text(43.5, float(cf["M_rep"][-1]) + 0.075,
                        "one-parameter As-site CF", fontsize=TS["fignote"],
                        color=INK2, ha="right", va="bottom", bbox=label_box)
            ax.text(43.5, 0.50, "simulated METTL3 KD",
                    fontsize=TS["fignote"], color=CAT[2], ha="right", va="bottom",
                    bbox=label_box)
            # Three arms sit exactly on 1 and the last drawn hides the rest.
            ax.text(43.5, 1.025, "ctrl / As / As+FTOi",
                    fontsize=TS["fignote"], color=MUTED, ha="right", va="bottom",
                    bbox=label_box)

    # ---- row 3: reader-specific consequences and the three KCC outputs
    ax = fig.add_subplot(gs[2, 0])
    for c in PLOTTED:
        ax.plot(t, runs[c.key]["A3B"] / ctl["A3B"], color=COND_COLOR[c.key], lw=1.25, zorder=3)
        ax.plot(t, runs[c.key]["NEDD4L"] / ctl["NEDD4L"], color=COND_COLOR[c.key],
                lw=1.25, ls=(0, (3, 1.6)), zorder=3)
    ax.axhline(1.0, color=RULE, lw=0.6, ls=(0, (2.5, 2.0)), zorder=0)
    _style(ax, "Reader-specific outcome", "fold vs control", "i")
    ax.set_ylim(0.08, 3.35)
    ax.text(0.04, 0.95, "APOBEC3B (YTHDF2)", transform=ax.transAxes, fontsize=TS["fignote"],
            color=INK2, ha="left", va="top")
    ax.text(0.96, 0.02, "NEDD4L (IGF2BP), dashed", transform=ax.transAxes,
            fontsize=TS["fignote"], color=INK2, ha="right", va="bottom")

    for col, (obs, title, ylab, tag, kcc, mode, tagpos) in enumerate([
            ("N", "Mutational burden", "fold vs control", "j", "KCC2", "fold", (0.04, 0.93, "left", "top")),
            ("Q", "DDB2/NER proxy", "fold vs control", "k", "KCC3 proxy \u2014 assumed\nzero input", "fold", (0.04, 0.96, "left", "top")),
            ("lnP", "Clonal expansion", "log$_2$ fold vs control", "l", "KCC10", "log2", (0.04, 0.06, "left", "bottom"))], start=1):
        ax = fig.add_subplot(gs[2, col])
        draw(ax, obs, mode=mode, ref=0.0 if mode == "log2" else 1.0,
             band_keys=("as", "as_m3kd") if obs == "lnP" else ("as",))
        _style(ax, title, ylab, tag)
        if obs == "Q":
            ax.set_ylim(0.5, 2.0)
            for k in ("as_m3kd",):
                if k in ens:
                    _band(ax, t, ens[k][obs], COND_COLOR[k])
            ax.text(0.05, 0.36, "ctrl / As / As+FTOi coincide",
                    transform=ax.transAxes, fontsize=TS["fignote"],
                    color=MUTED, ha="left", va="bottom")
            if cf is not None:
                # ONE parameter changed: the repair site responds to the arsenic
                # writer induction like every other site. Drawn with its own
                # 5-95% band, because the nominal trace alone overstates how
                # cleanly this downstream implication is separated. Its band
                # is NOT compared with the empirical DDB2 null; that like-for-
                # like M_rep comparison is made in (g).
                if "Q_lo" in cf:
                    ax.fill_between(t, cf["Q_lo"], cf["Q_hi"],
                                    color=tint(INK2, 0.13), linewidth=0, zorder=1)
                ax.plot(t, cf["Q"], color=INK2, lw=1.0, ls=(0, (3, 1.6)), zorder=4)
                ax.text(0.95, 0.60, "downstream Q from\nthe same counterfactual",
                        transform=ax.transAxes, fontsize=TS["fignote"],
                        color=INK2, ha="right", va="bottom", linespacing=1.3)
            # No annotation for the METTL3 arm: the legend already names it and
            # the drop is the panel's only moving line.
        if obs == "N":
            # Log scale: the METTL3 arm runs far above the arsenic effect and a
            # linear axis renders the latter invisible. Range from the data --
            # the old fixed 16 cap ran the METTL3 trace off the top.
            ax.set_yscale("log")
            hi = max(float(np.max(_ratio(runs[c.key]["N"], runs["control"]["N"])))
                     for c in PLOTTED)
            ax.set_ylim(0.85, hi * 1.35)
            ticks = [x for x in (1, 2, 5, 10, 20, 50) if x <= hi * 1.35]
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(x) for x in ticks])
        tx, ty, tha, tva = tagpos
        ax.text(tx, ty, kcc, transform=ax.transAxes, fontsize=TS["fignote"],
                color=CAT[0], ha=tha, va=tva, fontweight="bold")

    for ax in fig.axes:
        ax.set_xlabel("nominal model days", fontsize=TS["fignote"], color=INK2, labelpad=1.5)

    # ---- header ---------------------------------------------------------
    fig.text(0.062, 0.982,
             "Mechanistic simulation of the EMD1 cascade: "
             "KCC5 $\\rightarrow$ EMD1 (KCC4) $\\rightarrow$ KCC2, KCC3, KCC10",
             fontsize=TS["title"], color=INK, fontweight="bold", va="top")
    fig.text(0.062, 0.956,
             "Arsenic exemplar. EMD1 is a vector of site-specific m$^6$A states (e-h), not one global variable; reader-specific consequences follow in (i-l).\n"
             "All site panels (e-h) are fold vs each draw's own control. The published METTL14-DDB2-YTHDF1 coupling is retained, but its magnitude is assumed.\n"
             f"Annotation-fixed first-three-exon reanalysis (g): DDB2 As {REPAIR_DDB2['arsenic'][0]:.2f}x "
             f"[{REPAIR_NULL_BY_CONTRAST['arsenic'][0]:.3f}-{REPAIR_NULL_BY_CONTRAST['arsenic'][1]:.3f}]; "
             f"METTL14 KD {REPAIR_DDB2['writer_kd'][0]:.2f}x "
             f"[{REPAIR_NULL_BY_CONTRAST['writer_kd'][0]:.3f}-{REPAIR_NULL_BY_CONTRAST['writer_kd'][1]:.3f}]. Neither is an outlier.\n"
             "The simulated METTL3-KD response and one-parameter arsenic-site counterfactual are structural model implications, not matched validations.\n"
             "The one-parameter counterfactual (grey dashed) is compared with the DDB2 arsenic null as repair-site m$^6$A in (g).\n"
             "Panel (k) shows the DDB2/NER proxy Q; no default transfer to aggregate lesion repair. The repair-site band overlaps the null; a moderate alternative is not excluded.\n"
             f"Bands: heuristic 5th-95th ranges over {n_draws} independent draws (not CI/CrI); green writer-KD bands expose its uncertain expansion suppression.\n"
             "Illustrative kinetics on a nominal time axis. The antioxidant FTO-decay conflict was a pooling artefact: no detectable eraser effect at the nearest stage (see audit).",
             fontsize=TS["fignote"], color=INK2, va="top", linespacing=1.2)

    handles = [Line2D([], [], color=COND_COLOR[c.key], lw=1.4, label=c.label) for c in PLOTTED]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.058, 0.825),
               frameon=False, fontsize=TS["body"], labelcolor=INK2,
               handlelength=1.8, handletextpad=0.6, borderaxespad=0, ncols=4,
               columnspacing=2.0)

    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "svg", "png", "eps"):
        q = outdir / f"figure2b_emd1_simulation.{ext}"
        fig.savefig(q)
        paths.append(q)
    q = outdir / "figure2b_emd1_simulation.tif"
    fig.savefig(q, pil_kwargs={"compression": "tiff_lzw"})
    paths.append(q)
    plt.close(fig)
    return paths
