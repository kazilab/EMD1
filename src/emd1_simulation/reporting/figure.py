"""Figures and captions for the EMD1 arsenic-exposure model."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

# Use an off-screen backend for file exports and retain the notebook backend.
try:
    get_ipython()  # noqa: F821
except NameError:
    mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ..simulation.conditions import PLOTTED
from ..processing.repair_evidence import REPAIR_DDB2, REPAIR_NULL_BY_CONTRAST

MM = 1 / 25.4
W_DOUBLE = 183 * MM

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
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

FIGURE_GROUPS = {
    "exposure_adaptation": {
        "label": "Figure 1",
        "number": "1",
        "title": "Simulated m⁶A regulation and antioxidant adaptation during arsenic exposure",
        "panels": ("a", "b", "h", "c"),
        "stem": "emd1_figure1_exposure_adaptation",
        "shape": (1, 4),
        "base_height": 2.25,
    },
    "reader_specificity": {
        "label": "Figure 2",
        "number": "2",
        "title": "Transcript-specific methylation and reader-dependent responses in the arsenic model",
        "panels": ("d", "e", "f", "i"),
        "stem": "emd1_figure2_reader_specificity",
        "shape": (1, 4),
        "base_height": 2.25,
    },
    "downstream_outputs": {
        "label": "Figure S3",
        "number": "S3",
        "title": "Fixation and population expansion in the RNA methylation model",
        "panels": ("j", "l"),
        "stem": "emd1_figureS3_downstream_outputs",
        "shape": (1, 2),
    },
    "conditional_repair": {
        "label": "Figure S4",
        "number": "S4",
        "title": "Modeled DDB2 regulation and published regional methylation measurements",
        "panels": ("g", "k"),
        "stem": "emd1_figureS4_conditional_repair",
        "shape": (1, 2),
    },
}
FIGURE_FORMATS = ("pdf", "svg", "png", "eps", "tif")
LEGACY_GROUP_STEMS = (
    "emd1_figure3_conditional_repair",
    "emd1_supplementary_figure1_downstream_outputs",
    "emd1_supplementary_figure2_conditional_repair",
)


def tint(hex_color: str, frac: float) -> str:
    """Blend toward white, returning a SOLID hex (EPS-safe; no alpha)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (round(c + (255 - c) * (1 - frac)) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _style(ax, title, ylabel, tag):
    ax.set_title(title, fontsize=TS["label"], color=INK, pad=3.5, loc="left")
    ax.set_ylabel(ylabel, fontsize=TS["fignote"], color=INK2, labelpad=2)
    ax.tick_params(labelsize=TS["fignote"], colors=INK2, pad=1.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.text(-0.20, 1.14, tag, transform=ax.transAxes, fontsize=TS["head"],
            fontweight="bold", color=INK, va="top", ha="left")


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


def _expand_ylim_to_data(ax) -> None:
    """Keep custom-run lines and shaded collections inside the visible range.

    ``dataLim`` includes the fill-between collections as well as line data;
    relim() would omit those collections. Expand the canonical limits only
    where needed, with multiplicative padding on logarithmic axes.
    """
    data_lo, data_hi = ax.dataLim.intervaly
    if not np.isfinite([data_lo, data_hi]).all():
        return
    current_lo, current_hi = ax.get_ylim()
    if ax.get_yscale() == "log":
        data_lo = data_lo if data_lo > 0 else ax.dataLim.minposy
        if not np.isfinite(data_lo) or data_lo <= 0 or data_hi <= 0:
            return
        span = max(np.log(data_hi / data_lo), np.log(current_hi / current_lo))
        factor = np.exp(max(0.04 * span, 0.02))
        lower = data_lo / factor if data_lo < current_lo else current_lo
        upper = data_hi * factor if data_hi > current_hi else current_hi
    else:
        span = max(data_hi - data_lo, current_hi - current_lo)
        pad = max(0.04 * span, 1e-9)
        lower = data_lo - pad if data_lo < current_lo else current_lo
        upper = data_hi + pad if data_hi > current_hi else current_hi
    ax.set_ylim(lower, upper)


def _draw_panel(ax, original, tag, runs, ens, t, cf):
    """Draw the specified model quantity and its reference curves."""
    ctl = {o: float(v[-1]) for o, v in runs["control"].items()}
    time_limits = (float(t[0]), float(t[-1]))

    def draw(obs, *, mode="fold", band_keys=("as",)):
        for key in band_keys:
            if key in ens:
                _band(ax, t, ens[key][obs], COND_COLOR[key])
        for condition in PLOTTED:
            trajectory = runs[condition.key]
            values = (_ratio(trajectory[obs], runs["control"][obs]) if mode == "fold"
                      else (trajectory["lnP"] - runs["control"]["lnP"]) / np.log(2.0))
            ax.plot(t, values, color=COND_COLOR[condition.key], lw=1.25,
                    zorder=3, solid_capstyle="round")
        ax.axhline(0.0 if mode == "log2" else 1.0, color=RULE, lw=0.6,
                   ls=(0, (2.5, 2.0)), zorder=0)

    if original == "b":
        for key, ls in (("Fo", "-"), ("W", (0, (1.1, 1.3))), ("Ab", (0, (3, 1.6)))):
            ax.plot(t, runs["as"][key] / ctl[key], color=CAT[0], lw=1.25, ls=ls, zorder=3)
        ax.axhline(1.0, color=RULE, lw=0.6, ls=(0, (2.5, 2.0)), zorder=0)
        top = max(float(np.max(runs[c.key][k] / ctl[k]))
                  for c in PLOTTED for k in ("Fo", "W", "Ab"))
        ax.set_ylim(0, top * 1.10)
        _style(ax, "Writer / eraser: arsenic arm", "fold vs control", tag)
    elif original == "i":
        for condition in PLOTTED:
            ax.plot(t, runs[condition.key]["A3B"] / ctl["A3B"],
                    color=COND_COLOR[condition.key], lw=1.25, zorder=3)
            ax.plot(t, runs[condition.key]["NEDD4L"] / ctl["NEDD4L"],
                    color=COND_COLOR[condition.key], lw=1.25,
                    ls=(0, (3, 1.6)), zorder=3)
        ax.axhline(1.0, color=RULE, lw=0.6, ls=(0, (2.5, 2.0)), zorder=0)
        _style(ax, "Reader-specific outcome", "fold vs control", tag)
        ax.set_ylim(0.08, 3.35)
    elif original in ("e", "f", "g", "h"):
        obs, title = {
            "e": ("M_a3b", "APOBEC3B site"),
            "f": ("M_ned", "NEDD4L site"),
            "g": ("M_rep", "DNA-repair site (DDB2)"),
            "h": ("M_aox", "Antioxidant module"),
        }[original]
        if original == "g":
            lo, hi = REPAIR_NULL_BY_CONTRAST["arsenic"]
            ax.fill_between(t, lo, hi, color=tint(INK2, 0.11), linewidth=0, zorder=0.5)
        _band(ax, t, ens["as"][obs], CAT[0])
        if original in ("g", "h") and "as_m3kd" in ens:
            _band(ax, t, ens["as_m3kd"][obs], CAT[2])
        for condition in PLOTTED:
            ax.plot(t, runs[condition.key][obs] / ctl[obs],
                    color=COND_COLOR[condition.key], lw=1.25, zorder=3)
        ax.axhline(1.0, color=RULE, lw=0.6, ls=(0, (2.5, 2.0)), zorder=0)
        _style(ax, title, "m$^6$A, fold vs control", tag)
        ax.set_ylim(0.15, 2.25)
        if original == "g":
            for level, colour in ((REPAIR_DDB2["arsenic"][0], CAT[0]),
                                  (REPAIR_DDB2["writer_kd"][0], INK)):
                ax.plot(time_limits, [level, level], color=colour, lw=0.8,
                        ls=(0, (1.2, 1.4)), zorder=4)
            if cf is not None:
                if "M_rep_lo" in cf:
                    ax.fill_between(t, cf["M_rep_lo"], cf["M_rep_hi"],
                                    color=tint(INK2, 0.22), linewidth=0, zorder=2)
                ax.plot(t, cf["M_rep"], color=INK2, lw=1.0,
                        ls=(0, (3, 1.6)), zorder=4)
    else:
        obs, title, mode = {
            "a": ("R", "Oxidative stress", "fold"),
            "c": ("A", "Antioxidant capacity", "fold"),
            "d": ("M_mean4", "Mean m$^6$A (four sites)", "fold"),
            "j": ("N", "Mutation-fixation index", "fold"),
            "k": ("Q", "DDB2/NER proxy", "fold"),
            "l": ("lnP", "Population expansion", "log2"),
        }[original]
        draw(obs, mode=mode, band_keys=("as", "as_m3kd") if original in ("c", "l") else ("as",))
        _style(ax, title, "log$_2$ fold vs control" if mode == "log2" else "fold vs control", tag)
        if original == "c":
            ax.set_ylim(0.55, 2.05)
        elif original == "d":
            ax.set_ylim(0.35, 1.95)
        elif original == "k":
            ax.set_ylim(0.5, 2.0)
            if "as_m3kd" in ens:
                _band(ax, t, ens["as_m3kd"][obs], COND_COLOR["as_m3kd"])
            if cf is not None:
                if "Q_lo" in cf:
                    ax.fill_between(t, cf["Q_lo"], cf["Q_hi"],
                                    color=tint(INK2, 0.13), linewidth=0, zorder=1)
                ax.plot(t, cf["Q"], color=INK2, lw=1.0,
                        ls=(0, (3, 1.6)), zorder=4)
        elif original == "j":
            ax.set_yscale("log")
            hi = max(float(np.max(_ratio(runs[c.key]["N"], runs["control"]["N"])))
                     for c in PLOTTED)
            ax.set_ylim(0.85, hi * 1.35)
            ticks = [x for x in (1, 2, 5, 10, 20, 50) if x <= hi * 1.35]
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(x) for x in ticks])


def _condition_key(ax, *, loc, anchor):
    """Compact condition key shared by all publication graphics."""
    labels = {"control": "Control", "as": "Arsenic",
              "as_ftoi": "As + FTOi", "as_m3kd": "As + METTL3 KD"}
    handles = [Line2D([], [], color=COND_COLOR[c.key], lw=1.25,
                      label=labels[c.key]) for c in PLOTTED]
    legend = ax.legend(handles=handles, loc=loc, bbox_to_anchor=anchor,
                       frameon=False, fontsize=5.8, labelcolor=INK2,
                       handlelength=1.6, handletextpad=0.4, labelspacing=0.1,
                       borderpad=0.15, borderaxespad=0)
    legend.set_gid("condition-key")
    return legend


def _group_panel_keys(fig, group, include_counterfactual):
    """Integrate condition, molecular and source keys within their panels."""
    if group == "exposure_adaptation":
        _exposure_panel_keys(fig)
    elif group == "reader_specificity":
        _condition_key(fig.axes[0], loc="upper right", anchor=(0.98, 0.98))
        handles = [Line2D([], [], color=INK2, lw=1.25, ls=style, label=label)
                   for style, label in (("-", "APOBEC3B"),
                                        ((0, (3, 1.6)), "NEDD4L"))]
        legend = fig.axes[3].legend(
            handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.99),
            frameon=False, fontsize=6.3, labelcolor=INK2,
            handlelength=1.8, handletextpad=0.4, labelspacing=0.2,
            borderpad=0.15, borderaxespad=0)
        legend.set_gid("reader-key")
    elif group == "downstream_outputs":
        _condition_key(fig.axes[0], loc="upper left", anchor=(0.03, 0.98))
    elif group == "conditional_repair":
        _condition_key(fig.axes[1], loc="upper right", anchor=(0.98, 0.98))
        handles = [
            Line2D([], [], color=CAT[0], lw=0.8, ls=(0, (1.2, 1.4)),
                   label="Measured arsenic"),
            Line2D([], [], color=INK, lw=0.8, ls=(0, (1.2, 1.4)),
                   label="Measured METTL14 KD"),
        ]
        if include_counterfactual:
            handles.append(Line2D([], [], color=INK2, lw=1.0, ls=(0, (3, 1.6)),
                                  label="Writer-input alternative"))
        legend = fig.axes[0].legend(
            handles=handles, loc="upper left", bbox_to_anchor=(0.03, 0.98),
            frameon=False, fontsize=5.8, labelcolor=INK2,
            handlelength=1.8, handletextpad=0.4, labelspacing=0.1,
            borderpad=0.15, borderaxespad=0)
        legend.set_gid("source-key")


def _exposure_panel_keys(fig):
    """Label the exposure conditions and molecular trajectories."""
    _condition_key(fig.axes[0], loc="lower left", anchor=(0.20, 0.065))
    for line, label in zip(fig.axes[1].lines[:3], ("FTO", "METTL3/14", "ALKBH5")):
        fig.axes[1].annotate(
            label, xy=(line.get_xdata()[-1], line.get_ydata()[-1]),
            xytext=(-3, 3), textcoords="offset points", ha="right", va="bottom",
            fontsize=TS["fignote"], color=line.get_color(), clip_on=True)


def create_figure(runs, ens, t, cf=None,
                  parameter_overrides: dict[str, float] | None = None, *,
                  show_condition_legend: bool = False, group: str | None = None):
    """Return a Matplotlib figure, optionally restricted to a group in FIGURE_GROUPS."""
    if group is not None and group not in FIGURE_GROUPS:
        raise ValueError(f"Unknown figure group {group!r}; choose from {tuple(FIGURE_GROUPS)}")
    specification = FIGURE_GROUPS[group] if group is not None else None
    panels = specification["panels"] if specification else tuple("abcdefghijkl")
    nrows, ncols = specification["shape"] if specification else (3, 4)
    base_height = (specification.get("base_height", 4.85 if nrows == 2 else 2.75)
                   if specification else 5.70)
    internal_keys = specification is not None
    style_height = 0.0
    key_height = 0.40 if show_condition_legend and not internal_keys else 0.0
    height = base_height + style_height + key_height
    t = np.asarray(t, dtype=float)
    fig = plt.figure(figsize=(W_DOUBLE, height))
    gs = fig.add_gridspec(
        nrows, ncols, left=0.085 if specification else 0.075, right=0.985,
        top=(base_height - 0.32) / height,
        bottom=(0.50 if specification else 0.60) / height,
        hspace=0.62, wspace=0.38 if specification else 0.40)
    for index, original in enumerate(panels):
        ax = fig.add_subplot(gs[index // ncols, index % ncols])
        tag = chr(ord("a") + index) if specification else original
        _draw_panel(ax, original, tag, runs, ens, t, cf)
        if specification:
            ax.set_title("", loc="left")
        ax.set_gid(f"panel-{tag}")
        if parameter_overrides:
            _expand_ylim_to_data(ax)
        ax.set_xlim(float(t[0]), float(t[-1]))
        ax.set_xticks(np.linspace(float(t[0]), float(t[-1]), 4))
        ax.set_xlabel("nominal model days", fontsize=TS["fignote"], color=INK2, labelpad=1.5)
    if internal_keys:
        _group_panel_keys(fig, group, cf is not None)
    if show_condition_legend and not internal_keys:
        handles = [Line2D([], [], color=COND_COLOR[condition.key], lw=1.5,
                          label=condition.label) for condition in PLOTTED]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 1 - 0.10 / height),
                   ncols=4, frameon=False, fontsize=TS["fignote"], labelcolor=INK2,
                   handlelength=1.9, handletextpad=0.5, columnspacing=1.3,
                   borderaxespad=0)
    return fig


def create_figures(runs, ens, t, cf=None,
                   parameter_overrides: dict[str, float] | None = None, *,
                   show_condition_legend: bool = False):
    """Return two main and two supplementary figures from completed arrays."""
    return {key: create_figure(
        runs, ens, t, cf=cf, parameter_overrides=parameter_overrides,
        show_condition_legend=show_condition_legend, group=key)
        for key in FIGURE_GROUPS}


def build_figure(runs, ens, t, outdir: Path, cf=None, n_ensemble: int | None = None,
                 parameter_overrides: dict[str, float] | None = None) -> list[Path]:
    """Export the graphic and its caption."""
    t = np.asarray(t, dtype=float)
    time_limits = (float(t[0]), float(t[-1]))
    n_draws = int(n_ensemble) if n_ensemble is not None else _ensemble_n(ens)
    fig = create_figure(runs, ens, t, cf=cf, parameter_overrides=parameter_overrides)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "svg", "png", "eps", "tif"):
        path = outdir / f"emd1_simulation.{ext}"
        options = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
        fig.savefig(path, **options)
        paths.append(path)
    paths.append(write_legend(outdir, n_draws,
                              t_start_days=time_limits[0], t_end_days=time_limits[1],
                              parameter_overrides=parameter_overrides, include_counterfactual=cf is not None))
    plt.close(fig)
    return paths


PANEL_CAPTIONS = {
    "a": "Reactive oxygen species (ROS), expressed as fold change from control.",
    "b": "FTO (solid), METTL3/14 (dotted) and ALKBH5 (dashed) in the arsenic arm. Eraser curves show model states before intervention multipliers.",
    "c": "Aggregate transcript abundance in the writer-associated antioxidant branch, used as a proxy for antioxidant activity.",
    "d": "Mean m⁶A occupancy gives equal weight to APOBEC3B, NEDD4L, DDB2 and the antioxidant module.",
    "e": "Modeled m⁶A occupancy at the APOBEC3B site, with FTO as the eraser.",
    "f": "Illustrative NEDD4L-site m⁶A occupancy. A quantitative source comparison requires a verified genomic interval.",
    "g": "Modeled m⁶A occupancy at the DDB2 repair site.",
    "h": "Modeled m⁶A occupancy in the writer-associated antioxidant branch, with contributions from FTO and ALKBH5. PRDX5/YTHDF2 is a separate mechanism outside the model.",
    "i": "APOBEC3B abundance (solid) reflects YTHDF2-associated decay. NEDD4L abundance (dashed) reflects IGF2BP-associated stabilization. Reader gains are assumed.",
    "j": "Cumulative fixation index relative to control, on a logarithmic axis. Its arbitrary scale has no established conversion to mutations per genome.",
    "k": "DDB2 translation as a proxy for nucleotide excision repair (NER). Its default contribution to aggregate lesion removal is zero.",
    "l": "Population expansion relative to control (log₂ fold change). A population growing more slowly than control can still increase from its starting size.",
}


def _write_caption(path: Path, title: str, panels: tuple[str, ...], n_draws: int,
                   t_start_days: float, t_end_days: float,
                   parameter_overrides: dict[str, float] | None,
                   include_counterfactual: bool) -> Path:
    labels = {panel: chr(ord("a") + index) for index, panel in enumerate(panels)}
    text = (f"# {title}\n\n"
            f"Illustrative model trajectories over {t_start_days:g}–{t_end_days:g} nominal days. "
            "Grey denotes control, blue arsenic (As), orange arsenic with FTO inhibition (FTOi), "
            "and green arsenic with METTL3 knockdown (KD).\n\n")
    for panel, label in labels.items():
        text += f"**{label}** {PANEL_CAPTIONS[panel]}\n\n"
    blue = ", ".join(labels[p] for p in panels if p not in ("b", "i"))
    green = ", ".join(labels[p] for p in panels if p in ("c", "g", "h", "k", "l"))
    text += (f"Blue bands ({blue}) show the 5th–95th percentiles for arsenic "
             f"across {n_draws} independent parameter draws, each normalized to its own control.")
    if green:
        text += f" Green bands ({green}) give the same range for arsenic with METTL3 knockdown."
    text += " These bands describe sensitivity to assumed parameter values.\n"

    if "g" in panels:
        ars = REPAIR_DDB2["arsenic"][0]
        kd = REPAIR_DDB2["writer_kd"][0]
        lo, hi = REPAIR_NULL_BY_CONTRAST["arsenic"]
        kd_lo, kd_hi = REPAIR_NULL_BY_CONTRAST["writer_kd"]
        text += (f"\nIn panel {labels['g']}, horizontal dotted lines show DDB2 enrichment "
                 f"over the first three annotated exons after arsenic exposure (blue, {ars:.2f}×) "
                 f"or METTL14 knockdown (navy, {kd:.2f}×). Both contrasts are normalized to "
                 f"their respective background-transcript medians. The pale grey band spans the "
                 f"arsenic background 5th–95th percentiles ({lo:.3f}–{hi:.3f}). The corresponding "
                 f"METTL14 range is {kd_lo:.3f}–{kd_hi:.3f}. These cross-sectional contrasts are "
                 "distinct from the modeled trajectories and the simulated METTL3 knockdown. "
                 "[GSE145923, Cui et al.](https://doi.org/10.1038/s41467-021-22469-6) and "
                 "[GSE145924, Yang et al.](https://doi.org/10.1073/pnas.2025948118).\n")
        if include_counterfactual:
            text += (f"\nIn panels {labels['g']} and {labels['k']}, the charcoal dashed curves "
                     "show the effect of setting `w_rep_writer = 1` at the configured values "
                     "of the other parameters. Grey bands show its sensitivity range.\n")
        if not parameter_overrides:
            text += ("\nThe default repair model assumes zero arsenic-induced writer input "
                     "at this site. The control, arsenic and arsenic with FTO inhibition "
                     "trajectories coincide at 1.\n")
    if parameter_overrides:
        text += "\n| Parameter | Configured value |\n|---|---:|\n"
        for name, value in sorted(parameter_overrides.items()):
            text += f"| `{name}` | {value:.12g} |\n"
    path.write_text(text, encoding="utf-8")
    return path


def write_legend(outdir: Path, n_draws: int, *,
                 t_start_days: float = 0.0, t_end_days: float = 45.0,
                 parameter_overrides: dict[str, float] | None = None,
                 include_counterfactual: bool = True) -> Path:
    """Write a caption for the full set of model trajectories."""
    return _write_caption(outdir / "emd1_simulation_legend.md", "EMD1 mechanistic simulation",
                          tuple("abcdefghijkl"), n_draws, t_start_days, t_end_days,
                          parameter_overrides, include_counterfactual)


def write_group_legend(outdir: Path, group: str, n_draws: int, *,
                       t_start_days: float = 0.0, t_end_days: float = 45.0,
                       parameter_overrides: dict[str, float] | None = None,
                       include_counterfactual: bool = True) -> Path:
    """Write the caption for a manuscript figure."""
    spec = FIGURE_GROUPS[group]
    return _write_caption(outdir / f"{spec['stem']}_legend.md", f"{spec['label']}. {spec['title']}",
                          spec["panels"], n_draws, t_start_days, t_end_days,
                          parameter_overrides, include_counterfactual)


def remove_figure_exports(outdir: Path, *, legacy_only: bool = False) -> None:
    """Remove only recognized generated graphics, captions and previews."""
    stems = ["emd1_simulation", *LEGACY_GROUP_STEMS]
    if not legacy_only:
        stems += [spec["stem"] for spec in FIGURE_GROUPS.values()]
    for stem in stems:
        for suffix in FIGURE_FORMATS:
            (outdir / f"{stem}.{suffix}").unlink(missing_ok=True)
        (outdir / f"{stem}_legend.md").unlink(missing_ok=True)
        (outdir / f"{stem}_with_key.png").unlink(missing_ok=True)
    (outdir / "emd1_figure_panel_map.csv").unlink(missing_ok=True)


def build_figures(runs, ens, t, outdir: Path, cf=None, n_ensemble: int | None = None,
                  parameter_overrides: dict[str, float] | None = None) -> list[Path]:
    """Export manuscript figures and captions."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t = np.asarray(t, dtype=float)
    n_draws = int(n_ensemble) if n_ensemble is not None else _ensemble_n(ens)
    figures = create_figures(runs, ens, t, cf=cf, parameter_overrides=parameter_overrides)
    paths = []
    try:
        for group, figure in figures.items():
            spec = FIGURE_GROUPS[group]
            for extension in FIGURE_FORMATS:
                path = outdir / f"{spec['stem']}.{extension}"
                options = {"pil_kwargs": {"compression": "tiff_lzw"}} if extension == "tif" else {}
                figure.savefig(path, **options)
                paths.append(path)
            paths.append(write_group_legend(
                outdir, group, n_draws, t_start_days=float(t[0]), t_end_days=float(t[-1]),
                parameter_overrides=parameter_overrides, include_counterfactual=cf is not None))
    finally:
        for figure in figures.values():
            plt.close(figure)
    for spec in FIGURE_GROUPS.values():
        (outdir / f"{spec['stem']}_with_key.png").unlink(missing_ok=True)
    remove_figure_exports(outdir, legacy_only=True)
    return paths
