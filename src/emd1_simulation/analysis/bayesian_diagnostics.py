"""Format chain diagnostics without depending on JAX or NumPyro."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def format_diagnostics(
    statistics: Mapping[str, Mapping],
    names: Sequence[str] | None = None,
    divergences: int | None = None,
) -> str:
    """Report worst R-hat and minimum ESS with explicit site coverage.

    The default includes every supplied site, including hierarchical scales,
    latent effects and panel offsets. Explicit ``names`` are prefix filters;
    their output is labelled as a selected subset, not global convergence.
    Every selected diagnostic must be present, finite and positive. Undefined
    values are unavailable; they are never replaced by a plausible value.
    """
    if names is not None and (not names or any(not isinstance(n, str) or not n for n in names)):
        raise ValueError("names must be None or a nonempty sequence of site-name prefixes")
    selected = (dict(statistics) if names is None else
                {name: values for name, values in statistics.items()
                 if any(name.startswith(prefix) for prefix in names)})
    missing = ([] if names is None else [prefix for prefix in names
               if not any(name.startswith(prefix) for name in selected)])

    def aggregate(field: str, reduction, precision: str) -> str:
        if not selected:
            return "unavailable (no selected sites)"
        if missing:
            return "unavailable (missing selected sites)"
        arrays = []
        for values in selected.values():
            if field not in values:
                return "unavailable (missing diagnostic)"
            array = np.asarray(values[field], dtype=float)
            if not array.size or not np.isfinite(array).all() or np.any(array <= 0):
                return "unavailable (undefined diagnostic)"
            arrays.append(array.ravel())
        value = float(reduction(np.concatenate(arrays)))
        return format(value, precision)

    r_hat = aggregate("r_hat", np.max, ".4f")
    ess = aggregate("n_eff", np.min, ".0f")
    scope = "all supplied sites" if names is None else "selected subset"
    sites = ", ".join(sorted(selected)) or "none"
    result = (f"coverage = {scope} ({len(selected)}/{len(statistics)}; {sites})   "
              f"max r_hat = {r_hat}   min ESS = {ess}")
    if missing:
        result += "   missing requested prefixes = " + ", ".join(missing)
    if divergences is not None:
        result += f"   divergences = {divergences}"
    return result
