from IPython.core.magics.execution import TimeitResult
import matplotlib.pyplot as plt
import numpy as np


def plot_timings(
        timings,
        ax=None,
        id_to_x=None,
        which="best",
        sort_ids=True,
        set_ax_props=True,
        ax_props=None,
        plot_props=None,
        fill_between_props=None):
    """Plot timings

    Args:
        timinings: Timings mapping (maps run ID strings to TimeitResults)

    Keyword args:
        ax: Matplotlib axes to draw on. Will be created if `None`
        id_to_x: Optional function converting run ID strings to x-values.
        which: Which timings to show (one of "average", "best", "worst")
        sort_ids: If True, sorts timings based on run IDs.
        set_ax_props: If True, set axes properties via :meth:`ax.set`
        ax_props: Keyword args passed on to :meth:`ax.set`
        plot_props: Keyword args passed on to :meth:`ax.plot`
        plot_props: Keyword args passed on to :meth:`ax.fill_between`
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if id_to_x is None:
        id_to_x = lambda x: x

    x, t = zip(
        *(
            (id_to_x(run_id), getattr(timing, which))
            for run_id, timing in timings.items()
            if isinstance(timing, TimeitResult)
        )
    )

    if which == "average":
        std = [
            getattr(timing, "stdev")
            for timing in timings.values()
            if isinstance(timing, TimeitResult)
            ]
        std = np.asarray(std)

    x = np.asarray(x)
    t = np.asarray(t)

    if sort_ids:
        sorti = np.argsort(x)
        x = x[sorti]
        t = t[sorti]

        if which == "average":
            std = std[sorti]

    default_plot_props = {
        }

    if plot_props is not None:
        default_plot_props.update(plot_props)

    default_fill_between_props = {
        "alpha": 0.5,
        "edgecolor": 'none',
        }

    if fill_between_props is not None:
        default_fill_between_props.update(fill_between_props)

    if which == "average":
        ax.fill_between(
            x,
            t - std,
            t + std,
            **default_fill_between_props
        )

    line = ax.plot(x, t, **default_plot_props)

    if set_ax_props:
        default_ax_props = {
            "xlim": (x[0], x[-1]),
            "xlabel": "run ID",
            "ylabel": ("time / s")
            }

        if ax_props is not None:
            default_ax_props.update(ax_props)

        ax.set(**default_ax_props)

    return line
