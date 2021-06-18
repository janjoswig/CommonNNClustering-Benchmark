import json
from operator import itemgetter

from IPython.core.magics.execution import TimeitResult
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

from cnnclustering import cluster


class TimeitResultEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, TimeitResult):
            return {
                "_type": "TimeitResult",
                "loops": obj.loops,
                "repeat": obj.repeat,
                "best": obj.best,
                "worst": obj.worst,
                "all_runs": obj.all_runs,
                "compile_time": obj.compile_time,
                "precision": obj._precision
                }
        return super().default(self, obj)


class TimitResultDecoder(json.JSONDecoder):
    """Use :func:`as_TimeitResult` object hook on json load instead"""
    def default(self, obj):
        decoded_obj = super().default(self, obj)
        if isinstance(decoded_obj, list) and obj[0] == "TimeitResult":
            return TimeitResult(*obj[1].values())
        return decoded_obj


def as_TimeitResult(obj):
    try:
        _type = obj.pop('_type')
    except KeyError:
        _type = None

    if _type is None:
        return obj

    if _type == 'TimeitResult':
        return TimeitResult(*obj.values())

    return obj


def combine_timeit_results(*results):
    """Merge TimeitResults

    Args:
        *results: TimeitResults instances

    Returns:
        TimeitResult

    Note:
        Be careful about merging results from different sources.
        Merging e.g. takes the minimum number of loops done in
        each timing.
    """

    loops = min(r.loops for r in results)
    repeat = sum(r.repeat for r in results)
    all_runs = []
    for r in results:
        all_runs.extend(r.all_runs)
    best = min(all_runs)
    worst = max(all_runs)
    precision = max(r._precision for r in results)
    compile_time = max(r.compile_time for r in results)

    return TimeitResult(
        loops, repeat, best, worst, all_runs, compile_time, precision
        )


def save_report(timings, report_file, overwrite=False):
    report_file.parent.mkdir(exist_ok=True, parents=True)

    if report_file.is_file() and not overwrite:
        print("Report already exists. Set `overwrite = True` to force overwrite.")
        return

    with open(report_file, "w") as f:
        json.dump(timings, f, indent=4, cls=TimeitResultEncoder)


def load_report(report_file):

    with open(report_file) as f:
        timings = json.load(f, object_hook=as_TimeitResult)

    return timings


def add_previous_reports(timings, *previous_timings):

    for pt in previous_timings:
        for run_id, timing in pt.items():
            if run_id in timings:
                combined = combine_timeit_results(
                    timings[run_id], timing
                )
            else:
                combined = timing

            timings[run_id] = combined


def get_ratios(timings, base=None, which="best"):
    """Get relative performance of runs based on timings

    Args:
        timinings: Timings mapping (maps run ID strings to TimeitResults)

    Keyword args:
        base: Use the timing for this run ID as the baseline (factor 1)
        which: Which timing should be shown ("average", "best", or "worst")
    """

    if base is not None:
        base = getattr(timings[base], which)
    else:
        base = min(
            getattr(x, which)
            for x in timings.values()
            if isinstance(x, TimeitResult)
        )

    sorted_ratios = sorted(
        [
            (k, getattr(v, which) / base)
            for k, v in timings.items()
            if isinstance(v, TimeitResult)
        ],
        key=itemgetter(1)
    )

    return sorted_ratios


def print_ratios(ratios):
    """Pretty print timing ratios

    Args:
        ratios: Ratios obtained with :func:`get_ratios` as
            list of run ID/factor tuples.
    """

    print(f"{'Run ID':>10}: Factor")
    print("=======================")
    for run_id, factor in ratios:
        print(f"{run_id:>10}: {factor:.2f}")


class RecordEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, cluster.Record):
            return {
                "_type": "Record",
                "n_points": obj.n_points,
                "radius_cutoff": obj.radius_cutoff,
                "cnn_cutoff": obj.cnn_cutoff,
                "member_cutoff": obj.member_cutoff,
                "max_clusters": obj.max_clusters,
                "n_clusters": obj.n_clusters,
                "ratio_largest": obj.ratio_largest,
                "ratio_noise": obj.ratio_noise,
                "execution_time": obj.execution_time,
                }
        return super().default(self, obj)


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
        "linestyle": "--",
        "marker": "o",
        "markersize": 5,
        "markeredgecolor": "k"
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


def as_Record(obj):
    try:
        _type = obj.pop('_type')
    except KeyError:
        _type = None

    if _type is None:
        return obj

    if _type == 'Record':
        return cluster.Record(*obj.values())

    return obj


def compute_neighbours(data, radius, sort=False):

    tree = KDTree(data)
    neighbourhoods = tree.query_radius(
        data, r=radius, return_distance=False
        )

    if sort:
        for n in neighbourhoods:
            n.sort()

    return neighbourhoods


def gen_no_structure_points(size, random_state=2021):
    rng = np.random.default_rng(random_state)
    no_structure = rng.random(size)

    return StandardScaler().fit_transform(no_structure)


def gen_blobs_points(size, random_state=8, **kwargs):
    blobs, _ = datasets.make_blobs(
        n_samples=size[0],
        n_features=size[1],
        random_state=random_state,
        **kwargs
        )
    return StandardScaler().fit_transform(blobs)


def gen_circles_points(size, random_state=8, **kwargs):
    if size[1] != 2:
        raise RuntimeError("Can only generate circles in 2D.")

    circles, _ = datasets.make_circles(
        n_samples=size[0],
        random_state=random_state,
        **kwargs
        )
    return StandardScaler().fit_transform(circles)


def gen_moons_points(size, random_state=8, **kwargs):
    if size[1] != 2:
        raise RuntimeError("Can only generate circles in 2D.")

    circles, _ = datasets.make_moons(
        n_samples=size[0],
        random_state=random_state,
        **kwargs
        )
    return StandardScaler().fit_transform(circles)