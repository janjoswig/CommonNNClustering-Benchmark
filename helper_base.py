import json
import pathlib
import timeit

import numpy as np
from scipy.optimize import curve_fit
from sklearn import datasets
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

from cnnclustering import cluster


def indent_at_parens(s):
    """Take a string and introduce indention at parentheses"""

    o = ""
    level = 1
    saw_comma = False
    for c in s:
        if saw_comma:
            if c == " ":
                o += f"\n{'    ' * (level - 1)}"
            else:
                o += f"\n{'    ' * (level - 1)}{c}"
            saw_comma = False
            continue

        if c == "(":
            o += f"(\n{'    ' * level}"
            level += 1
            continue

        if c == ")":
            level -= 1
            o += f"\n{'    ' * level})"
            continue

        if c == ",":
            saw_comma = True
            o += ","
            continue

        o += c

    return o


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


def save_records(clusterings, record_file, scan_ids=None, overwrite=False):
    record_file = pathlib.Path(record_file)
    record_file.parent.mkdir(exist_ok=True, parents=True)

    if not isinstance(clusterings, list):
        clusterings = [clusterings]

    if scan_ids is None:
        scan_ids = record_file.stem

    if not isinstance(scan_ids, list):
        scan_ids = [scan_ids]

    assert len(scan_ids) == len(clusterings)

    if record_file.is_file() and not overwrite:
        raise RuntimeError("File exists: str(record_file)")

    with open(record_file, "w") as fp:
        json.dump(
            {
                i: c.summary._list
                for i, c in zip(scan_ids, clusterings)
            },
            fp, cls=RecordEncoder, indent=4
            )


def load_records(record_file):
    with open(record_file, "r") as fp:
        record_dict = json.load(fp, object_hook=as_Record)
    return {k: cluster.Summary(v) for k, v in record_dict.items()}


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


def growth(n, a, b):
    return a * n**b


def growth_with_c(n, a, b, c):
    return a * n**b + c


def scale(x, y, newx, f=growth, **kwargs):

    try:
        popt, pcov = curve_fit(f, x, y, **kwargs)
        perr = np.sqrt(np.diag(pcov))
    except RuntimeError as error:
        print(error)
    else:
        return growth(newx, *popt), (popt, perr)


def collect_timings(
        gen_func, setup_func, run_arguments_list,
        transform_func=None, timings=None, repeats=10):
    """Orchestrate timings

    Args:
        gen_func: A function, returning data. Called with
            run arguments "gen".
        setup_func: A function, accepting data and returning a
            function which should be timed. Called with
            run arguments "setup".
        run_argumens_list: A list of run arguments.

    Keyword args:
        transform_func: A function, transforming generated data before setup.
        timings: An optional timings mapping which results should
            put into.
        repeats: Repeats the timing *n* times. Using timeit -n/-r directly would
            not ensure running the setup before each timing.

    Returns:
        timings mapping
    """

    # Timed function has to be in global namespace to be discovered by %timeit magic
    global timed_args
    global timed_kwargs
    global timed_func

    if timings is None:
        timings = {}

    for run_index, arguments in enumerate(run_arguments_list):

        gen_args, gen_kwargs = arguments.get("gen", ((), {}))
        data = gen_func(*gen_args, **gen_kwargs)

        if transform_func is not None:
            trans_args, trans_kwargs = arguments.get("transform", ((), {}))
            data = transform_func(data, *trans_args, **trans_kwargs)

        timeit_results = []
        for _ in range(repeats):
            setup_args, setup_kwargs = arguments.get("setup", ((), {}))
            timed_func = setup_func(data, *setup_args, **setup_kwargs)

            timed_args, timed_kwargs = arguments.get("timed", ((), {}))
            # o = %timeit -n 1 -r 1 -q -o timed_func(*timed_args, **timed_kwargs)
            o = timeit.timeit(
                "timed_func(*timed_args, **timed_kwargs)",
                number=1,
                globals=globals()
                )
            timeit_results.append(o)

        run_id = arguments.get("id", str(run_index))
        timings[run_id] = timeit_results

    return timings


class Run:

    def __init__(
            self,
            run_name, case_name,
            function_map, run_argument_list_gen,
            **kwargs):
        self.run_name = run_name
        self.case_name = case_name
        self.function_map = function_map
        self.run_argument_list = run_argument_list_gen(**kwargs)
