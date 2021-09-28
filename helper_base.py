from datetime import datetime
import gc
import json
import pathlib
import timeit

import numpy as np
from scipy.optimize import curve_fit
from sklearn import datasets
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

from cnnclustering import cluster

import helper_timeit


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
    """Transform function"""

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
        return f(newx, *popt), (popt, perr)


def collect_timings(
        gen_func, setup_func, run_arguments_list,
        transform_func=None, timings=None, repeats=10):
    """Orchestrate timings

    Note:
        This has been superseeded by :func:`time_unit`.

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


def time_unit(bm_unit, repeats=10):
    """Perform a single timing

    Args:
        bm_unit: A single instance of :obj:`BMUnit`

    Keyword args:
        repeat: How many times the timing should be repeated
    """

    if bm_unit.gen_func is not None:
        data = bm_unit.gen_func(*bm_unit.gen_args, **bm_unit.gen_kwargs)
    else:
        data = None

    if bm_unit.transform_func is not None:
        data = bm_unit.transform_func(
            data,
            *bm_unit.transform_args, **bm_unit.transform_kwargs
            )

    assert bm_unit.setup_func is not None

    timeit_results = []
    for _ in range(repeats):
        timed_func = bm_unit.setup_func(
            data,
            *bm_unit.setup_args, **bm_unit.setup_kwargs,
            )

        o = timeit.timeit(
            "t(*args, **kwargs)",
            number=1,
            globals={
                "t": timed_func,
                "args": bm_unit.timed_args,
                "kwargs": bm_unit.timed_kwargs
                }
            )

        timeit_results.append(o)

        gc.collect()

    return timeit_results


class Run:

    def __init__(
            self,
            run_name,
            bm_units=None):
        self.run_name = run_name
        self.bm_units = bm_units
        self.reset()

    def reset(self):
        self._timings = {}

    @property
    def timings(self):
        return {
            k: helper_timeit.raw_to_timeitresult(v)
            if not isinstance(v, str) else v
            for k, v in self._timings.items()
            }

    def scaling(self, id_to_n=None, mask=None, **kwargs):

        if id_to_n is None:
            id_to_n = lambda x: int(x)

        if mask is None:
            mask = set()

        x, y = zip(
            *(
                (id_to_n(k), v.best)
                for k, v in self.timings.items()
                if (isinstance(v, helper_timeit.TimeitResult)) & (k not in mask)
                )
            )
        x = np.asarray(x)
        y = np.asarray(y)
        sorti = np.argsort(x)
        x = x[sorti]
        y = y[sorti]

        newx = np.linspace(x[0], x[-1], 100)
        fity, (popt, perr) = scale(x, y, newx, **kwargs)
        return (newx, fity), (popt, perr)

    # @profile
    def collect(
            self,
            repeats=10,
            report_file=None,
            v=False):

        assert self.bm_units is not None

        for unit in self.bm_units:
            if v:
                print(f"Unit: {unit.id}")

            self._timings[unit.id] = time_unit(
                unit, repeats=repeats
                )

            if report_file is not None:
                self.save_timings(report_file)

            if v:
                print(f"    finished {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")

    def save_timings(self, report_file):
        with open(report_file, "w") as fp:
            json.dump(self._timings, fp, indent=4)

    def __str__(self):
        return_str = (
            "Run\n"
            f"    {self.run_name}\n"
            f"    {self.bm_units}\n"
        )

        return return_str


class BMUnit:

    def __init__(
            self, id,
            gen_func=None, gen_args=None, gen_kwargs=None,
            transform_func=None, transform_args=None, transform_kwargs=None,
            setup_func=None, setup_args=None, setup_kwargs=None,
            timed_args=None, timed_kwargs=None):

        self.id = id

        self.gen_func = gen_func
        if gen_args is None:
            gen_args = ()
        self.gen_args = gen_args
        if gen_kwargs is None:
            gen_kwargs = {}
        self.gen_kwargs = gen_kwargs

        self.transform_func = transform_func
        if transform_args is None:
            transform_args = ()
        self.transform_args = transform_args
        if transform_kwargs is None:
            transform_kwargs = {}
        self.transform_kwargs = transform_kwargs

        self.setup_func = setup_func
        if setup_args is None:
            setup_args = ()
        self.setup_args = setup_args
        if setup_kwargs is None:
            setup_kwargs = {}
        self.setup_kwargs = setup_kwargs

        if timed_args is None:
            timed_args = ()
        self.timed_args = timed_args
        if timed_kwargs is None:
            timed_kwargs = {}
        self.timed_kwargs = timed_kwargs

    def __str__(self):
        return_str = (
            "BMUnit\n"
            f"    id={self.id}\n"
            f"    gen_func={self.gen_func.__name__ if self.gen_func is not None else None}\n"
            f"        gen_args={self.gen_args}\n"
            f"        gen_kwargs={self.gen_kwargs}\n"
            f"    transform_func={self.transform_func.__name__ if self.transform_func is not None else None}\n"
            f"        transform_args={self.transform_args}\n"
            f"        transform_kwargs={self.transform_kwargs}\n"
            f"    setup_func={self.setup_func.__name__ if self.setup_func is not None else None}\n"
            f"        setup_args={self.setup_args}\n"
            f"        setup_kwargs={self.setup_kwargs}\n"
            f"    timed_args={self.timed_args}\n"
            f"    timed_kwargs={self.timed_kwargs}\n"
        )

        return return_str

    def __repr__(self):
        return f"{type(self).__name__}(id={self.id})"
