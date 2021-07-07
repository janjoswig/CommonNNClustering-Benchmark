import json
import pathlib
import timeit

import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

from cnnclustering import cluster, hooks
from cnnclustering import _fit, _types, _primitive_types


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


def setup_commonnn_clustering__fit(
        data,
        registered_recipe_key="None",
        preparation_hook=hooks.prepare_pass,
        recipe=None):
    """Prepare benchmark of :meth:`cnnclustering.cluster.Clustering._fit`"""

    if recipe is None:
        recipe = {}

    builder = cluster.ClusteringBuilder(
        data,
        preparation_hook=preparation_hook,
        registered_recipe_key=registered_recipe_key,
        **recipe
        )
    clustering = builder.build()
    clustering._labels = _types.Labels(
        np.zeros(
            clustering._input_data.n_points,
            order="C", dtype=_primitive_types.P_AINDEX
            )
        )

    return clustering._fit


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


default_recipe = {
    "input_data": _types.InputDataExtComponentsMemoryview,
    "fitter": _fit.FitterExtBFS,
    "fitter.neighbours_getter": _types.NeighboursGetterExtBruteForce,
    "fitter.ngetter.distance_getter": _types.DistanceGetterExtMetric,
    "fitter.ngetter.dgetter.metric": _types.MetricExtEuclideanReduced,
    "fitter.neighbours": (_types.NeighboursExtVectorCPPUnorderedSet, (5000,), {}),
    "fitter.neighbour_neighbours": (
        _types.NeighboursExtVectorCPPUnorderedSet, (5000,), {}
        ),
    "fitter.similarity_checker": _types.SimilarityCheckerExtSwitchContains,
    "fitter.queue": _types.QueueExtFIFOQueue,
}

distance_recipe = {
    "input_data": _types.InputDataExtComponentsMemoryview,
    "fitter": _fit.FitterExtBFS,
    "fitter.neighbours_getter": _types.NeighboursGetterExtBruteForce,
    "fitter.ngetter.distance_getter": _types.DistanceGetterExtMetric,
    "fitter.ngetter.dgetter.metric": _types.MetricExtPrecomputed,
    "fitter.neighbours": (_types.NeighboursExtVectorCPPUnorderedSet, (5000,), {}),
    "fitter.neighbour_neighbours": (
        _types.NeighboursExtVectorCPPUnorderedSet, (5000,), {}
        ),
    "fitter.similarity_checker": _types.SimilarityCheckerExtSwitchContains,
    "fitter.queue": _types.QueueExtFIFOQueue,
}

neighbours_recipe = {
    "input_data": _types.InputDataExtNeighbourhoodsMemoryview,
    "fitter": _fit.FitterExtBFS,
    "fitter.neighbours_getter": _types.NeighboursGetterExtLookup,
    "fitter.neighbours": (_types.NeighboursExtVectorCPPUnorderedSet, (5000,), {}),
    "fitter.neighbour_neighbours": (
        _types.NeighboursExtVectorCPPUnorderedSet, (5000,), {}
        ),
    "fitter.similarity_checker": _types.SimilarityCheckerExtSwitchContains,
    "fitter.queue": _types.QueueExtFIFOQueue,
}

neighbours_sorted_recipe = {
    "input_data": _types.InputDataExtNeighbourhoodsMemoryview,
    "fitter": _fit.FitterExtBFS,
    "fitter.neighbours_getter": _types.NeighboursGetterExtLookup,
    "fitter.neighbours": (_types.NeighboursExtVector, (5000,), {}),
    "fitter.neighbour_neighbours": (_types.NeighboursExtVector, (5000,), {}),
    "fitter.similarity_checker": _types.SimilarityCheckerExtScreensorted,
    "fitter.queue": _types.QueueExtFIFOQueue,
}

if __name__ == "__main__":
    report_dir = pathlib.Path("reports/curta/cnnclustering_fit")
    if not report_dir.is_dir():
        report_dir.mkdir(parents=True, exist_ok=True)

    n_points_list = [500 * 2**x for x in range(2)]

    RUN_ARGUMENTS_MAP = {}
    RUN_TIMINGS_MAP = {}

    run_name = "no_structure_run_a"
    report_file = report_dir / f"{run_name}_raw.json"

    gen_func = gen_no_structure_points
    transform_func = None
    setup_func = setup_commonnn_clustering__fit

    radius_cutoff = 0.25
    cnn_cutoff = 0

    RUN_ARGUMENTS_MAP[run_name] = []
    for n_points in n_points_list:
        RUN_ARGUMENTS_MAP[run_name].append(
            {
                "id": str(n_points),
                "gen": (
                    ((n_points, 2),), {}
                ),
                "setup": (
                    (), {"recipe": default_recipe}
                ),
                "timed": (
                    (_types.ClusterParameters(radius_cutoff, cnn_cutoff),), {}
                ),
            }
        )

    RUN_TIMINGS_MAP[run_name] = {}
    collect_timings(
        gen_func,
        setup_func,
        RUN_ARGUMENTS_MAP[run_name],
        transform_func=transform_func,
        timings=RUN_TIMINGS_MAP[run_name]
    )

    with open(report_file, "w") as fp:
        json.dump(RUN_TIMINGS_MAP[run_name], fp, indent=4)
