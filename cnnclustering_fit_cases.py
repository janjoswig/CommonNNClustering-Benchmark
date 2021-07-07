import numpy as np

from cnnclustering import cluster, hooks
from cnnclustering import _fit, _types, _primitive_types


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


def setup_commonnn_clustering_fit(
        data,
        registered_recipe_key="None",
        preparation_hook=hooks.prepare_pass,
        recipe=None):
    """Prepare benchmark of :meth:`cnnclustering.cluster.Clustering.fit`"""

    if recipe is None:
        recipe = {}

    builder = cluster.ClusteringBuilder(
        data,
        preparation_hook=preparation_hook,
        registered_recipe_key=registered_recipe_key,
        **recipe
        )
    clustering = builder.build()

    return clustering.fit


def setup_commonnn_clustering_complete(
        data,
        transform_func=None,
        transform_args=None,
        transform_kwargs=None,
        registered_recipe_key="None",
        preparation_hook=hooks.prepare_pass,
        recipe=None):
    """Prepare benchmark of :meth:`cnnclustering.cluster.Clustering._fit` inkl. preparatiotransform_func"""

    if recipe is None:
        recipe = {}

    def fit_complete(*args, **kwargs):

        if transform_func is not None:
            if transform_args is None:
                transform_args = ()

            if transform_kwargs is None:
                transform_kwargs = {}

            data = transform_func(data, *transform_args, **transform_kwargs)

        builder = cluster.ClusteringBuilder(
            data,
            preparation_hook=preparation_hook,
            registered_recipe_key=registered_recipe_key,
            **recipe
            )
        clustering = builder.build()
        clustering.fit(*args, **kwargs)

    return fit_complete


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


def gen_run_argument_list(
        r, c, n_list,
        transform_args=None, transform_kwargs=None,
        setup_args=None, setup_kwargs=None):

    if transform_args is None:
        transform_args = ()

    if transform_kwargs is None:
        transform_kwargs = {}

    if setup_args is None:
        setup_args = ()

    if setup_kwargs is None:
        setup_kwargs = {}

    run_argument_list = []
    for n in n_list:
        run_argument_list.append(
            {
                "id": str(n),
                "gen": (
                    ((n, 2),), {}
                ),
                "transform": (
                    transform_args, transform_kwargs
                ),
                "setup": (
                    setup_args, setup_kwargs
                ),
                "timed": (
                    (_types.ClusterParameters(r, c),), {}
                ),
            }
        )

    return run_argument_list
