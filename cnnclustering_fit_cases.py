from functools import partial

import numpy as np

from cnnclustering import cluster, recipes
from cnnclustering import _fit, _types, _primitive_types

import helper_base


def setup_commonnn_clustering__fit(
        data,
        registered_recipe_key="None",
        preparation_hook=recipes.prepare_pass,
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
        preparation_hook=recipes.prepare_pass,
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
        preparation_hook=recipes.prepare_pass,
        recipe=None):
    """Prepare benchmark of :meth:`cnnclustering.cluster.Clustering._fit`

    Timings will include preparatiotransform_func
    """

    if recipe is None:
        recipe = {}

    if transform_func is None:
        transform_func = lambda x: x

    if transform_args is None:
        transform_args = ()

    if transform_kwargs is None:
        transform_kwargs = {}

    def fit_complete(*args, **kwargs):

        builder = cluster.ClusteringBuilder(
            transform_func(data, *transform_args, **transform_kwargs),
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


def gen_bm_units_cnnclustering__fit(
        r_list, c_list, d_list, n_list,
        gen_func, gen_kwargs=None,
        transform_func=None, transform_args=None, transform_kwargs=None,
        setup_args=None, setup_kwargs=None):
    """Case generation function to vary data set size"""

    if not isinstance(r_list, list):
        r_list = [r_list]

    if not isinstance(c_list, list):
        c_list = [c_list]

    if not isinstance(d_list, list):
        d_list = [d_list]

    if not isinstance(n_list, list):
        n_list = [n_list]

    parameter_length = max(len(_list) for _list in [r_list, c_list, d_list, n_list])

    if len(r_list) != parameter_length:
        r_list = r_list[:1] * parameter_length

    if len(c_list) != parameter_length:
        c_list = c_list[:1] * parameter_length

    if len(d_list) != parameter_length:
        d_list = d_list[:1] * parameter_length

    if len(n_list) != parameter_length:
        n_list = n_list[:1] * parameter_length

    for index, n in enumerate(n_list):

        sub = partial(
            substitute_placeholders,
            placeholder_map={
                "<r>": r_list[index],
                "<c>": c_list[index],
                "<d>": d_list[index],
                "<n>": n_list[index],
                }
            )

        bm_unit = helper_base.BMUnit(
            id=str(n),
            gen_func=gen_func,
            gen_args=((n, d_list[index]),),
            gen_kwargs=sub(gen_kwargs),
            transform_func=transform_func,
            transform_args=sub(transform_args),
            transform_kwargs=sub(transform_kwargs),
            setup_func=setup_commonnn_clustering__fit,
            setup_args=sub(setup_args),
            setup_kwargs=sub(setup_kwargs),
            timed_args=(_types.ClusterParameters(r_list[index], c_list[index]),),
            timed_kwargs={}
            )

        yield bm_unit


def gen_bm_units_cnnclustering_complete(
        r_list, c_list, d_list, n_list,
        gen_func, gen_kwargs=None,
        transform_func=None, transform_args=None, transform_kwargs=None,
        setup_args=None, setup_kwargs=None):
    """Case generation function to vary data set size"""

    if not isinstance(r_list, list):
        r_list = [r_list]

    if not isinstance(c_list, list):
        c_list = [c_list]

    if not isinstance(d_list, list):
        d_list = [d_list]

    if not isinstance(n_list, list):
        n_list = [n_list]

    parameter_length = max(len(_list) for _list in [r_list, c_list, d_list, n_list])

    if len(r_list) != parameter_length:
        r_list = r_list[:1] * parameter_length

    if len(c_list) != parameter_length:
        c_list = c_list[:1] * parameter_length

    if len(d_list) != parameter_length:
        d_list = d_list[:1] * parameter_length

    if len(n_list) != parameter_length:
        n_list = n_list[:1] * parameter_length

    for index, n in enumerate(n_list):

        sub = partial(
            substitute_placeholders,
            placeholder_map={
                "<r>": r_list[index],
                "<c>": c_list[index],
                "<d>": d_list[index],
                "<n>": n_list[index],
                }
            )

        bm_unit = helper_base.BMUnit(
            id=str(n),
            gen_func=gen_func,
            gen_args=((n, d_list[index]),),
            gen_kwargs=sub(gen_kwargs),
            transform_func=transform_func,
            transform_args=sub(transform_args),
            transform_kwargs=sub(transform_kwargs),
            setup_func=setup_commonnn_clustering_complete,
            setup_args=sub(setup_args),
            setup_kwargs=sub(setup_kwargs),
            timed_args=(r_list[index], c_list[index]), timed_kwargs={
                "record": False, "record_time": False,
                "info": False, "sort_by_size": False
                }
            )

        yield bm_unit


def substitute_placeholders(value, placeholder_map):

    if isinstance(value, str):
        return placeholder_map.get(value, value)

    if isinstance(value, tuple):
        # args
        processed_args = []
        for arg in value:
            processed_args.append(substitute_placeholders(arg, placeholder_map))

        return tuple(processed_args)

    if isinstance(value, dict):
        # kwargs
        processed_kwargs = {}
        for kw, v in value.items():
            processed_kwargs[kw] = substitute_placeholders(v, placeholder_map)

        return processed_kwargs

    return value
