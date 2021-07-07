import json
import pathlib

from sklearn.metrics import pairwise_distances

from cnnclustering import hooks

import helper_base
import cnnclustering_fit_cases as cases


report_dir = pathlib.Path("reports/curta/cnnclustering_fit")
if not report_dir.is_dir():
    report_dir.mkdir(parents=True, exist_ok=True)

RUN_ARGUMENTS_MAP = {}
RUN_TIMINGS_MAP = {}

n_points_list = [500 * 2**x for x in range(10)]

run_list = [
    helper_base.Run(
        "no_structure", "a_a",
        {
            "gen_func": helper_base.gen_no_structure_points,
            "setup_func": cases.setup_commonnn_clustering__fit,
        },
        cases.gen_run_argument_list,
        r=0.25, c=0, n_list=n_points_list,
        setup_kwargs={"recipe": cases.default_recipe}
    ),
    helper_base.Run(
        "no_structure", "b_a",
        {
            "gen_func": helper_base.gen_no_structure_points,
            "transform_func": pairwise_distances,
            "setup_func": cases.setup_commonnn_clustering__fit,
        },
        cases.gen_run_argument_list,
        r=0.25, c=0, n_list=[500 * 2**x for x in range(8)],
        setup_kwargs={"recipe": cases.distance_recipe},
    ),
    helper_base.Run(
        "no_structure", "c_a",
        {
            "gen_func": helper_base.gen_no_structure_points,
            "transform_func": helper_base.compute_neighbours,
            "setup_func": cases.setup_commonnn_clustering__fit,
        },
        cases.gen_run_argument_list,
        r=0.25, c=0,
        n_list=n_points_list,
        transform_args=(0.25,),
        setup_kwargs={
            "preparation_hook": hooks.prepare_neighbourhoods,
            "recipe": cases.neighbours_recipe
            },
    ),
    helper_base.Run(
        "no_structure", "d_a",
        {
            "gen_func": helper_base.gen_no_structure_points,
            "transform_func": helper_base.compute_neighbours,
            "setup_func": cases.setup_commonnn_clustering__fit,
        },
        cases.gen_run_argument_list,
        r=0.25, c=0,
        n_list=n_points_list,
        transform_args=(0.25,),
        transform_kwargs={"sort": True},
        setup_kwargs={
            "preparation_hook": hooks.prepare_neighbourhoods,
            "recipe": cases.neighbours_sorted_recipe
            },
    ),
    helper_base.Run(
        "no_structure", "d_b",
        {
            "gen_func": helper_base.gen_no_structure_points,
            "transform_func": helper_base.compute_neighbours,
            "setup_func": cases.setup_commonnn_clustering__fit,
        },
        cases.gen_run_argument_list,
        r=0.25, c=50,
        n_list=n_points_list,
        transform_args=(0.25,),
        transform_kwargs={"sort": True},
        setup_kwargs={
            "preparation_hook": hooks.prepare_neighbourhoods,
            "recipe": cases.neighbours_sorted_recipe
            },
    ),
    helper_base.Run(
        "no_structure", "d_c",
        {
            "gen_func": helper_base.gen_no_structure_points,
            "transform_func": helper_base.compute_neighbours,
            "setup_func": cases.setup_commonnn_clustering__fit,
        },
        cases.gen_run_argument_list,
        r=0.25, c=100,
        n_list=n_points_list,
        transform_args=(0.25,),
        transform_kwargs={"sort": True},
        setup_kwargs={
            "preparation_hook": hooks.prepare_neighbourhoods,
            "recipe": cases.neighbours_sorted_recipe
            },
    ),
    helper_base.Run(
        "no_structure", "d_d",
        {
            "gen_func": helper_base.gen_no_structure_points,
            "transform_func": helper_base.compute_neighbours,
            "setup_func": cases.setup_commonnn_clustering__fit,
        },
        cases.gen_run_argument_list,
        r=0.1, c=0,
        n_list=n_points_list,
        transform_args=(0.1,),
        transform_kwargs={"sort": True},
        setup_kwargs={
            "preparation_hook": hooks.prepare_neighbourhoods,
            "recipe": cases.neighbours_sorted_recipe
            },
    ),
    helper_base.Run(
        "no_structure", "d_e",
        {
            "gen_func": helper_base.gen_no_structure_points,
            "transform_func": helper_base.compute_neighbours,
            "setup_func": cases.setup_commonnn_clustering__fit,
        },
        cases.gen_run_argument_list,
        r=0.1, c=50,
        n_list=n_points_list,
        transform_args=(0.1,),
        transform_kwargs={"sort": True},
        setup_kwargs={
            "preparation_hook": hooks.prepare_neighbourhoods,
            "recipe": cases.neighbours_sorted_recipe
            },
    ),
    helper_base.Run(
        "no_structure", "d_f",
        {
            "gen_func": helper_base.gen_no_structure_points,
            "transform_func": helper_base.compute_neighbours,
            "setup_func": cases.setup_commonnn_clustering__fit,
        },
        cases.gen_run_argument_list,
        r=0.1, c=100,
        n_list=n_points_list,
        transform_args=(0.1,),
        transform_kwargs={"sort": True},
        setup_kwargs={
            "preparation_hook": hooks.prepare_neighbourhoods,
            "recipe": cases.neighbours_sorted_recipe
            },
    ),
]

if __name__ == "__main__":
    for run in run_list:
        full_run_name = f"{run.run_name}_run_{run.case_name}"
        report_file = report_dir / f"{full_run_name}_raw.json"

        RUN_TIMINGS_MAP[full_run_name] = {}

        helper_base.collect_timings(
            run_arguments_list=run.run_argument_list,
            timings=RUN_TIMINGS_MAP[full_run_name],
            **run.function_map
        )

        with open(report_file, "w") as fp:
            json.dump(RUN_TIMINGS_MAP[full_run_name], fp, indent=4)
