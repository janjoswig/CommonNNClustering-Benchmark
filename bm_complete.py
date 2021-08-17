import json
import pathlib

from sklearn.metrics import pairwise_distances

from cnnclustering import hooks

import helper_base
import cnnclustering_fit_cases as cases


report_dir = pathlib.Path("reports/curta/cnnclustering_fit/complete")
if not report_dir.is_dir():
    report_dir.mkdir(parents=True, exist_ok=True)

RUN_ARGUMENTS_MAP = {}
RUN_TIMINGS_MAP = {}

n_points_list = [500 * 2**x for x in range(10)]

run_list = [
    helper_base.Run(
        "varied", "a_a",
        {
            "gen_func": helper_base.gen_blobs_points,
            "setup_func": cases.setup_commonnn_clustering_complete,
        },
        cases.gen_run_argument_list_cnnclustering_complete,
        r=0.18, c=20, n_list=n_points_list,
        gen_kwargs={"random_state": 170, "cluster_std": [1.0, 2.5, 0.5]},
        setup_kwargs={
            "recipe": cases.default_recipe
            }
    ),
    helper_base.Run(
        "varied", "b_a",
        {
            "gen_func": helper_base.gen_blobs_points,
            "transform_func": pairwise_distances,
            "setup_func": cases.setup_commonnn_clustering_complete,
        },
        cases.gen_run_argument_list_cnnclustering_complete,
        r=0.18, c=20, n_list=[500 * 2**x for x in range(8)],
        gen_kwargs={"random_state": 170, "cluster_std": [1.0, 2.5, 0.5]},
        setup_kwargs={"recipe": cases.distance_recipe},
    ),
    helper_base.Run(
        "varied", "c_a",
        {
            "gen_func": helper_base.gen_blobs_points,
            "setup_func": cases.setup_commonnn_clustering_complete,
        },
        cases.gen_run_argument_list_cnnclustering_complete,
        r=0.18, c=20,
        n_list=n_points_list,
        gen_kwargs={"random_state": 170, "cluster_std": [1.0, 2.5, 0.5]},
        setup_kwargs={
            "transform_func": helper_base.compute_neighbours,
            "transform_args": (0.18,),
            "preparation_hook": hooks.prepare_neighbourhoods,
            "recipe": cases.neighbours_recipe
            },
    ),
    helper_base.Run(
        "varied", "d_a",
        {
            "gen_func": helper_base.gen_blobs_points,
            "setup_func": cases.setup_commonnn_clustering_complete,
        },
        cases.gen_run_argument_list_cnnclustering_complete,
        r=0.18, c=20,
        n_list=n_points_list,
        gen_kwargs={"random_state": 170, "cluster_std": [1.0, 2.5, 0.5]},
        setup_kwargs={
            "transform_args": (0.18,),
            "transform_kwargs": {"sort": True},
            "transform_func": helper_base.compute_neighbours,
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
