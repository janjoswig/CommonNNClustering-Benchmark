import json
import pathlib

from sklearn.metrics import pairwise_distances

from cnnclustering import hooks

import helper_base
import cnnclustering_fit_cases as cases


report_dir = pathlib.Path("reports/curta/cnnclustering_fit/complete")
if not report_dir.is_dir():
    report_dir.mkdir(parents=True, exist_ok=True)

n_points_list = [500 * 2**x for x in range(10)]

# run_list = (
    # helper_base.Run(
    #     "varied", "a_a",
    #     {
    #         "gen_func": helper_base.gen_blobs_points,
    #         "setup_func": cases.setup_commonnn_clustering_complete,
    #     },
    #     cases.gen_run_argument_list_cnnclustering_complete,
    #     r=0.18, c=20, n_list=n_points_list,
    #     gen_kwargs={"random_state": 170, "cluster_std": [1.0, 2.5, 0.5]},
    #     setup_kwargs={
    #         "recipe": cases.default_recipe
    #         }
    # ),
    # helper_base.Run(
    #     "varied", "b_a",
    #     {
    #         "gen_func": helper_base.gen_blobs_points,
    #         "transform_func": pairwise_distances,
    #         "setup_func": cases.setup_commonnn_clustering_complete,
    #     },
    #     cases.gen_run_argument_list_cnnclustering_complete,
    #     r=0.18, c=20, n_list=[500 * 2**x for x in range(8)],
    #     gen_kwargs={"random_state": 170, "cluster_std": [1.0, 2.5, 0.5]},
    #     setup_kwargs={"recipe": cases.distance_recipe},
    # ),
    # (
    #     helper_base.Run,
    #     (
    #         "varied", "c_a",
    #         {
    #             "gen_func": helper_base.gen_blobs_points,
    #             "setup_func": cases.setup_commonnn_clustering_complete,
    #         },
    #         cases.gen_run_argument_list_cnnclustering_complete
    #         ),
    #     {
    #         "r": 0.18, "c": 20,
    #         "n_list": n_points_list,
    #         "gen_kwargs": {
    #             "random_state": 170,
    #             "cluster_std": [1.0, 2.5, 0.5]
    #             },
    #         "setup_kwargs": {
    #             "transform_func": helper_base.compute_neighbours,
    #             "transform_args": (0.18,),
    #             "preparation_hook": hooks.prepare_neighbourhoods,
    #             "recipe": cases.neighbours_recipe
    #             }
    #         }
    #     ),
    # helper_base.Run(
    #     "varied", "d_a",
    #     {
    #         "gen_func": helper_base.gen_blobs_points,
    #         "setup_func": cases.setup_commonnn_clustering_complete,
    #     },
    #     cases.gen_run_argument_list_cnnclustering_complete,
    #     r=0.18, c=20,
    #     n_list=n_points_list,
    #     gen_kwargs={"random_state": 170, "cluster_std": [1.0, 2.5, 0.5]},
    #     setup_kwargs={
    #         "transform_args": (0.18,),
    #         "transform_kwargs": {"sort": True},
    #         "transform_func": helper_base.compute_neighbours,
    #         "preparation_hook": hooks.prepare_neighbourhoods,
    #         "recipe": cases.neighbours_sorted_recipe
    #         },
    # ),
#)


raw_run_list = [
    (
        "varied_c_a",
        {
            "r": 0.18, "c": 20, "d": 2,
            "n_list": n_points_list,
            "gen_func": helper_base.gen_blobs_points,
            "gen_kwargs": {
                "random_state": 170,
                "cluster_std": [1.0, 2.5, 0.5]
                },
            "setup_kwargs": {
                "transform_func": helper_base.compute_neighbours,
                "transform_args": (0.18,),
                "preparation_hook": hooks.prepare_neighbourhoods,
                "recipe": cases.neighbours_recipe
                }
            }
        )
    ]

run_list = (
    helper_base.Run(
        run_name,
        cases.gen_bm_units_cnnclustering_complete(**kwargs),
    )
    for run_name, kwargs in raw_run_list
)

if __name__ == "__main__":
    for run in run_list:

        report_file = report_dir / f"{run.run_name}_raw.json"

        run.collect()

        with open(report_file, "w") as fp:
            json.dump(run.timings, fp, indent=4)
