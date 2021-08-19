import json
import pathlib

from sklearn.metrics import pairwise_distances

from cnnclustering import hooks

import helper_base
import cnnclustering_fit_cases as cases


report_dir = pathlib.Path("reports/qcm07/cnnclustering_fit/complete")
if not report_dir.is_dir():
    report_dir.mkdir(parents=True, exist_ok=True)

n_points_list = [500 * 2**x for x in range(8)]
r_list = [0.2 * 0.95**x for x in range(8)]
c_list = [int(2 * 2**x) for x in range(8)]

#n_points_list = n_points_list[4]
#n_points_list = n_points_list[4]
#n_points_list = n_points_list[4]


raw_run_list = [
    (
        "varied_c_a",
        {
            "r_list": r_list, "c_list": c_list, "d_list": 2,
            "n_list": n_points_list,
            "gen_func": helper_base.gen_blobs_points,
            "gen_kwargs": {
                "random_state": 170,
                "cluster_std": [1.0, 2.5, 0.5]
                },
            "setup_kwargs": {
                "transform_func": helper_base.compute_neighbours,
                "transform_args": ("<r>",),
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

        run.collect(v=True)

        with open(report_file, "w") as fp:
            json.dump(run.timings, fp, indent=4)
