import json
import pathlib

from sklearn.metrics import pairwise_distances

from cnnclustering import recipes

import helper_base
import cnnclustering_fit_cases as cases


report_dir = pathlib.Path("reports/curta/cnnclustering_fit/complete")
if not report_dir.is_dir():
    report_dir.mkdir(parents=True, exist_ok=True)

n_points_list = [500 * 2**x for x in range(10)]
r_list = [0.2 * 0.95**x for x in range(10)]
c_list = [int(2 * 2**x) for x in range(10)]

raw_run_list = [
    (
        "varied_a_a_scale_r",
        {
            "r_list": r_list[-2:], "c_list": c_list[-2:], "d_list": 2,
            "n_list": n_points_list[-2:],
            "gen_func": helper_base.gen_blobs_points,
            "gen_kwargs": {
                "random_state": 170,
                "cluster_std": [1.0, 2.5, 0.5]
                },
            "setup_kwargs": {
                "recipe": cases.default_recipe
                }
            }
        ),
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

        run.collect(v=True, report_file=report_file)
