import pathlib
import sys

repo_dir = pathlib.Path("/home/janjoswig/repo/CommonNNClustering/docsrc/benchmark")
sys.path.insert(0, str(repo_dir))

from sklearn.metrics import pairwise_distances

from cnnclustering import recipes

import helper_base
import cnnclustering_fit_cases as cases


report_dir = repo_dir / "reports/curta/cnnclustering_fit/complete"
if not report_dir.is_dir():
    report_dir.mkdir(parents=True, exist_ok=True)

start = 8   # inclusive
stop = 10   # exclusive

part = 2

n_points_list = [500 * 2**x for x in range(10)]
r_list = [0.2 * 0.9**x for x in range(10)]
# c_list = [int(2 * 2**x) for x in range(10)]
c = 50
d = 3


raw_run_list = [
    (
        "varied_b_a_scale_r",
        {
            "r_list": r_list[start:stop],
            "c_list": c,
            "d_list": d,
            "n_list": n_points_list[start:stop],
            "gen_func": helper_base.gen_blobs_points,
            "gen_kwargs": {
                "random_state": 170,
                "cluster_std": [1.0, 2.5, 0.5]
                },
            "setup_kwargs": {
                "transform_func": pairwise_distances,
                "recipe": cases.distance_recipe
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

        report_file = report_dir / f"{run.run_name}_raw_{part}.json"

        run.collect(
            v=True, report_file=report_file,
<<<<<<< HEAD
            repeats=5
=======
            repeats=2
>>>>>>> 3c885cadf8a86ff901c89249232740b6e86a1329
            )
