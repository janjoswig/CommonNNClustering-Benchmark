import json
import pathlib

from sklearn.metrics import pairwise_distances

from cnnclustering import hooks

import helper_base
import cnnclustering_fit_cases as cases


report_dir = pathlib.Path("reports/curta/cnnclustering_fit")
if not report_dir.is_dir():
    report_dir.mkdir(parents=True, exist_ok=True)

n_points_list = [500 * 2**x for x in range(10)]
r_list = 0.25
c_list = 0

raw_run_list = [
    (
        "no_structure_e_a",
        {
            "r_list": r_list, "c_list": c_list, "d_list": 2,
            "n_list": n_points_list,
            "gen_func": helper_base.gen_no_structure_points,
            "setup_kwargs": {
                "transform_func": helper_base.compute_neighbours,
                "transform_args": ("<r>",),
                "transform_kwargs": {"sort": True},
                "preparation_hook": hooks.prepare_neighbourhoods,
                "recipe": cases.neighbours_recipe
                }
            }
        ),
]

run_list = (
    helper_base.Run(
        run_name,
        cases.gen_bm_units_cnnclustering__fit(**kwargs),
    )
    for run_name, kwargs in raw_run_list
)

if __name__ == "__main__":
    for run in run_list:

        report_file = report_dir / f"{run.run_name}_raw.json"

        run.collect(v=True)

        with open(report_file, "w") as fp:
            json.dump(run.timings, fp, indent=4)