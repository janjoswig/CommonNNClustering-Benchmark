from commonnnbm import base
import commonnnbm.cases.commonnn_fit as cases


n_points_list = [500 * 2**x for x in range(10)]
r_list = [0.2 * 0.9**x for x in range(10)]
# c_list = [int(2 * 2**x) for x in range(10)]
c_list = [50 for _ in range(10)]
d_list = [2 for _ in range(10)]

start = 0  # inclusive
end = 3   # exclusive

raw_run_list = [
    (
        "varied_a_a_scale_r",
        {
            "r_list": r_list[start:end],
            "c_list": c_list[start:end],
            "d_list": d_list[start:end],
            "n_list": n_points_list[start:end],
            "gen_func": base.gen_blobs_points,
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
    base.Run(
        run_name,
        cases.gen_bm_units_cnnclustering_complete(**kwargs),
    )
    for run_name, kwargs in raw_run_list
)

runs_report_dir = "cnnclustering_fit/complete"
