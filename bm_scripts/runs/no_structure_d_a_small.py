from commonnnbm import base
import commonnnbm.cases.commonnn_fit as cases


n_points_list = [500 * 2**x for x in range(10)]
r_list = [0.25 for x in range(10)]
c_list = [0 for _ in range(10)]
d_list = [2 for _ in range(10)]

start = 0  # inclusive
end = 6    # exclusive

raw_run_list = [
    (
        "no_structure_d_a",
        {
            "r_list": r_list[start:end],
            "c_list": c_list[start:end],
            "d_list": d_list[start:end],
            "n_list": n_points_list[start:end],
            "gen_func": base.gen_no_structure_points,
            "transform_func": base.compute_neighbours,
            "transform_args": ("<r>",),
            "transform_kwargs": {"sort": True},
            "setup_kwargs": {
                "recipe": "sorted_neighbourhoods"
                }
            }
        ),
    ]

run_list = (
    base.Run(
        run_name,
        cases.gen_bm_units_commonnn__fit(**kwargs),
    )
    for run_name, kwargs in raw_run_list
)

runs_report_dir = "commonnn_fit"
