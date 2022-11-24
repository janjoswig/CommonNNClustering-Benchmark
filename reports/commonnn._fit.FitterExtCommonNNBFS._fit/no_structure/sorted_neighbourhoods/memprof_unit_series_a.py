import subprocess

from commonnnbm import base

n_iterations = 10
n_points_list = [500 * 2**x for x in range(n_iterations)]
r_list = [0.25 for x in range(n_iterations)]
c_list = [0 for _ in range(n_iterations)]

for i in range(len(n_points_list)):
    for _ in range(5):
        process = subprocess.run(
            f"python -m memory_profiler memprof_unit.py -n {n_points_list[i]} -r {r_list[i]} -sim {c_list[i]}",
            shell=True,
            capture_output=True,
            encoding="utf8"
        )

        base.total_increment_from_memprof_report(process.stdout)
