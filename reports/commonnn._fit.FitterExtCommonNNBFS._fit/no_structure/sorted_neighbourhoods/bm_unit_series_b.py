import subprocess


n_iterations = 10
n_points_list = [500 * 2**x for x in range(n_iterations)]
r_list = [0.1 for x in range(n_iterations)]
c_list = [0 for _ in range(n_iterations)]

start = 0
end = 10

for i in range(start, end):
    subprocess.run(
        f"python bm_unit.py -n {n_points_list[i]} -r {r_list[i]} -sim {c_list[i]}",
        shell=True
    )
