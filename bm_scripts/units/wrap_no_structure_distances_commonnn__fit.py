import subprocess

n_iterations = 5
n_points_list = [500 * 2**x for x in range(n_iterations)]
r_list = [0.25 for x in range(n_iterations)]
c_list = [0 for _ in range(n_iterations)]

for i in range(len(n_points_list)):
    subprocess.run(
        f"python no_structure_distances_commonnn__fit.py -n {n_points_list[i]} -r {r_list[i]} -sim {c_list[i]}",
        shell=True
        )
    ofile = f"/home/janjoswig/repo/CommonNNClustering-Benchmark/reports_unit/qcm07/no_structure/commonnn__fit/distances/{n_points_list[i]}_2_{r_list[i]}_{c_list[i]}_scaleneprofile.json"
    subprocess.run(
        f"scalene --cli --off --json --outfile {ofile} no_structure_distances_commonnn__fit.py -n {n_points_list[i]} -r {r_list[i]} -sim {c_list[i]} --profile",
        shell=True
        )