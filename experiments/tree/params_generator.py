import os


params_path = "experiments/tree/params"
path_language = "experiments/tree/language_monoid.json"
seeds = [i for i in range(100)]
problems = ["navier_stokes", "schrodinger", "burger"]


if os.path.exists(params_path):
    os.remove(params_path)

for seed in seeds:
    for problem in problems:
        with open(params_path, "a") as f:
            f.write(f"{problem} -s {seed} -m 5 -k 5 -f 1000 -l {path_language} \n")