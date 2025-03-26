# run_experiments.py

import subprocess
import numpy as np

def update_config(B1, B2,r):
    with open('config/config.py', 'r') as file:
        lines = file.readlines()

    with open('config/config.py', 'w') as file:
        for line in lines:
            if line.startswith('B1 ='):
                file.write(f'B1 = {B1}\n')
            elif line.startswith('B2 ='):
                file.write(f'B2 = {B2}\n')
            elif line.startswith('r ='):
                file.write(f'r = {r}\n')
            else:
                file.write(line)

def run_experiment(case):
    command = ["python", "Opinf_with_reg_val.py", case]
    subprocess.run(command)

def run_post(r):
    command = ["python", "plot_coeffs.py", r]
    subprocess.run(command)

if __name__ == "__main__":

#   for r in [6, 8, 10, 16, 18, 20, 26, 28, 30, 36, 38, 40, 46, 48]:
#   for r in [6, 12, 18, 24, 30, 36, 42, 46]:
#   for r in [6, 12, 18, 24]:
#   for r in [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]:
#   for r in [4, 8, 12, 16, 20, 24]:
    for r in [28, 32, 36, 40]:
        for num in [8]:
#           B2_values = [np.logspace(-3., 5., num=num), np.logspace(-2., 6., num=num), np.logspace(-1., 7., num=num)] # These values are not working for r values >= 24.
#           B2_values = [np.logspace(0., 8., num=num), np.logspace(1., 9., num=num), np.logspace(2., 10., num=num), np.logspace(3., 11., num=num)]
#           B2_values = [np.logspace(0., 2., num=num), np.logspace(2., 4., num=num), np.logspace(4., 6., num=num), np.logspace(6., 8., num=num), np.logspace(8., 10., num=num)]
#           B1_values = [np.logspace(0., 5., num=6), np.logspace(0., 5., num=6), np.logspace(0., 5., num=6), np.logspace(0., 5., num=6)]
    # Generate B2_values using np.unique and np.concatenate
            B2_values = []
#           ranges = [(0., 1.), (1., 2.), (2., 3.), (3., 4.), (4., 5.), (5., 6.), (6., 7.), (7., 8.), (8., 9.),(9., 10.), (10., 11.), (11., 12.)]
            ranges = [(5., 6.), (6., 7.), (7., 8.), (8., 9.),(9., 10.), (10., 11.), (11., 12.)]

            for start, end in ranges:
#               B2_values.append(np.unique(np.concatenate([np.logspace(start, end, num=num),np.logspace(end, end+1, num=num)])))
                B2_values.append(np.unique(np.logspace(start, end, num=num)))

            B1_values = [np.logspace(0., 5., num=6)] * len(B2_values)

            for B1, B2 in zip(B1_values, B2_values):
                    update_config(B1.tolist(), B2.tolist(),r)
                    run_experiment('noefr')
                    run_experiment('efr')
                    run_post(str(r))
