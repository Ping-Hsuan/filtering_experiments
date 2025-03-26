import numpy as np
from itertools import product
from sys import argv
from time import time

#Qhat_full_file  = './OpInf_results/ref_red_sol_training_r22.npy'
#Qhat_full_file  = './datasets_cases_360_deg/ref_red_sol_full_domain_train_size_1268.npy'
Qhat_full_file  = './datasets_cases_360_deg/ref_red_sol_full_domain_train_size_2536.npy'


# DoF setup
ns 	= 12
nx 	= 314874
n  	= int(nx*ns)

# number of training snapshots
nt = 1268

# state variable names
state_variables = ['Density', 'Temperature', 'Pressure', \
                   'CH4_mass_fraction', 'O2_mass_fraction', 'H2O_mass_fraction', 'CO2_mass_fraction', 'H2_mass_fraction', 'CO_mass_fraction', \
                   'U-Velocity',  'V-Velocity', 'W-Velocity']
 
# number of time instants over the time domain of interest (training + prediction)
nt_p = 2536

# number of time instants over the time domain of interest (training + validation + prediction)
nt_all = 3751

# define target retained energy for the OpInf ROM
# target_ret_energy = 0.9996

# target reduced dimension
r = 28

# ranges for the regularization parameter pairs
B1 = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
B2 = [1000000.0, 1389495.4943731388, 1930697.7288832497, 2682695.7952797273, 3727593.720314938, 5179474.6792312125, 7196856.730011513, 10000000.0]
#B1 = np.logspace(0., 5., 12)
#B2 = np.logspace(-2., 6., 16)

# Create label for directory name
B2_max, B2_min, B2_length = map(lambda x: x(B2), [np.max, np.min, len])
B1_max, B1_min, B1_length = map(lambda x: x(B1), [np.max, np.min, len])
B1_min_exp, B1_max_exp, B2_min_exp, B2_max_exp = map(lambda x: int(np.floor(np.log10(x))), [B1_min, B1_max, B2_min, B2_max])
reg_name = f"B1_{B1_min_exp}_{B1_max_exp}_{B1_length}_B2_{B2_min_exp}_{B2_max_exp}_{B2_length}"

#fmode_list = np.linspace(-2, -20, 10, dtype=int)
#fmode_list = np.arange(-2, -min(r+1, 21), -2, dtype=int)
#fmode_list = np.linspace(-1, max(-r,-10), min(r,10), dtype=int)
fmode_list = np.linspace(-1, max(-r,-5), min(r,5), dtype=int)
#chi_list = np.logspace(-3., 0., 16)
chi_list = np.logspace(-3., 0., 8)


rcut_max, rcut_min, rcut_length = map(lambda x: x(fmode_list), [np.max, np.min, len])
chi_max, chi_min, chi_length = map(lambda x: x(chi_list), [np.max, np.min, len])
chi_min_exp, chi_max_exp = map(lambda x: int(np.floor(np.log10(x))), [chi_min, chi_max])
reg_f_name = f"rcut_{rcut_min}_{rcut_max}_{rcut_length}_chi_{chi_min_exp}_{chi_max_exp}_{chi_length}"

# maximum variance of reduced training data for optimal regularization parameter selection
max_growth = 1.2

# flag to determine whether the (transformed) training data are centered with respect to the temporal mean 
CENTERING = True

# flag to determine whether the (transformed) training data are scaled by the maximum absolute value of each state variable
SCALING = True

# flag to determine whether we postprocess the OpInf reduced solution
POSTPROC = False

# flag to determine whether we compute the ROM approximate solution in the original cooridnates in the full domain 
# at user specified time instants (specified in target_time_instants)
POSTPROC_FULL_DOM_SOL = False

# flag to determine whether we compute the ROM approximate solution in the original coordinates
# at user specified probe locations (specified in target_probe_indices)
POSTPROC_PROBES = True

# time instants at which to save the OpInf approximate solutions mappend to the original coordinates
target_time_instants 	= [50, 240, 355, 410]

dir_save = 'stdreg_set2'