import numpy as np
import h5py as h5
from mpi4py import MPI
from itertools import product
from sys import argv
from time import time

Qhat_full_file  = './OpInf_results/ref_red_sol_training_r22.npy'


# DoF setup
ns 	= 12
nx 	= 314874
n  	= int(nx*ns)

# number of training snapshots
nt = 294

# state variable names
state_variables = ['Density', 'Temperature', 'Pressure', \
                   'CH4_mass_fraction', 'O2_mass_fraction', 'H2O_mass_fraction', 'CO2_mass_fraction', 'H2_mass_fraction', 'CO_mass_fraction', \
                   'U-Velocity',  'V-Velocity', 'W-Velocity']
 
# number of time instants over the time domain of interest (training + prediction)
nt_p = 441

# define target retained energy for the OpInf ROM
# target_ret_energy = 0.9996

# target reduced dimension
r = 22

# ranges for the regularization parameter pairs
B1 = np.logspace(-2., 5., num=20)
B2 = np.logspace(6., 12., num=20)

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