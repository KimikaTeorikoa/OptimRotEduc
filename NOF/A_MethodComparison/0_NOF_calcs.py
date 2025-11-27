# %% [markdown]
# # Natural Orbital Functionals: Method Comparison at 0.02 and 0.002 step size

# %% [markdown]
# Import libreries

# %%
from scipy.linalg import eigh
import pynof
import numpy as np
import csv
import sys
import time as t
# Define energy.py path, in our case is in myCode dir, two directories above
sys.path.insert(1, '../../myCode/')
# read files with optimization functions
from energy import *

# %% [markdown]
# Create molecule and choose basis set

# %%
# Molecules definition
#Time for CO2: 295.7181 s
#Time for CO: 49.8354 s
#Time for H2O: 35.7509 s
#Total Time: 381.30437541007996

co = pynof.molecule("""
0 1
  C      0.0       0.0         0.0
  O      0.0       0.0         1.12     
""")

co2 = pynof.molecule("""
0 1
C    0.0000    0.0000    0.0000   
O    1.1000    0.0000    0.0000   
O    -1.1000   -0.0000    0.0000 
""")

h2o = pynof.molecule("""
0 1
  O  0.0000   0.000   0.116
  H  0.0000   0.749  -0.453
  H  0.0000  -0.749  -0.453
""")

# Dictionary to facilitate molecule selection with 
# output file name

molecules = {
    "CO": co,
    "CO2": co2,
    "H2O": h2o,
}

# %% [markdown]
# Run NOF calculations with several optimization algorithms at 0.002 step size.

# %%
times = {} 
molecule_list = ["CO2","CO","H2O"]

alpha_list=[0.02,0.002]

t_st=t.time()
for molecule in molecule_list:
    st=t.time()
    for alpha in alpha_list:  
        mol = molecules[molecule]

        # Select basis set
        basis = "cc-pvdz"

        p = pynof.param(mol,basis)

        # Here we select the NOF functional.
        p.maxit = 45
        p.alpha = alpha

        algorithms = [ 'SD', 'RMSPROP', 'ADAGRAD', 'ADAM' ]

        # File name to store the data
        filename = f"{molecule}_{p.alpha}.csv"

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["alg", "iteration", "energy"])  # Header

            for alg in algorithms:
                p.orb_method=alg
                _, energy_data = calc_nof_orbrot(mol, p)
                for i_ext, energy in energy_data:
                    writer.writerow([alg, i_ext, energy])
    et=t.time()
    times[molecule] = et - st
t_et=t.time()  

for molecule in molecule_list:
     print(f"Time for {molecule}: {times[molecule]:.4f} s")

print(f"Total Time:",t_et-t_st)


