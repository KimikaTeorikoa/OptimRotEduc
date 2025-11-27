# %% [markdown]
# # Natural Orbital Functional Theory: Dynamic (Scheduled) step size

# %% [markdown]
# Import libreries

# %%
import pynof
from scipy.linalg import eigh
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
#Molecules Definition
#Times given in Seconds, gotten on a Linux PC (12th Gen Intel(R) Core(TM) i9-12900K)
#with export OMP_NUM_THREADS=8
#Time for BF3: 521.1482 s
#Time for HNO3: 710.2246 s
#Time for O3: 109.8807 s

bf3 = pynof.molecule("""
0 1
 F     4.007177     0.908823     0.000000
 F     1.724922     0.908844     0.000000
 F     2.866001    -1.067658    -0.000000
 B     2.866001     0.249991    -0.000000
""")


hno3 = pynof.molecule("""
 0 1
    N       -0.151833    0.032380   -0.000040
    O       -1.021558   -0.782138    0.000011
    O        1.148151   -0.517360    0.000013
    O       -0.208610    1.237710    0.000014
    H        1.718969    0.267641   -0.000028
""")

o3 = pynof.molecule("""
 0 1
 O     1.068900     0.000000     0.215300
 O    -0.024957     0.000000    -0.445578
 O    -1.108044     0.000000     0.232808
 """)
# Dictionary to facilitate molecule selection with 
# output file name

molecules = {
    "BF3": bf3,
    "HNO3": hno3,
    "O3": o3,
}


# %% [markdown]
# Run calculations with a dynamyc alpha value

# %%
times = {}
molecule_list = ["BF3","HNO3","O3"]
alpha_values = [0.002, 0.02, 0.04, 0.08]   

t_st=t.time()
for molecule in molecule_list:
    st=t.time()

    #Define calulation
    mol = molecules[molecule]
    # Select basis set
    basis = "cc-pvdz"
    p = pynof.param(mol,basis)
    p.maxit = 30
    p.orb_method="ADAM"

    # File name to store the data
    filename = f"{molecule}_{"NOF"}.csv"

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["alpha", "iteration", "energy"])  # Header

        for alpha in alpha_values:
            p.alpha = alpha
            # Select molecule
            _, energy_data = calc_nof_orbrot(mol, p)
            for i_ext, energy in energy_data:
                writer.writerow([alpha, i_ext, energy])

        # ADAM algorithm including dynamic modification, ADAM2
        p.orb_method="ADAM2"
        p.alpha = 0.02
        _, energy_data = calc_nof_orbrot(mol, p)
        for i_ext, energy in energy_data:
            writer.writerow([0.0, i_ext, energy])
        et=t.time()
        times[molecule]=et-st
t_et=t.time()

for molecule in molecule_list:
     print(f"Time for {molecule}: {times[molecule]:.4f} s")

print(f"Total Time:",t_et-t_st)


