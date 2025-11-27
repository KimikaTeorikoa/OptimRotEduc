# %% [markdown]
# # Hartree-Fock: Step Size

# %% [markdown]
# Import libraries

# %%
import csv
import pynof
# import energy.py library including optimization methods
import sys
import time as t
# Define energy.py path, in our case is in myCode dir, two directories above
sys.path.insert(1, '../../myCode/')
# read files with optimization functions
from energy import calc_hf_orbrot

# %% [markdown]
# Create molecule and choose basis set

# %%
#Molecules Definition
#Times given in Seconds, gotten on a Linux PC (12th Gen Intel(R) Core(TM) i9-12900K)
#with export OMP_NUM_THREADS=8
#Time for H2O: 135 
#Time for CO2: 1052
#Total Time: 1188

h2o = pynof.molecule("""
0 1
  O  0.0000   0.000   0.116
  H  0.0000   0.749  -0.453
  H  0.0000  -0.749  -0.453
""")

co2 = pynof.molecule("""
0 1
C    0.0000    0.0000    0.0000   
O    1.1000    0.0000    0.0000   
O    -1.1000   -0.0000    0.0000 
""")

molecules = {
    "CO2": co2,
    "H2O": h2o,
}

# %% [markdown]
# Minimize Hartree-Fock for each algorithm and each alpha value (step size) and store data in csv file.

# %%
times = {}
# Algorithms to be tested
algorithms = [ 'SD', 'CG', 'RMSPROP', 'ADAGRAD', 'ADAM' ]
# earning rate values
alpha_values = [0.002, 0.005, 0.02, 0.04, 0.08, 0.16]

molecule_list = ["H2O","CO2"]
#Basis set definition

basis = "cc-pvdz"
#Total start time
t_st = t.time()

for molecule in molecule_list:
    #Start time for molecule
    st = t.time()
 
    #Define system
    mol = molecules[molecule]
    p = pynof.param(mol,basis)
    p.maxit = 60

    for alg in algorithms:
        p.orb_method=alg
        filename = f"{molecule}_{alg}.csv"
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["alpha", "iteration", "energy"])  # Header

            for alpha in alpha_values:
                p.alpha = alpha
                #Run HF calc for each case
                _, energy_data = calc_hf_orbrot(mol, p)
                for i_ext, energy in energy_data:
                    writer.writerow([alpha, i_ext, energy])
    #End time for molecule
    et = t.time()
    times[molecule]=et-st

#Total start  time
t_et = t.time()

#Print times per molecule
for molecule in molecule_list:
    print(f"Time for {molecule}: {times[molecule]:.4f} s")

print(f"Total Time:",t_et-t_st)


