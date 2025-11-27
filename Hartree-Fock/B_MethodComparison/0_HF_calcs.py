# %% [markdown]
# # Hartree-Fock: Comparison of optimization algorithms at Best Learning Rate

# %% [markdown]
# Import libreries

# %%
import csv
import pynof
import numpy as np
from scipy.linalg import eigh
import sys
import time as t
# Define energy.py path, in our case is in myCode dir, two directories above
sys.path.insert(1, '../../myCode/')
# read files with optimization functions
from energy import *

# %% [markdown]
#  Molecules definition

# %%
#Molecules Definition
#Times given in Seconds, gotten on a Linux PC (12th Gen Intel(R) Core(TM) i9-12900K)
#Time for BF3: 230.3235 s
#Time for CHCL3: 763.0696 s
#Time for ALOH: 684.0536 s
#Total Time: 1677.4466578960419

# Molecules coordinates
bf3 = pynof.molecule("""
0 1
 F     4.007177     0.908823     0.000000
 F     1.724922     0.908844     0.000000
 F     2.866001    -1.067658    -0.000000
 B     2.866001     0.249991    -0.000000 
""")

chcl3 = pynof.molecule("""
  0 1
Cl        -0.09620       -1.67890        0.13940
Cl        -1.40590        0.92280        0.13940
Cl         1.50220        0.75610        0.13930
C         -0.00010        0.00000       -0.41810
H         -0.00010       -0.00010       -1.51110
""")

aloh = pynof.molecule("""
0 1
Al     0.072995     0.126285    -0.072819
 O    -0.117348     0.068538     1.623683
 O     1.640841     0.345700    -0.695473
 O    -1.181058    -0.017712    -1.213874
 H     2.443051     0.441622    -0.181373
 H    -0.892422    -0.042102     2.172874
 H    -2.121611    -0.146659    -1.103014
""")

# Dictionary to facilitate molecule selection with 
# output file name

molecules = {
    "BF3": bf3,
    "CHCL3": chcl3,
    "ALOH": aloh,
}

# %% [markdown]
# We run the optimization using SD, CG, RMSProp, ADAGRAD and ADAM at the best leraning rate values for comparison. The optimization data is written into a csv files.

# %%
# Dictionary to store time
times  = {}

algorithms = [ 'SD', 'CG', 'RMSPROP', 'ADAGRAD', 'ADAM' ]
molecule_list = ["BF3","CHCL3","ALOH"]  

# Select basis set
basis = "cc-pvdz"

t_st = t.time()
for molecule in molecule_list:
   st=t.time()
   #Define system
   mol = molecules[molecule]
   p = pynof.param(mol,basis)
   p.maxit=60
   
   # File name to store the data
   filename = f"{molecule}.csv"

   with open(filename, mode='w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(["alg", "iteration", "energy"])  # Header

      for alg in algorithms:
         p.orb_method=alg
         if alg == 'SD':
            p.alpha = 0.02
         if alg == 'CG':
            p.alpha = 0.02
         elif alg == 'ADAGRAD':
            p.alpha = 0.08
         elif alg == 'RMSPROP':
            p.alpha = 0.002
         elif alg == 'ADAM':
            p.alpha = 0.02           
         _, energy_data = calc_hf_orbrot(mol, p)
         for i_ext, energy in energy_data:
               writer.writerow([alg, i_ext, energy])
   et = t.time()
   times[molecule]=et-st
t_et = t.time()

#Total start  time
t_et = t.time()

#Print times per molecule
for molecule in molecule_list:
    print(f"Time for {molecule}: {times[molecule]:.4f} s")

print(f"Total Time:",t_et-t_st)


