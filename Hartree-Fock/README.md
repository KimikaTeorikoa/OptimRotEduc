# Hartree-Fock: Algorithm coding and Step size Tunning

This folder contains the files for solving Hartree-Fock with Steepest gradient descent, Conjugate Gradients, AdaGrad, RMSProp and ADAM.

## Content

An example for the water molecule is provided inside the (H2O) folder, containing the following files:
- 0_HF-H2O.ipynb: Contain the code to solve Hartree-Fock using Steepest gradient descent, Conjugate Gradients, AdaGrad, RMSProp and ADAM with several step sizes (leraning rates) and write the results into csv files.
- 1_Plot-SD.ipynb: Read the csv file and plot the energy with respect to the iteration number when the Steepest Descent method is used.
- 2_Plot-CD.ipynb: Read the csv file and plot the energy with respect to the iteration number when the Conjugate Gradients method is used.
- 3_Plot-ADAGrad.ipynb: Read the csv file and plot the energy with respect to the iteration number when the Adaptive Gradient method is used.
- 4_Plot-RMSProp.ipynb: Read the csv file and plot the energy with respect to the iteration number when the Root Mean Square Propagation method is used.
- 5_Plot-ADAM.ipynb: Read the csv file and plot the energy with respect to the iteration number when the Adaptive Momentum method is used.
