# OptimRotEduc: Optimization Algorithms for Orbital Rotations in the Self Consistent Field for Education

This repo contains the files for teaching Steepest Gradient Descent, Conjugate Gradients, AdaGrad, RMSProp and ADAM in the context of Hartree-Fock and Natural Orbital Theory. The content is intended to be used as a module in a course on numerical methods for quantum chemistry. 

## 💡 Installation

```bash
conda create -y -n pynof python=3.12
conda install psi4 -c conda-forge 
pip install numpy matplotlib scipy jupyter notebook pynof
```

Remember to activate the environment for usage each time
```bash
conda activate pynof
```

## 📝 Usage

You can find the notebooks divided by folder.

Start with Hartree-Fock, where you can explore a) the effect of the step size on the different optimization algorithms, and b) compare the convergency of the algorithms at their best step size.

Follow with Natural Orbital Functional theory, where you can a) apply the optimization algorithms in a more complex landscape that includes electron correlation, and b) introduce the concept of a dynamic (scheduled) step size for improving convergency.

## Ackownledgements

The content is based on and makes use of PyNOF (based on the Donostia Natural Orbital Functional Software) and Psi4NumPy (based on Psi4) in a Jupyter Notebook workspace.
