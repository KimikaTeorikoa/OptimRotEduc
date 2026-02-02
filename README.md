# Teaching modern Optimization Algorithms within the SCF

This repository contains materials for teaching Steepest Gradient Descent, Conjugate Gradients, AdaGrad, RMSProp, and ADAM within the framework of Hartree–Fock and Natural Orbital Theory. The contents supplement the manuscript (DOI: pending), and together, are intended to be used as a module in a course on numerical methods for quantum chemistry.

## 💡 Installation

You need to have [(mini/ana)conda](https://www.anaconda.com/) installed. The rest of the required packages can be installed in an environment by executing the following lines in the command line:

```bash
# Created a conda environment. Here we name it pynof.
conda create -y -n pynof python=3.13
# Once created the environment we load it and install the required software
conda activate pynof

conda install psi4 -c conda-forge 
pip install numpy matplotlib scipy jupyter notebook pynof

# Clone the repo to get the notebooks and the code that must be modified
git clone https://github.com/felipelewyee/OptimRotEduc
```

## 📝 Usage

The file myCode/minimization.py contains a function called *orbopt_sd*, which can be used for doing orbital optimization with the steepest gradient descent algorithm. Students may use this algorithm as a model to code other gradient based minimization algorithms  (*orbopt_cg*, *orbopt_rmsprop*, *orbopt_adagrad*, and *orbopt_adam*) **required** to run the notebooks.

Once implemented, students can explore their perfomrance in Hartree-Fock and Natural Orbital Functional (NOF) theory. The content is organized in folders:

- Start with Hartree-Fock (HF folder), where you will a) code the optimization algorithms (in myCode/minimization.py  file) and explore the effect of the step size and b) compare the convergency of the algorithms at their best step size.
- Follow with Natural Orbital Functional (NOF folder) theory, where you can a) apply the optimization algorithms in a more complex landscape that includes electron correlation, and b) introduce the concept of a dynamic (scheduled) step size for improving convergency.

## Ackownledgements

The content is based on and makes use of [PyNOF](https://github.com/felipelewyee/PyNOF) (based on the Donostia Natural Orbital Functional Software, [DoNOF](https://github.com/DoNOF/DoNOFsw)) and [Psi4NumPy](https://github.com/psi4/psi4numpy) (based on [Psi4](https://github.com/psi4/psi4)) in a Jupyter Notebook workspace.

## About

These module was created by Ph.D. Juan Felipe Huan Lew-Yee, Prof. Xabier Lopez, Prof. Elixabete Rezabal, Prof. Mario Piris, and Prof. Jose M. Mercero for teaching in courses at the Chemistry Faculty of the **Euskal Herriko Unibertsitatea (EHU)** in the Basque Country.
