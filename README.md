# OptimRotEduc: Teaching Optimization Algorithms in SCF
This repo contains the files for teaching Steepest Gradient Descent, Conjugate Gradients, AdaGrad, RMSProp and ADAM in the context of Hartree-Fock and Natural Orbital Theory. The content is intended to be used as a module in a course on numerical methods for quantum chemistry. 

## 💡 Installation

You need to have ![anaconda](https://www.anaconda.com/) installed. The rest of the required packages can be installed in an environment by executing the following lines in the command line:

```bash
conda create -y -n pynof python
conda activate pynof
conda install psi4 -c conda-forge 
pip install numpy matplotlib scipy jupyter notebook pynof
```

Remember to activate the environment every time you work on the notebooks:
```bash
conda activate pynof
```

## 📝 Usage

You can find the notebooks divided by folder.

- Start with Hartree-Fock, where you can explore a) the effect of the step size on the different optimization algorithms, and b) compare the convergency of the algorithms at their best step size.

- Follow with Natural Orbital Functional theory, where you can a) apply the optimization algorithms in a more complex landscape that includes electron correlation, and b) introduce the concept of a dynamic (scheduled) step size for improving convergency.

## Ackownledgements

The content is based on and makes use of ![PyNOF](https://github.com/felipelewyee/PyNOF) (based on the Donostia Natural Orbital Functional Software, ![DoNOF](https://github.com/DoNOF/DoNOFsw)) and ![Psi4NumPy](https://github.com/psi4/psi4numpy) (based on ![Psi4](https://github.com/psi4/psi4)) in a Jupyter Notebook workspace.

## About

These exercises were created by Ph.D. Juan Felipe Huan Lew-Yee, Prof. Xabier Lopez, Prof. Elixabete Rezabal, Prof. Mario Piris, and Prof. Jose M. Mercero for teaching in courses at the Chemistry Faculty of the **Euskal Herriko Unibertsitatea (EHU)** in the Basque Country.
