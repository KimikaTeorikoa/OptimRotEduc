### Code structure and instructional workflow

This repository is organized to support an instructional module on numerical
optimization in quantum chemistry. The different Python files play distinct and
complementary roles:

- `energy.py`  
  Provides a fixed computational backend adapted from the original PyNOF
  implementation. This file contains the evaluation of the electronic energy,
  convergence criteria, and the optimization loop. It is not intended to be
  modified by students and ensures that all optimization algorithms are tested
  against the same energy functional and stopping conditions.

- `minimization.py`  
  This is the main working file for students. It contains a reference
  implementation of the steepest descent (SD) algorithm adapted to PyNOF, which
  students use as a starting point to implement additional optimization
  algorithms (e.g., conjugate gradients, RMSProp, Adam).

- `minimizationSolved.py`  
  Provides a complete reference solution with all required optimization
  algorithms already implemented. This file is intended for instructors and
  reviewers, allowing them to execute straightforward the accompanying Jupyter
  notebooks. Students are not expected to modify or use this file during the activity.


