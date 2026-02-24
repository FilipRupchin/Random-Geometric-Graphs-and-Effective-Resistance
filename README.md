# Random Geometric Graphs and Effective Resistance

Repository for the study of asymptotic limits of effective resistances on Random Geometric Graphs (RGGs) with periodic boundary conditions.

## Overview

This project investigates the thermodynamic limits of effective resistance on RGGs as the number of nodes $n \to \infty$, with a focus on two connectivity regimes: the **supercritical regime**, in which the mean degree $k$ exceeds the percolation threshold (approximately 4.52, per Quintanilla et al. (2000) and Penrose (2003)), and the **connected regime**, in which $k$ scales with $ln(n)$ to guarantee full connectivity almost surely per Penrose (2003). 

The central aim is to characterise how effective resistance between vertices in the Largest Connected Component depends on geometric and structural quantities in each regime. We hope to re-open the study of effective resistances on thermodynamic limits of RGGs through the following numerical results presented in the jupyter notebooks in this repository. 

We use Lasso Regression to build models to predict effective resistance between nodes. In order to justify some of the features chosen to model effective resistance we reference previous work by Cserti (1999) and von Luxburg et al. (2014). We reference Cserti in the feature selection for our study of square lattice graphs with periodic boundary conditions. We then perturb the vertices of these graphs with normal distributions to simulate RGGs and the transition between rigidity and randomness. We reference von Luxburg et al. (2014) to include the sum of inverse degrees of nodes as a feature. In fact, in a very dense regime thermodynamic limit of RGGs, von Luxburg et al. show that the effective resistance is proportional to the the sum of inverse degrees of the nodes being measured. We hope to extend these past results numerically to the supercritical and connected regimes of RGGs.

## Contents

### `RGG_Library`
A custom built Python package for generating RGGs, with support for configurable system size, base space, mean degree, connectivity regimes, random seeds, and lattice structures, along with utilities for sampling commute times, computing graph statistics, and visualizing RGGs. More information in the RGG_Library README file.

## References

- J. Quintanilla, S. Torquato, and R. M. Ziff, *"Efficient measurement of the percolation threshold for fully penetrable discs,"* J. Phys. A: Math. Gen. **33**, L399–L407 (2000).
- M. Penrose, *Random Geometric Graphs*, Oxford University Press (2003).
- J. Cserti, *"Application of the lattice Green's function for calculating the resistance of an infinite network of resistors,"* Am. J. Phys. **68**, 896–906 (1999).
- U. von Luxburg, A. Radl, and M. Hein, *"Hitting and commute times in large random neighborhood graphs,"* J. Mach. Learn. Res. **15**, 1751–1798 (2014).