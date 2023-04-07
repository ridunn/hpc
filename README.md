# HPC Project Code

This repositry contains the code produced for the High Performance Computing project looking at Parallel Molecular Dynamics with Short Ranged Potentials.

Molecular dynamics models use the Lennard-Jones potential, and use Verlet integration.

## Serial Code

The serial code is contained in the notebook Serial_Code.ipynb. This produces an animation and plots of the kinetic, potential and total energy.

## Parallel Full Model

The full parallel code used is contained in the folder Parallel_Code. How to run this code is explained in that folder.

## Benchmarks

The benchmarks.py file just produces outputs based on the benchmarks obtained in the parallel full model. The output is 5 plots.

## Parallel Lattice Model

Notice that to run you first need to run precompute.py to make the local file of pre computed forces and them you can move on to running Lattice_Spyder_MPI.py, this is run the same way as the Parallel Full model.
