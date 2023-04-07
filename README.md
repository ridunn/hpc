# HPC Project Code

This repositry contains the code produced for the High Performance Computing project looking at Parallel Molecular Dynamics with Short Ranged Potentials.

Molecular dynamics models use the Lennard-Jones potential, and use Verlet integration.

## Serial Code

The serial code is contained in the notebook Molecules_better.ipynb, where we will indicate where variables can be updated.

## Parallel Full Model

The full parallel code used is contained in the folder MPI_SETUP. This consists of three files; Compute_Forces.py, MPI_Spider.py, and Send_Positons.py.

The code is run from terminal using:

mpiexec -n python MPI_Spyder.py

where n dictates the number of cores used.

## Parallel Lattice Model

## Video Rendering
