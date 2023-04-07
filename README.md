# HPC Project Code

This repositry contains the code produced for the High Performance Computing project looking at Parallel Molecular Dynamics with Short Ranged Potentials.

Molecular dynamics models use the Lennard-Jones potential, and use Verlet integration.

## Serial Code

The serial code is contained in the notebook Serial_Code.ipynb. This produces an animation and plots of the kinetic, potential and total energy.

## Parallel Full Model

The full parallel code used is contained in the folder Parallel_Code. Instructions to run this code are contained in the folder.

## Benchmarks

The benchmarks.py file just produces outputs based on the benchmarks obtained in the parallel full model. The output is five plots.

## Parallel Lattice Model

The parallel lattice model is comtained in the Lattice Method folder.

You must first run precompute.py to make the local file of pre-computed forces.

Them run Lattice_Spyder_MPI.py, following the same steps as used to run the Parallel Full model.
