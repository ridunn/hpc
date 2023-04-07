# Running the parallel code.

Place all 4 files in the same directory.

The MPI_Spyder.py and MPI_Spyder_Benchmarking.py files are very similar, execpt that MPI_Spyder.py generates an mp4 file showing the evolution of the particles and the Kinetic energy whereas the benchmarking file produces a text file of how long it took to run on each core and the kinetic energy. These will be placed in the current directory.

To launch using MPI, run the following command in a terminal.

mpiexec -n python MPI_Spyder.py 

This will launch the code that generates the mp4 file, where n denotes the number of cores(prime numbers don't work). 

mpiexec -n python MPI_Spyder_Benchmarking.py 

to run the benchmarking file.

The following packages are used in the code.

- datetime (for benchmarking)
- numpy
- mpi4py
- matplotlib
- ffmpeg (for animation)
