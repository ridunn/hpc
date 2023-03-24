# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 12:08:02 2023

@author: Brian Hennessy
"""
#Import packages
import Send_Positions

#send_positions_near_sides()
import Compute_Forces
import numpy as np
import mpi4py.rc
mpi4py.rc.initialize = False  
from mpi4py import MPI
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
MPI.Init()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nr_proc = comm.Get_size()
mod = np.sqrt(nr_proc)

#Create id from core rank
y_pos = rank%mod
x_pos = int((rank-rank%mod)/mod)

ID = np.array([x_pos,y_pos])

#Constants
DIM = 2
sigma = 0.05
nx = 10

d_perfect = 2**(1/6)*sigma
epsilon = 10
BoxSize = 1
Rcutoff = 2.5*sigma

#Initialize positions on cores
x_range = np.array([ID[1]/mod,(ID[1]+1)/mod])
y_range = np.array([ID[0]/mod,(ID[0]+1)/mod])

core_dist = BoxSize/mod
dist_init = np.linspace(0,0.8,nx)
upper_x = x_range[1]
lower_x = x_range[0]
x_init = dist_init[np.where(upper_x >= dist_init)]
x_init = x_init[np.where(lower_x < x_init)]

upper_y = y_range[1]
lower_y = y_range[0]
y_init = dist_init[np.where(upper_y >= dist_init)]
y_init = y_init[np.where(lower_y < y_init)]
nx, ny = len(x_init),len(y_init)
N = nx*ny
pos = np.zeros([nx*ny,DIM])
for i in range(nx):
    for j in range(ny):
        pos[i+nx*j] = [x_init[i]*BoxSize,y_init[j]*BoxSize]
        
Nsteps = 1000
dt = 1/1000

def update_core_pos(pos):
    all_pos = comm.gather(pos,root=0)
    
    all_core_pos = comm.bcast(all_pos,root=0)
    
    all_core_pos = np.concatenate(all_core_pos)
    if rank == 0:
        im = [plt.scatter(all_core_pos[:,0],all_core_pos[:,1],color='b')]
    else:
        im = None
    #print("\n",all_core_pos,rank)
    pos_x = all_core_pos[np.where((all_core_pos[:,0]>=lower_x)&(all_core_pos[:,0]<=upper_x))]
    pos = pos_x[np.where((pos_x[:,1]>=lower_y)&(pos_x[:,1]<=upper_y))]
    return pos,im

def main(pos,Nsteps,dt,epsilon,BoxSize,DIM):
    N = len(pos)
    ims = [None for i in range(Nsteps)]
    vel = (np.zeros([N,DIM]))
    acc = (np.zeros([N,DIM]))
    E = np.zeros(Nsteps+1)
    E[0] = sum([sum(vel[i,:]**2) for i in range(N)])
    for k in range(Nsteps):
        pos = (pos + dt*vel + 0.5*dt**2*acc)
        pos_val = update_core_pos(pos)
        ims[k] = pos_val[1]
        pos = pos_val[0]
        N = len(pos)
        vel = (np.zeros([N,DIM]))
        acc = (np.zeros([N,DIM]))
        #print(update_core_pos(pos))
        vel = vel +1/2*dt*acc
        acc = Compute_Forces.Compute_Local_Forces(pos, epsilon, BoxSize, DIM, N, Rcutoff, sigma)
        CORNERS = Send_Positions.send_positions_near_sides(comm, pos, ID, x_range, y_range, Rcutoff, mod)
        acc = Compute_Forces.Compute_Non_Local_Forces(acc,pos,CORNERS,epsilon,BoxSize,DIM,N,Rcutoff,sigma,x_range,y_range,ID)
        vel = vel + 1/2*dt*acc
        
        pos = pos%BoxSize
        
        E[k+1] = 1/2*sum([sum(vel[i,:]**2) for i in range(N)])
    ims = comm.bcast(ims,root=0)
    return acc,E,ims

a = main(pos,Nsteps,dt,epsilon,BoxSize,DIM)
#print("This is process {} and local force I have is {}.".format(rank, a[0]))

#if rank == 0:
Ene = np.array(comm.gather(a[1],root=1))
#print(Ene.sum(axis=0))



if rank ==0:
    #print(rank)
    fig = plt.figure(figsize = (4,4), dpi=150)
    ims = a[2]
    print(ims)
    #plt.xlim([0,1*BoxSize])
   # plt.ylim([0,1*BoxSize])
    plt.title("Molecular Dynamics")
    plt.rcParams["animation.html"]= 'html5'
    ani = animation.ArtistAnimation(fig,ims,interval = 40)
    ani.save("Test_ani.mp4")
else:
    pass


if rank ==1: 
    plt.plot(Ene.sum(axis=0)/100)
    plt.savefig("Test.png")
    plt.close()
else:
    pass
#print("UPPER RIGHT",a[1][0][-1],"\n FROM UPPER RIGHT",a[1][1][-1],"\n RANK",ID)
MPI.Finalize()
