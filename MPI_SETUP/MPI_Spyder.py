# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 12:08:02 2023

@author: Brian Hennessy
"""

#Import packages
import Send_Positions
import Compute_Forces
import numpy as np
import mpi4py.rc
mpi4py.rc.initialize = False  
from mpi4py import MPI
import matplotlib.pyplot as plt
#from matplotlib.animation import PillowWriter
from matplotlib import animation
MPI.Init()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nr_proc = comm.Get_size()
mod = np.sqrt(nr_proc)

#Create id from core rank
x_pos = rank%mod
y_pos = int((rank-rank%mod)/mod)

ID = np.array([x_pos,y_pos])
#print("My rank is {} and ID is {}".format(rank,ID))

#Constants
DIM = 2
sigma = 0.015
nx = 30

d_perfect = 2**(1/6)*sigma
epsilon = 10
BoxSize = 1
Rcutoff = 2.5*sigma

#Initialize positions on cores
x_range = np.array([ID[0]/mod,(ID[0]+1)/mod])
y_range = np.array([ID[1]/mod,(ID[1]+1)/mod])

core_dist = BoxSize/mod
dist_init = np.linspace(0,0.8,nx)
upper_x = x_range[1]
lower_x = x_range[0]
x_init = dist_init[np.where(upper_x > dist_init)]
x_init = x_init[np.where(lower_x <= x_init)]

upper_y = y_range[1]
lower_y = y_range[0]
y_init = dist_init[np.where(upper_y > dist_init)]
y_init = y_init[np.where(lower_y <= y_init)]
nx, ny = len(x_init),len(y_init)
N = nx*ny
pos = np.zeros([nx*ny,DIM])
for i in range(nx):
    for j in range(ny):
        pos[i+nx*j] = [x_init[i]*BoxSize,y_init[j]*BoxSize]

#print(len(pos))
Nsteps = 40
dt = 1/1000
#print(len(pos))
def update_core_pos(pos,vel,acc):
    all_pos = comm.gather(pos,root=0)
    all_vel = comm.gather(vel,root=1)
    all_acc = comm.gather(acc,root=2)
    all_core_pos = comm.bcast(all_pos,root=0)
    all_core_pos = np.concatenate(all_core_pos)
    all_core_vel = comm.bcast(all_vel,root=1)
    all_core_vel = np.concatenate(all_core_vel)
    all_core_acc = comm.bcast(all_acc,root=2)
    all_core_acc = np.concatenate(all_core_acc)
    
    pos_x = all_core_pos[np.where((all_core_pos[:,0]>=lower_x)&(all_core_pos[:,0]<=upper_x))]
    vel_x = all_core_vel[np.where((all_core_pos[:,0]>=lower_x)&(all_core_pos[:,0]<=upper_x))]
    acc_x = all_core_acc[np.where((all_core_pos[:,0]>=lower_x)&(all_core_pos[:,0]<=upper_x))]
    pos = pos_x[np.where((pos_x[:,1]>=lower_y)&(pos_x[:,1]<=upper_y))]
    vel = vel_x[np.where((pos_x[:,1]>=lower_y)&(pos_x[:,1]<=upper_y))]
    acc = acc_x[np.where((pos_x[:,1]>=lower_y)&(pos_x[:,1]<=upper_y))]
    return pos,vel,acc

def main(pos,Nsteps,dt,epsilon,BoxSize,DIM):
    N = len(pos)
    ims = []
    vel = (np.zeros([N,DIM]))
    acc = (np.zeros([N,DIM]))
   # fig = plt.figure(figsize = (4,4), dpi=150)

    E = np.zeros(Nsteps+1)
    ax = plt.axes()
    E[0] = sum([sum(vel[i,:]**2) for i in range(N)])
    for k in range(Nsteps):
        pos = (pos + dt*vel + 0.5*dt**2*acc)
        #print(update_core_pos(pos))
        vel = vel +1/2*dt*acc
        acc = Compute_Forces.Compute_Local_Forces(pos, epsilon, BoxSize, DIM, N, Rcutoff, sigma)
        CORNERS = Send_Positions.send_positions_near_sides(comm, pos, ID, x_range, y_range, Rcutoff, mod)
        acc = Compute_Forces.Compute_Non_Local_Forces(acc,pos,CORNERS,epsilon,BoxSize,DIM,N,Rcutoff,sigma,x_range,y_range,ID)
        vel = vel + 1/2*dt*acc
        pos = pos%BoxSize
        pos_val = update_core_pos(pos,vel,acc)
        
        pos = pos_val[0]
        
        all_core_pos = comm.gather(pos,root=0)
        if rank == 0: 
            all_core_pos = np.concatenate(all_core_pos)
            img = ax.scatter(all_core_pos[:,0],all_core_pos[:,1],color='b')
            ims.append([img])
        vel = pos_val[1]
       
        #print(pos[0][0])
        acc = pos_val[2]
        N = len(pos)
        E[k+1]=sum([sum(vel[i,:]**2) for i in range(N)])
        
    #ims = comm.bcast(ims,root=0)
    if rank ==0:
        pass
        plt.title("Molecular Dynamics")
        plt.xlim([0,1])
        plt.ylim([0,1])
        #plt.rcParams["animation.html"]= 'html5'
        #ani = animation.ArtistAnimation(fig,ims,interval = 40,blit=True)
        #ani.save("900_particles.mp4")
    return acc,E

a = main(pos,Nsteps,dt,epsilon,BoxSize,DIM)
#print("This is process {} and local force I have is {}.".format(rank, a[0]))

#if rank == 0:
Ene = np.array(comm.gather(a[1],root=1))




if rank ==-0:

    
   # ims = a[2]
    #print(ims)
    #print(ims[-1][0])
    #ims[0][0]
    #plt.savefig("Last_img.png")
    
    #plt.show()
    plt.close()
else:
    pass


if rank ==1: 
    fig2 = plt.figure(figsize = (8,4), dpi=150)
    plt.plot(Ene.sum(axis=0)/nx**2)
    plt.savefig("Energy_78.png")
    plt.close()
else:
    pass
#print("UPPER RIGHT",a[1][0][-1],"\n FROM UPPER RIGHT",a[1][1][-1],"\n RANK",ID)
MPI.Finalize()
