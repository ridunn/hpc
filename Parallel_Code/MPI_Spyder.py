# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 12:08:02 2023

@author: Brian Hennessy
"""

#Import packages
import Send_Positions
import Compute_Forces
import datetime
now = datetime.datetime.now()
import numpy as np
import mpi4py.rc
mpi4py.rc.initialize = False  
from mpi4py import MPI
import matplotlib.pyplot as plt

from matplotlib import animation


def factor_int(n):
    val = np.ceil(np.sqrt(n))
    val2 = int(n/val)
    while val2 * val != float(n):
        val -= 1
        val2 = int(n/val)
    return val, val2


MPI.Init()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nr_proc = comm.Get_size()
mod = factor_int(nr_proc)

mod_x = max(mod)
mod_y = int(min(mod))
#Createm id from core rank
x_pos = rank%mod_x
y_pos = int((rank-rank%mod_x)/mod_x)



ID = np.array([x_pos,y_pos])

#Constants
DIM = 2
sigma = 0.03
nx = 32
N_particles = nx**2
np.random.seed(0)
d_perfect = 2**(1/6)*sigma
epsilon = 10
BoxSize = 1
Rcutoff = 2.5*sigma

#Initialize positions on cores
x_range = np.array([ID[0]/mod_x,(ID[0]+1)/mod_x])
y_range = np.array([ID[1]/mod_y,(ID[1]+1)/mod_y])


dist_init = np.linspace(0,0.9,nx)
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

Nsteps = 100
dt = 1/10000


def rank_from_ID(ID):
    return int(mod_x*ID[1]+ID[0])

def update_core_pos_2(pos,vel,acc):
    pos = pos%BoxSize
    
    INDEX = np.any([pos[:,0]<lower_x,pos[:,0]>upper_x,pos[:,1]<lower_y,pos[:,1]>upper_y],axis=0)
    
    INDEX = np.where(INDEX)
    pos_left = pos[INDEX]
    vel_left = vel[INDEX]
    acc_left = acc[INDEX]
   
    pos = np.delete(pos,INDEX,axis=0)
    vel = np.delete(vel,INDEX,axis=0)
    acc = np.delete(acc,INDEX,axis=0)
    
    pos_dict = {(i,j): [] for j in range(int(mod_y)) for i in range(int(mod_x))}
    
    for i in range(len(pos_left)):
        x,y = int(np.floor(pos_left[i][0]*mod_x)),int(np.floor(pos_left[i][1]*mod_y))
        
        if (x,y) in pos_dict.keys():
            pos_dict[(x,y)].append([pos_left[i],vel_left[i],acc_left[i]])
        else:
            pos_dict[(x,y)] = [[pos_left[i],vel_left[i],acc_left[i]]]

    for l in pos_dict.keys():
        
        h = len(pos_dict[l])
        
        if np.any(ID!=l):
            if h != 0:
                pos_dict[l] = np.concatenate(pos_dict[l],axis=0).reshape(h,6)
                comm.isend(pos_dict[l],dest = rank_from_ID(l),tag = rank_from_ID(l))
            else:
                comm.isend(pos_dict[l],dest = rank_from_ID(l),tag = rank_from_ID(l))  
        else:
            continue
            
    
    for h in pos_dict.keys():
        if np.any(ID!=h):
            pos_new = comm.recv(source=rank_from_ID(h),tag=rank_from_ID(ID))
            if len(pos_new) != 0:
                if len(pos)!=0:
                    pos = np.concatenate([pos,pos_new[:,0:2]])
                    vel = np.concatenate([vel,pos_new[:,2:4]])
                    acc = np.concatenate([acc,pos_new[:,4:6]])
                else:
                    pos = pos_new[:,0:2]
                    vel = pos_new[:,2:4]
                    acc = pos_new[:,4:6]
                
                
        else:
            continue
    comm.barrier()
    return pos,vel,acc
    
                   
    
def main(pos,Nsteps,dt,epsilon,BoxSize,DIM):
    N = len(pos)
    ims = []
    vel = (np.zeros([N,DIM]))
    acc = (np.zeros([N,DIM]))
    fig = plt.figure(figsize = (4,4), dpi=250)

    E = np.zeros(Nsteps+1)
    ax = plt.axes()
    E[0] = sum([sum(vel[i,:]**2) for i in range(N)])
    for k in range(Nsteps):

        vel = vel +1/2*dt*acc
        
        pos = (pos + dt*vel)
        pos = pos%BoxSize
        pos_val = update_core_pos_2(pos,vel,acc)
        
        pos = pos_val[0]
        vel = pos_val[1]
        acc = pos_val[2]
        N = len(pos)
        
        acc = Compute_Forces.Compute_Local_Forces(pos, epsilon, BoxSize, DIM, N, Rcutoff, sigma)
        CORNERS = Send_Positions.send_positions_near_sides(comm, pos, ID, x_range, y_range, Rcutoff, mod_x,mod_y)
        acc = Compute_Forces.Compute_Non_Local_Forces(acc,pos,CORNERS,epsilon,BoxSize,DIM,N,Rcutoff,sigma,x_range,y_range,ID)
        vel = vel + 1/2*dt*acc
        
        #Collective Communication just for outputs.
        all_core_pos = comm.gather(pos,root=0)
        if rank == 0: 
            all_core_pos = np.concatenate(all_core_pos)
            img = ax.scatter(all_core_pos[:,0],all_core_pos[:,1],color='b',linewidth=1.5)
            title = ax.text(0.15*BoxSize,1.05*BoxSize,"N = {} Radius = {} t = {}".format(N_particles,sigma/2,'%s' % float(str('%s' % float('%.2g' % float(k*dt)))[:4])))
            ims.append([img,title])
            
        E[k+1]=sum([sum(vel[i,:]**2) for i in range(N)])
        
    
    if rank == 0:
        pass
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.rcParams["animation.html"]= 'html5'
        ani = animation.ArtistAnimation(fig,ims,interval = 20,blit=True)
        ani.save("Animation_{}_particles_{}_cores.mp4".format(N_particles,nr_proc))
    return acc,E

a = main(pos,Nsteps,dt,epsilon,BoxSize,DIM)

Ene = np.array(comm.gather(a[1],root=1))


if rank ==1: 
    fig2 = plt.figure(figsize = (8,4), dpi=150)
    x = np.linspace(0,dt*Nsteps,Nsteps+1)
    plt.plot(x,1/2*Ene.sum(axis=0)/nx**2)
    plt.xlabel("Time")
    plt.ylabel("Kinetic Energy")
    plt.title("Average Kinetic Energy against time.")
    plt.savefig("Average_Kinetic_Energy_{}_Particles_{}_cores.png".format(N_particles,nr_proc))
    plt.close()
else:
    pass

MPI.Finalize()
