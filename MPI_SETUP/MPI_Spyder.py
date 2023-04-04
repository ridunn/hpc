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
print(mod_x,mod_y)
#Createm id from core rank
x_pos = rank%mod_x
y_pos = int((rank-rank%mod_x)/mod_x)





ID = np.array([x_pos,y_pos])
print("My rank is {} and ID is {}".format(rank,ID))

#Constants
DIM = 2
sigma = 0.06
nx = 15
N_particles = nx**2
np.random.seed(0)
d_perfect = 2**(1/6)*sigma
epsilon = 10
BoxSize = 1
Rcutoff = 3*sigma

#Initialize positions on cores
x_range = np.array([ID[0]/mod_x,(ID[0]+1)/mod_x])
y_range = np.array([ID[1]/mod_y,(ID[1]+1)/mod_y])

#core_dist = BoxSize/mod
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
        pos[i+nx*j] = [x_init[i]*BoxSize,y_init[j]*BoxSize]#+0.001*np.random.normal(0,1,size=2)

print("\n -------{}".format(ID))
#print(len(pos))
Nsteps = 1000
dt = 1/2000
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


def rank_from_ID(ID):
    return int(mod_x*ID[1]+ID[0])

def update_core_pos_2(pos,vel,acc):
    pos = pos%BoxSize
    #pos_2 = pos
    #vel_2 = vel
    #acc_2 = acc
    INDEX = np.any([pos[:,0]<lower_x,pos[:,0]>upper_x,pos[:,1]<lower_y,pos[:,1]>upper_y],axis=0)
    #print(INDEX,"\n ")
    INDEX = np.where(INDEX)
    pos_left = pos[INDEX]
    vel_left = vel[INDEX]
    acc_left = acc[INDEX]
    #print(pos%BoxSize,"pos")
    #print(pos_left,"left")
    pos = np.delete(pos,INDEX,axis=0)
    vel = np.delete(vel,INDEX,axis=0)
    acc = np.delete(acc,INDEX,axis=0)
    #print(pos,ID,"new\n")
    #print(mod)
    pos_dict = {(i,j): [] for j in range(int(mod_y)) for i in range(int(mod_x))}
    #for k in pos_dict.keys():
        
        #print(rank_from_ID(k))
    #print(pos_dict.keys())
    for i in range(len(pos_left)):
        x,y = int(np.floor(pos_left[i][0]*mod_x)),int(np.floor(pos_left[i][1]*mod_y))
        if x == 2:
            print(pos_left[i])
        if (x,y) in pos_dict.keys():
            pos_dict[(x,y)].append([pos_left[i],vel_left[i],acc_left[i]])
        else:
            pos_dict[(x,y)] = [[pos_left[i],vel_left[i],acc_left[i]]]
   # print(pos_dict.keys())
    #print(pos_dict)
    for l in pos_dict.keys():
        if rank_from_ID(l) == 8:
            print(pos_dict)
        h = len(pos_dict[l])
        #print(np.any(ID!=l),ID,l)
        #print(h)
        if np.any(ID!=l):
            if h != 0:
                #print(pos_dict[l],"\n")#
                #print(l,rank_from_ID(l),mod_x)
                pos_dict[l] = np.concatenate(pos_dict[l],axis=0).reshape(h,6)
                #print(pos_dict[l][:,0:2],"ID_{}_sent".format(l))
                comm.isend(pos_dict[l],dest = rank_from_ID(l),tag = rank_from_ID(l))
                #print(pos_dict[l],"sent_{}_to_rank_{}_from_rank_{}".format(l,rank_from_ID(l),rank))
            else:
                #print(l,rank_from_ID(l))
                comm.isend(pos_dict[l],dest = rank_from_ID(l),tag = rank_from_ID(l))
                #print(pos_dict[l],"sent_{}_to_rank_{}_from_rank_{}".format(l,rank_from_ID(l),rank))
                
        else:
            continue
        comm.Barrier()
            
        #comm.isend(pos_dict[1],dest = rank_from_ID(l),tag = rank_from_ID[l]+)
        #comm.isend(pos_dict[2],dest = rank_from_ID(l),tag = rank_from_ID[l])
    
    for h in pos_dict.keys():
        if np.any(ID!=h):
           # print(h,ID)
           
            pos_new = comm.recv(source=rank_from_ID(h),tag=rank_from_ID(ID))
            #pos_new = req.Wait()
            
           # print(pos_new)
            if len(pos_new) != 0:
                if len(pos)!=0:
                #print(pos_new[:,0:2],"ID_{}_rec".format(ID))
                    pos = np.concatenate([pos,pos_new[:,0:2]])
                    vel = np.concatenate([vel,pos_new[:,2:4]])
                    acc = np.concatenate([acc,pos_new[:,4:6]])
                else:
                    pos = pos_new[:,0:2]
                    vel = pos_new[:,2:4]
                    acc = pos_new[:,4:6]
                #print(pos,"After Concat_{}".format(ID))
                
        else:
            continue
    comm.barrier()
    #print(pos_2_e)
    #print(pos_2_e)
    #print(pos_new)
    
    
    #print(pos,"ID_{}".format(ID))
    
        #    print(pos_new," recieved_{}".format(h))
         #   print(pos_new_2)
           # if pos_new != []:
          #      pos = np.concatenate([pos,pos_new[:,0:2]])
             #   vel = np.concatenate([vel,pos_new[:,2:4]])
            #    acc = np.concatenate([acc,pos_new[:,4:6]])
        
    #print(pos,acc,vel == update_core_pos(pos_2,vel_2,acc_2))
    return pos,vel,acc
    #omm.isend(LOWER,dest=rank_from_ID((ID[0],(ID[1]-1)%mod)),tag=3)
                   
    
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
        
        #print(update_core_pos(pos))
        vel = vel +1/2*dt*acc
        acc = Compute_Forces.Compute_Local_Forces(pos, epsilon, BoxSize, DIM, N, Rcutoff, sigma)
        CORNERS = Send_Positions.send_positions_near_sides(comm, pos, ID, x_range, y_range, Rcutoff, mod_x,mod_y)
        acc = Compute_Forces.Compute_Non_Local_Forces(acc,pos,CORNERS,epsilon,BoxSize,DIM,N,Rcutoff,sigma,x_range,y_range,ID)
        vel = vel + 1/2*dt*acc
        
        pos = (pos + dt*vel + 0.5*dt**2*acc)
        pos = pos%BoxSize
       # print(update_core_pos_2(pos,vel,acc)[0],"new_t_{}".format(ID))
        #print("\n --------- \n")
        #rint(update_core_pos(pos,vel,acc)[0],"original_{}".format(ID))
        #print("\n --------- \n")
        pos_val = update_core_pos_2(pos,vel,acc)
        
        pos = pos_val[0]
        
        all_core_pos = comm.gather(pos,root=0)
        if rank == 0: 
            all_core_pos = np.concatenate(all_core_pos)
            img = ax.scatter(all_core_pos[:,0],all_core_pos[:,1],color='b',linewidth=7)
            title = ax.text(0.15*BoxSize,1.05*BoxSize,"N = {} Radius = {} t = {}".format(N_particles,sigma/2,'%s' % float(str('%s' % float('%.2g' % float(k*dt)))[:4])))
            ims.append([img,title])
        vel = pos_val[1]
       
        #print(pos[0][0])
        acc = pos_val[2]
        N = len(pos)
        E[k+1]=sum([sum(vel[i,:]**2) for i in range(N)])
        
    #ims = comm.bcast(ims,root=0)
    if rank ==0:
        pass
       # plt.title("N = {} sigma = {}".format(N_particles,sigma))
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.rcParams["animation.html"]= 'html5'
        ani = animation.ArtistAnimation(fig,ims,interval = 40,blit=True)
        ani.save("Test_ANIS/{}_new_modulo_{}_6_cores.mp4".format(N_particles,nr_proc))
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
    #plt.savefig("Test_ANIS/{}_ENERGY_Test_sigma_smaller_symmetry_larger_dt_random_larger_cutoff.png".format(N_particles))
    plt.close()
else:
    pass
#print("UPPER RIGHT",a[1][0][-1],"\n FROM UPPER RIGHT",a[1][1][-1],"\n RANK",ID)
MPI.Finalize()
