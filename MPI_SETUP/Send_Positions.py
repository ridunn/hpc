# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 12:14:37 2023

@author: Brian Hennessy
"""
import numpy as np
def send_positions_near_sides(comm,pos,ID,x_range,y_range,Rcutoff,mod_x,mod_y):

    def rank_from_ID(ID):
        return int(mod_x*ID[1]+ID[0])
    
    upper_x = x_range[1]
    lower_x = x_range[0]
    upper_y = y_range[1]
    lower_y = y_range[0]
    #Left wall
    LEFT = pos[np.where(pos[:,0]<= lower_x + Rcutoff)]
    comm.isend(LEFT,dest=rank_from_ID(((ID[0]-1)%mod_x,ID[1])),tag=100)
    #FROM_RIGHT = comm.recv(LEFT,source=rank_from_ID((ID[0],(ID[1]-1)%mod)),tag=1)
    
    #Right wall
    RIGHT = pos[np.where(pos[:,0]>= upper_x - Rcutoff)]
    comm.isend(RIGHT,dest=rank_from_ID(((ID[0]+1)%mod_x,ID[1])),tag=200)

    #Lower wall
    LOWER = pos[np.where(pos[:,1] <= lower_y + Rcutoff)]
    comm.isend(LOWER,dest=rank_from_ID((ID[0],(ID[1]-1)%mod_y)),tag=300)
              
    #Upper wall
    UPPER = pos[np.where(pos[:,1] >= upper_y - Rcutoff)]
    comm.isend(UPPER,dest=rank_from_ID((ID[0],(ID[1]+1)%mod_y)),tag=400)
    
    #Lower left corner
    LOWER_LEFT = LEFT[np.where(LEFT[:,1]<= lower_y + Rcutoff)]
    comm.isend(LOWER_LEFT,dest=rank_from_ID(((ID[0]-1)%mod_x,(ID[1]-1)%mod_y)),tag=500)
    
    #Upper left corner
    UPPER_LEFT = LEFT[np.where(LEFT[:,1]>= upper_y - Rcutoff)]
    comm.isend(UPPER_LEFT,dest=rank_from_ID(((ID[0]-1)%mod_x,(ID[1]+1)%mod_y)),tag=600)
    
    #Lower right corner
    LOWER_RIGHT = RIGHT[np.where(RIGHT[:,1]<= lower_y + Rcutoff)]
    comm.isend(LOWER_RIGHT,dest=rank_from_ID(((ID[0]+1)%mod_x,(ID[1]-1)%mod_y)),tag=700)
    
    #Upper right corner
    UPPER_RIGHT = RIGHT[np.where(RIGHT[:,1]>= upper_y - Rcutoff)]
    comm.isend(UPPER_RIGHT,dest=rank_from_ID(((ID[0]+1)%mod_x,(ID[1]+1)%mod_y)),tag=800)
    #req.wait()

    
    FROM_LEFT = comm.recv(source=rank_from_ID(((ID[0]-1)%mod_x,ID[1])),tag=200)
    
    FROM_RIGHT = comm.recv(source=rank_from_ID(((ID[0]+1)%mod_x,ID[1])),tag=100)
    FROM_LOWER = comm.recv(source=rank_from_ID((ID[0],(ID[1]-1)%mod_y)),tag=400)
    FROM_UPPER = comm.recv(source=rank_from_ID((ID[0],(ID[1]+1)%mod_y)),tag=300)
    FROM_LOWER_LEFT = comm.recv(source=rank_from_ID(((ID[0]-1)%mod_x,(ID[1]-1)%mod_y)),tag=800)
    
    FROM_UPPER_LEFT = comm.recv(source=rank_from_ID(((ID[0]-1)%mod_x,(ID[1]+1)%mod_y)),tag=700)
    
    FROM_LOWER_RIGHT = comm.recv(source=rank_from_ID(((ID[0]+1)%mod_x,(ID[1]-1)%mod_y)),tag=600)
    FROM_UPPER_RIGHT = comm.recv(source=rank_from_ID(((ID[0]+1)%mod_x,(ID[1]+1)%mod_y)),tag=500)
                           

    
    return [[LEFT,FROM_LEFT],[RIGHT,FROM_RIGHT],[LOWER,FROM_LOWER],[UPPER,FROM_UPPER],
            [LOWER_LEFT,FROM_LOWER_LEFT],[LOWER_RIGHT,FROM_LOWER_RIGHT], 
            [UPPER_LEFT,FROM_UPPER_LEFT],[UPPER_RIGHT,FROM_UPPER_RIGHT]]
                                    
                                    
                                    