# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 12:49:16 2023

@author: BNicolas Cassia
"""
import numpy as np

mat = np.load("pre_computed_forces.npy")

def force(v):
    x = v[0]
    y = v[1]
    f = mat[int(abs(x)*(10**4)),int(abs(y)*(10**4))]
    x = f[0]*np.sign(x)
    y = f[1]*np.sign(y)
    return np.array([x,y])

def Compute_Local_Forces(pos,epsilon,BoxSize,DIM,N,Rcutoff,sigma):
    Sij = np.zeros(DIM)
    acc = np.zeros([N,DIM])
    for i in range(N-1):
        for j in range(i+1,N):
            Sij = (pos[i,:]-pos[j,:])
            f = force(Sij)
            acc[i,:] = acc[i,:]+f
            acc[j,:] = acc[j,:]-f#
    return acc

def Compute_Non_Local_Forces(acc,pos,CORNERS,epsilon,BoxSize,DIM,N,Rcutoff,sigma,x_range,y_range,ID):
    upper_x = x_range[1]
    lower_x = x_range[0]
    upper_y = y_range[1]
    lower_y = y_range[0]
    LEFT_INDEX = np.where(pos[:,0] <= lower_x + Rcutoff)
    RIGHT_INDEX = np.where(pos[:,0] >= upper_x - Rcutoff)
    UPPER_INDEX = np.where(pos[:,1] >= upper_y - Rcutoff)
    LOWER_INDEX = np.where(pos[:,1] <= lower_y + Rcutoff)
    LOWER_LEFT_INDEX = np.where((pos[:,0]<= lower_x + Rcutoff) &(pos[:,1]<= lower_y + Rcutoff))
    LOWER_RIGHT_INDEX = np.where((pos[:,0] >= upper_x - Rcutoff) &(pos[:,1]<= lower_y + Rcutoff))
    UPPER_LEFT_INDEX = np.where((pos[:,0]<= lower_x + Rcutoff) &(pos[:,1]>= upper_y - Rcutoff))
    UPPER_RIGHT_INDEX = np.where((pos[:,0] >= upper_x - Rcutoff) &(pos[:,1]>= upper_y - Rcutoff))
    INDEX = [LEFT_INDEX,RIGHT_INDEX,LOWER_INDEX,UPPER_INDEX,LOWER_LEFT_INDEX,LOWER_RIGHT_INDEX,UPPER_LEFT_INDEX,UPPER_RIGHT_INDEX]
    #print(INDEX[0])
    #print(INDEX.shape)
    for i in range(len(CORNERS)):
        for j in range(len(CORNERS[i][0])):
            for k in range(len(CORNERS[i][1])):
                Sjk = CORNERS[i][0][j]-CORNERS[i][1][k]
            
                for l in range(DIM):
                    if (np.abs(Sjk[l])>0.5):
                        Sjk[l] = Sjk[l]-np.copysign(1,Sjk[l])
                
                Rjk = BoxSize*Sjk
                
                Rsqjk = np.dot(Rjk,Rjk)
                
                if (Rsqjk<Rcutoff**2):
                    r2 = 1/Rsqjk
                    r6 = r2**3
                    r12 = r6**2
                    dphi = epsilon*24*r2*(2*sigma**12*r12-sigma**6*r6)
                    
                    acc[INDEX[i][0][j],:] = acc[INDEX[i][0][j],:]+dphi*Sjk 
    
    return acc
        
