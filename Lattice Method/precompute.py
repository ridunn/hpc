#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:32:29 2023

@author: nicolascassia
"""
import numpy as np

DIM = 2
sigma = 0.06
nx = 15
N_particles = nx**2
np.random.seed(0)
d_perfect = 2**(1/6)*sigma
epsilon = 10
BoxSize = 1
Rcutoff = 2.5*sigma

dim = 4
def force(Sij):
    dphi = 0
    for l in range(DIM):
        if (np.abs(Sij[l])>0.5):
            Sij[l] = Sij[l]-np.copysign(1,Sij[l])
    Rij = BoxSize*Sij
    Rsqij = np.dot(Rij,Rij)
    if (Rsqij<Rcutoff**2):
        r2 = 1/Rsqij
        r6 = r2**3
        r12 = r6**2
        dphi = epsilon*24*r2*(2*sigma**12*r12-sigma**6*r6)
    return dphi*Sij



mat = np.zeros((10**dim,10**dim,2))
for i in range(10**dim):
    for j in range(i+1,10**dim):
        v = np.array([i/(10**dim),j/(10**dim)])
        mat[i][j] = force(v)
        mat[j][i] = -mat[i][j]
np.save("pre_computed_forces",mat)
        