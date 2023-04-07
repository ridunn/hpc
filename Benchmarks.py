# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
x = np.array([1,4,6,8,9,10,12,14,15,16])
y = np.array([2106.34,180+40.55,120+54.76,120+17.95,120+6.85,120+10.15,60+47.51,60+58.49,60+33.55,60+26.81])
plt.figure(figsize=(6,4),dpi=250)
plt.plot(x,y,marker='o',label="1000 iterations")
plt.ylabel("Seconds")
plt.xlabel("Number of Cores")

plt.title("Time to Compute 1000 Iterations Against Number of Cores")
plt.xticks([1,4,6,8,9,10,12,14,15,16],labels=["1","4","6","8","9","10","12","14*","15*","16*"])
plt.legend()
#plt.xlim([4,16])
#plt.ylim([0,250])

speedup = y[0]/y

fig2 = plt.figure(figsize=(6,4),dpi=250)
plt.plot(x,speedup,marker='o',label="Normalised Against 1 Cores")
#plt.xlim([4,16])
plt.xlabel("Number of Cores")
plt.ylabel("Speedup")
plt.title("Speedup Against Number of Cores")

x2 = x[1:]
z2 = y[1:]

speedup_2 = z2[0]/z2
plt.plot(x2,speedup_2,'r',label="Normalised against 4 Cores",marker='o')

plt.legend(fontsize=10)

fig3 = plt.figure(figsize=(6,4),dpi = 250)
eff = np.zeros(10)
for i in range(len(eff)):
    eff[i] = speedup[i]/x[i]

eff_2 = np.zeros(9)
for i in range(len(eff_2)):
    eff_2[i] = 4*speedup_2[i]/x2[i]    

plt.plot(x,eff,'k',label="Normalised against 1 Core",marker='o',color='b')
#plt.plot(x2,eff_2,'r',label="Normalised against 4 Cores",marker='o')
plt.xlabel("Number of Cores")
plt.ylabel("Efficiency")
plt.title("Efficiency Normalised Against 1 Core")

fig5 = plt.figure(figsize=(6,4),dpi=250)
plt.plot(x2,eff_2,'r',label="Normalised against 4 Cores",marker='o')
plt.title("Efficiency Normalised Against 4 Cores")
plt.xlabel("Number of Cores")
plt.ylabel("Efficiency")
fig3 = plt.figure(figsize=(6,4),dpi=250)
k=4
r = 0.075
x = np.linspace(0,1000,101)
y = x*(x-1)/2

y2 = k*((x/k)*(x/k-1)/2+4*(x)**2*(r**2/k)+4*(x)**2*r**4)
k=9

y3 = k*((x/k)*(x/k-1)/2+4*(x)**2*(r**2/k)+4*(x)**2*r**4)
plt.plot(x,y,color='r', label='1 Core')
plt.plot(x,y2,color='b',label='4 Cores')
plt.plot(x,y3,color='g',label='9 Cores')
plt.legend(fontsize=10)
plt.ylabel("Number of Computations per Iteration")
plt.xlabel("Number of particles.")
plt.title("Number of Compuations Against Number of Particles. Rcutoff = {}".format(r))
