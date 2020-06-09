from numba import  cuda, float32, float64
import os
import pandas as pd
import numpy as np
from time import time
import datetime
from surface import Surface
import math
import matplotlib.pyplot as plt
from data import Data
from pulse import Pulse



TPB=16
@cuda.jit
def kernel_c(ans, x, y, k, offset, phi, A, F, psi):

    i,j = cuda.grid(2)

    if i >= x.size and j >= y.size:
        return

    for n in range(k.size-offset):
        for m in range(phi.size):
            kr = k[n]*(x[i]*math.cos(phi[m]) + y[j]*math.sin(phi[m]))      
            tmp =  math.cos(kr + psi[n][m]) * A[n] * F[n][m]
            tmp1 = - math.sin(kr + psi[n][m]) * A[n] * F[n][m]
            ans[0,j,i] +=  tmp
            ans[1,j,i] +=  tmp1 * k[n]
            ans[2,j,i] +=  tmp1 * k[n] * math.cos(phi[m])
            ans[3,j,i] +=  tmp1 * k[n] * math.sin(phi[m])

@cuda.jit
def kernel_ku(ans, x, y, k, offset,phi, A, F, psi):

    i,j = cuda.grid(2)

    if i >= x.size and j >= y.size:
        return

    for n in range(k.size-offset, k.size):
        for m in range(phi.size):
            kr = k[n]*(x[i]*math.cos(phi[m]) + y[j]*math.sin(phi[m]))      
            tmp  = + math.cos(kr + psi[n][m]) * A[n] * F[n][m]
            tmp1 = - math.sin(kr + psi[n][m]) * A[n] * F[n][m]
            ans[0,j,i] +=  tmp
            ans[1,j,i] +=  tmp1 * k[n]
            ans[2,j,i] +=  tmp1 * k[n] * math.cos(phi[m])
            ans[3,j,i] +=  tmp1 * k[n] * math.sin(phi[m])
            




N = 2**11
M = 2**9

offsetx = 1e9
# offsetx = 0
offsety = 1e10
# offsety = 0
Xmax  = 50
x0 = np.linspace(-Xmax,Xmax, 512) + offsetx
y0 = np.linspace(-Xmax,Xmax, 512) + offsety
x = cuda.to_device(x0)
y = cuda.to_device(y0)

wind = 30
data = Data(random_phases = 0, N = N, M=M, band='Ku',wind=wind, U10=5)
data_surface = data.surface()
data_spectrum = data.spectrum()
surface = Surface(data_surface, data_spectrum)

k = surface.k
phi = surface.phi
A = surface.A
F = surface.F
PHI_SIZE = F.shape[1]
psi = surface.psi
k = cuda.to_device(k)
phi = cuda.to_device(phi)
A = cuda.to_device(A)
F = cuda.to_device(F)
psi = cuda.to_device(psi)

threadsperblock = (TPB, TPB)
blockspergrid_x = math.ceil(x.size / threadsperblock[0])
blockspergrid_y = math.ceil(y.size / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)



offset = surface.k_ku.size - surface.k_c.size
band_c = np.zeros((4, x.size,y.size))
kernel_c[blockspergrid, threadsperblock](band_c, x, y, k, offset, phi, A, F, psi)
band_ku = np.array(band_c)
kernel_ku[blockspergrid, threadsperblock](band_ku, x, y, k, offset, phi, A, F, psi)





x0 = x0 - offsetx
y0 = y0 - offsety

z0 = 3e6
c = 3e8
omega = 2*np.pi*12e9
k = omega/c
timp = 3e-9
T0 = z0/c
T = np.linspace(T0-timp,np.sqrt(z0**2+Xmax**2)/c, 200)
P = np.zeros(T.size)



pulse = Pulse(band_c, 'C', x0, y0, [0, 0, z0], timp=timp)
theta0_c = pulse.main()
theta0_c = theta0_c.reshape((x.size,y.size))
pulse = Pulse(band_ku, 'Ku', x0, y0, [0, 0, z0], timp=timp)
theta0_ku = pulse.main()
theta0_ku = theta0_c.reshape((x.size,y.size))


data.export(x0,y0, surface,ku_band = band_ku, c_band=band_c, theta0_c=theta0_c,theta0_ku=theta0_ku)




# print(surface.U10)