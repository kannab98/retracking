import os
import pandas as pd
import numpy as np
import datetime
from surface import Surface
import math
import matplotlib.pyplot as plt
from data import Data
from pulse import Pulse




N = 2**9
M = 2**7

Xmax = 5
x0 = np.linspace(-Xmax,Xmax, 25)  
y0 = np.linspace(-Xmax,Xmax, 25)

wind = 0
data = Data(random_phases = 1, N = N, M = M, band='Ku',wind=wind, U10=5)
data_surface = data.surface()
data_spectrum = data.spectrum()
surface = Surface(data_surface, data_spectrum)


z0 = 30
c = 1500
omega = 2*np.pi*c/8e-3
k = omega/c
timp = 40e-6
T0 = z0/c

T = np.linspace(0.0199, np.sqrt(z0**2 + Xmax**2)/c,256)

count = 0
while count < 100:
    x0 = np.random.uniform(-Xmax, Xmax, size=25)  
    y0 = np.random.uniform(-Xmax, Xmax, size=25)
    x, y = np.meshgrid(x0, y0)
    band_c, band_ku = surface.surfaces_band([x,y],0)

    pulse = Pulse(band_ku, 'Ku', x0, y0, [0,0,z0], timp=timp, c=c, projection='polar')
    theta0 = pulse.main()
    index = pulse.index[0]

    count += index.size
    print(count)
    P = np.zeros(T.size)

    for i in range(T.size):
        P[i] += pulse.power(T[i], omega, timp, pulse.Rabs, pulse.theta)
    plt.ion()
    plt.clf()
    plt.plot(T, P)
    plt.show()
    plt.pause(1)


plt.plot(T, P)
plt.show()


