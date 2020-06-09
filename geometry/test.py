import numpy as np
import matplotlib.pyplot as plt

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from pulse import Pulse

xmax = 500
gridsize = 250
x0 = np.linspace(-xmax,xmax, gridsize)  
y0 = np.linspace(-xmax,xmax, gridsize)

z0 = 1e3

x,y = np.meshgrid(x0,y0)

phi0 = np.linspace(0, 2*np.pi, gridsize)
r0 = np.linspace(0, xmax, gridsize)
phi, r = np.meshgrid(phi0,r0)

surface = [np.zeros(x.size), np.zeros(x.size), np.zeros(x.size)]
pulse = Pulse(surface, 'Ku', x = x0, y = y0, r0=[0,0,z0])
theta0 = pulse.main()

plt.figure()
plt.contourf(x, y, theta0.reshape(x.shape))
plt.colorbar()



fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
pulse = Pulse(surface, 'Ku', x = r*np.cos(phi), y =r*np.sin(phi), r0=[0,0,z0])
theta0 = pulse.main()
im = ax.contourf(phi, r, theta0.reshape(x.shape))
fig.colorbar(im)


plt.show()
# 