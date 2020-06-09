import numpy as np
import matplotlib.pyplot as plt

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from retracking import Retracking

from pandas import read_csv
df = read_csv('imp04_10.dat', sep ='\s+', header = None).T
t0 = df.iloc[0].values*1e-9
pulse0 = df.iloc[1].values
pulse = pulse0
t = t0

retracking = Retracking()

N = np.argmax(pulse0)
plt.figure()
p = retracking.leading_edge(t, pulse)
plt.plot(t[N:], -p[0]*t[N:] + p[1], label="аппроксимация")
plt.plot(t[N:], np.log(pulse0[N:]), label='$P(t>t_{max})$')
plt.xlabel("$t,$ с")
plt.ylabel("ln$P(t),$ усл. ед.")
plt.legend(fontsize=14)
plt.savefig('fig1.png',dpi=300,bbox_inches="tight")



plt.figure()
plt.plot(t, np.exp(-p[0]*t + p[1]),label="exp")
p, func = retracking.trailing_edge(t, pulse)
plt.plot(t, func(t,p[0],p[1],p[2]),label="erf")
plt.plot(t,pulse0, label='исходный импульс')
plt.xlabel("$t,$ с")
plt.ylabel("$P(t),$ усл. ед.")
plt.legend(fontsize=14)
plt.savefig('fig2.png',dpi=300,bbox_inches="tight")

plt.figure()
p, func = retracking.pulse(t, pulse)
plt.plot(t, func(t,p[0],p[1],p[2],p[3],p[4]),label="аппроксимация")
plt.plot(t,pulse0, label='исходный импульс')
plt.legend(fontsize=14)
plt.xlabel("$t,$ с")
plt.ylabel("$P(t),$ усл. ед.")
plt.savefig('fig3.png',dpi=300,bbox_inches="tight")
plt.show()