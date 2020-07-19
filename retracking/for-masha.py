import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
import pandas as pd

from scipy import optimize as opt
from scipy.interpolate import interp1d


import pandas as pd
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 




def spectrum_correction(F, F_CUR, OUR_HZ):
    S = interp1d(F, OUR_HZ,kind='cubic')
    func = interp1d(F, F_CUR,kind='cubic')

    f = lambda x, F: func(x) - F  


    x0 = [[] for i in range(F.size)]
    ind = [[] for i in range(F.size)]

    bounds = ((F[0], F[-1]),)

    min1 = opt.minimize(lambda x,F: +f(x,F), F[0], args=(0), bounds=bounds).x[0]
    max1 = opt.minimize(lambda x,F: -f(x,F), F[0], args=(0), bounds=bounds).x[0]

    if min1 == F[0]:
        sign = +1
        ext = max1

    if max1 == F[0]:
        sign = -1
        ext = min1 


    exts = [F[0], ext]

    xtol=1e-5
    while exts[-1] <= 0.95*F[-1]:
        ext *= 1.1
        ext = opt.minimize(lambda x,F: sign*f(x,F), ext, 
                args=(0), tol=xtol, bounds=bounds, method='trust-constr').x[0]

        exts.append(ext)
        sign *= -1

    for i in range(F.size):
        for j in range(1,len(exts)):
            try:
                sol = opt.brenth(f, exts[j-1], exts[j], xtol = xtol, args=(F_CUR[i]) )
                x0[i].append(sol)
            except:
                pass

    S_cur = np.zeros(F.size)
    for i in range(F.size):
        for j in range(len(x0[i])):
            S_cur[i] += S(x0[i][j])
        if S_cur[i] == 0:
            S_cur[i] = None
    
    S_cur = S_cur[F_CUR.argsort()]
    F_CUR.sort()
    return F_CUR, S_cur 



files = [4, 5]
for i in range(len(files)):
    df = read_csv('retracking/wvw_%d.dat' % (files[i]), sep ='\s+', header = 0).T
    KT = df.iloc[0].values
    JONSWAP_KT = df.iloc[1].values
    OUR_Wt = df.iloc[2].values
    OUR_HZ = df.iloc[3].values
    WT = df.iloc[4].values
    F = df.iloc[5].values
    VCUR_HZ = df.iloc[6].values
    F_CUR0 = df.iloc[7].values
    F_CUR = df.iloc[8].values

    fig, ax = plt.subplots()



    f, S = spectrum_correction(F, F_CUR, OUR_HZ)
    data = {}
    data.update({'f': f })
    data.update({'S': S })
    data = pd.DataFrame(data)
    data.to_csv(os.path.join(currentdir, str(files[i]) + '.csv'), index = False, sep=';')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.plot(f, S,'-', label = "$\\tilde S(f)$", )
    ax.plot(F, OUR_HZ,'-', label = "$S(f)$", )
    ax.plot(F_CUR0, OUR_HZ,'-', label = "$S(f)$", )
    ax.set_xlabel("$f$, Гц ")
    ax.set_ylabel("$S(f)$")
    ax.legend()
    ax.set_title('%d.dat' % (files[i]) )

plt.show()

