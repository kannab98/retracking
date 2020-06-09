import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.optimize import curve_fit
from scipy.special import erf
from pandas import read_csv

class Retracking():
    def __init__(self):
        self.c = 299792458
        self.T = 3e-9

    def leading_edge(self,t,pulse):
        n = np.argmax(pulse)
        pulse = np.log(pulse[n:])
        t = t[n:]
        line = lambda t,alpha,b: -alpha*t + b   
        popt = curve_fit(line, 
                            xdata=t,
                            ydata=pulse,
                            p0=[1e6,0],
                        )[0]

        self.alpha = popt[0]
        return popt
    
    def trailing_edge(self, t, pulse):
        A0 = (max(pulse) - min(pulse))/2
        N = np.argmax(pulse)
        pulse = pulse[0:N]
        t = t[0:N]

        # print(A0,pulse,t)
        func = lambda t, A, tau, sigma_l:   A * (1 + erf( (t-tau)/sigma_l ))
        popt = curve_fit(func, 
                            xdata=t,
                            ydata=pulse,
                            p0=[A0, t.max()-t.min(), (t[-1]-t[0])/t.size],)[0]
                            
        self.A = popt[0]
        self.tau = popt[1]
        self.sigma_l = popt[2]
        return popt,func

    def pulse(self, t, pulse):
        self.leading_edge(t, pulse)
        self.trailing_edge(t, pulse)

        ice = lambda t, A,alpha,tau,sigma_l,T:  A * np.exp( -alpha * (t-tau/2) ) * (1 + erf( (t-tau)/sigma_l )) + T
        popt = curve_fit(ice, 
                            xdata=t,
                            ydata=pulse,
                            p0=[self.A, self.alpha, self.tau, self.sigma_l, 0],
                        )[0]

        return popt, ice 

    def height(self, sigma_l):
        sigma_p = 0.425 * self.T
        sigma_c = sigma_l/np.sqrt(2)
        # sigma_c = sigma_l
        sigma_s = np.sqrt((sigma_c**2 - sigma_p**2)/2)*self.c/np.sqrt(2)
        return sigma_s