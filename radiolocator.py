import numpy as np
from numpy import pi
from scipy.optimize import curve_fit
from scipy.special import erf

# import pandas.DataFrame


xrad = yrad = 0

class Radiolocator():
    def __init__(self, h=1e6, xi=0.0, theta=1.5, c=299792458, sigma=1, 
                    angles_in='degrees', pulse = np.array([None]), t = None):
        self.R = 6370e3 # Радиус Земли в метрах
        self.c = c # Скорость света в м/с
        if angles_in=='degrees':
            self.xi = np.deg2rad(xi) # Отклонение антенны в радианах
            self.theta = np.deg2rad(theta) # Ширина диаграммы направленности в радианах
            self.Gamma = self.gamma(self.theta)

        if pulse.any() != None:
            params = self.calc(t,pulse)
            print('xi = {},\nh = {},\nsigma_s = {}'.format(params[0],params[1],params[2]))

        else:
            self.h = h # Высота орбиты в метрах
            self.sigma_s = sigma # Дисперсия наклонов


        self.T = 3e-9 # Временное разрешение локатора


    def H(self,h):
        return h*( 1+ h/self.R )
    
    def A(self,gamma,xi,A0=1.):
        return A0*np.exp(-4/gamma * np.sin(xi)**2 )

    def u(self,t,alpha,sigma_c):
        return (t - alpha * sigma_c**2) / (np.sqrt(2) * sigma_c)

    def v(self,t,alpha,sigma_c):
        return alpha*(t - alpha/2 * sigma_c**2)

    def alpha(self,beta,delta):
        return delta - beta**2/4

    def delta(self,gamma,xi,h):
        return 4/gamma * self.c/self.H(h) * np.cos(2 * xi)
    
    def gamma(self,theta):
        return 2*np.sin(theta/2)**2/np.log(2)

    def beta(self,gamma,xi,h):
        return 4/gamma * np.sqrt(self.c/self.H(h)) * np.sin(2*xi)


    def sigma_c(self,sigma_s):
        sigma_p = 0.425 * self.T 
        return np.sqrt(sigma_p**2 + (2*sigma_s/self.c)**2 )

    def pulse(self,t, dim = 1):

        self.dim = dim
        gamma = self.Gamma
        delta = self.delta(gamma,self.xi,self.h)
        beta  = self.beta(gamma,self.xi,self.h)

        if dim == 1:
            alpha = self.alpha(beta,delta)
        else:
            alpha = self.alpha(beta/np.sqrt(2),delta)

        sigma_c = self.sigma_c(self.sigma_s)

        u = self.u(t, alpha, sigma_c)
        v = self.v(t, alpha, sigma_c)

        A = self.A(gamma,self.xi)
        pulse = A*np.exp(-v)*( 1 + erf(u) )
        
        if self.dim == 2:
            alpha = gamma
            u = self.u(t, alpha, sigma_c)
            v = self.v(t, alpha, sigma_c)
            pulse -= A/2*np.exp(-v)*( 1 + erf(u) )

        return pulse

    def pulse_v(self, v, dim = 1):

        self.dim = dim
        gamma = self.gamma(self.theta)

        delta = self.delta(gamma,self.xi,self.h)
        beta  = self.beta(gamma,self.xi,self.h)

        alpha = self.alpha(beta,delta)

        sigma_c = self.sigma_c(self.sigma_s)

        u = np.sqrt(2)*v/(alpha*sigma_c) - alpha*sigma_c/np.sqrt(2)

        A = self.A(gamma,self.xi)
        pulse = A*np.exp(-v)*( 1 + erf(u) )

        return pulse
    