
import numpy as np
from numpy import pi
from scipy import interpolate,integrate
from tqdm import tqdm
from numba import jit,njit, prange

from spectrum import Spectrum
import matplotlib.pyplot as plt



class Surface(Spectrum):
    def __init__(self, surface_data, spectrum_data):



        self.N = surface_data[0]
        self.M = surface_data[1]
        random_phases = surface_data[2]
        kfrag = surface_data[3]
        self.wind = surface_data[4]

        Spectrum.__init__(self, spectrum_data)
        self.spectrum = self.get_spectrum()


        if kfrag == 'log':
            self.k = np.logspace(np.log10(self.k_m/4), np.log10(self.k_edge['Ku']), self.N + 1)
        elif kfrag == 'quad':
            self.k = np.zeros(self.N+1)
            for i in range(self.N+1):
                self.k[i] = self.k_m/4 + (self.k_edge['Ku']-self.k_m/4)/(self.N+1)**2*i**2

        else:
            self.k = np.linspace(self.k_m/4, self.k_edge['Ku'], self.N + 1)

        self.k_ku = self.k[ np.where(self.k <= self.k_edge['Ku']) ]
        self.k_c = self.k[ np.where(self.k <= self.k_edge['C']) ]

        print(\
            "Параметры модели:\n\
                N={},\n\
                M={},\n\
                U={} м/с,\n\
                Band={}\n\
                mean=0".format(self.N,self.M,self.U10,self.band)
            )

        self.phi = np.linspace(-np.pi, np.pi,self.M + 1)
        self.phi_c = self.phi + self.wind




        if random_phases == 0:
            self.psi = np.array([
                    [0 for m in range(self.M) ] for n in range(self.N) ])
        elif random_phases == 1:
            self.psi = np.array([
                [ np.random.uniform(0,2*pi) for m in range(self.M)]
                            for n in range(self.N) ])

        print(\
            "Методы:\n\
                Случайные фазы     {}\n\
                Отбеливание        0\n\
                Заостренная волна  {}\n\
            ".format(bool(random_phases),  False)
            )
                            


        print('Вычисление амплитуд гармоник...')
        self.A = self.amplitude(self.k)
        self.F = self.angle(self.k,self.phi)
        print('Подготовка завершена.')

    def B(self,k):
          def b(k):
              b=(
                  -0.28+0.65*np.exp(-0.75*np.log(k/self.k_m))
                  +0.01*np.exp(-0.2+0.7*np.log10(k/self.k_m))
                )
              return b
          B=10**b(k)
          return B

    def Phi(self,k,phi):
        # Функция углового распределения
        phi = phi - self.wind
        normalization = lambda B: B/np.arctan(np.sinh(2* (pi)*B))
        B0 = self.B(k)
        A0 = normalization(B0)
        Phi = A0/np.cosh(2*B0*(phi) )
        return Phi


    def angle(self,k, phi):
        M = self.M
        N = self.N
        Phi = lambda phi,k: self.Phi(k,phi)
        integral = np.zeros((N,M))
        for i in range(N):
            for j in range(M):
                # integral[i][j] = integrate.quad( Phi, phi[j], phi[j+1], args=(k[i],) )[0]
                integral[i][j] = np.trapz( Phi( phi[j:j+2],k[i] ), phi[j:j+2])
        amplitude = np.sqrt(2 *integral )
        return amplitude

    def amplitude(self, k):
        N = k.size
        S = self.spectrum
        integral = np.zeros(k.size-1)
        for i in range(1,N):
            integral[i-1] = integrate.quad(S,k[i-1],k[i])[0] 
        amplitude = np.sqrt(2 *integral )
        return np.array(amplitude)


    
    def surfaces(self,r,t):
        N = self.N
        M= self.M
        k = self.k
        phi = self.phi
        A = self.A
        F = self.F
        psi = self.psi
        self.surface = 0
        self.surface_s = 0
        self.surface_v = 0
        self.surface_xx = 0
        self.surface_yy = 0
        self.surface_xt = 0
        self.surface_yt = 0
        self.cwm_x = 0
        self.cwm_y = 0
        self.cwm_xdot = 0
        self.cwm_ydot = 0
        # self.amplitudes = np.array([ A[i]*sum(F[i])  for i in range(N)])
        progress_bar = tqdm( total = N*M,  leave = False )
        for n in range(N):
            for m in range(M):
                kr = k[n]*(r[0]*np.cos(phi[m])+r[1]*np.sin(phi[m]))
                tmp = A[n] * \
                    np.cos( 
                        +kr
                        +psi[n][m]
                        +self.omega_k(k[n])*t) \
                        * F[n][m]

                tmp1 = -A[n] * \
                    np.sin( 
                        +kr
                        +psi[n][m]
                        +self.omega_k(k[n])*t) \
                        * F[n][m]

                self.surface += tmp

                self.surface_s  += k[n]*tmp1
                self.surface_xx += k[n]*np.cos(phi[m])*tmp
                self.surface_yy += k[n]*np.sin(phi[m])*tmp 

                self.surface_v  += self.omega_k(k[n])*tmp
                self.surface_xt += self.omega_k(k[n])*np.cos(phi[m])*tmp 
                self.surface_yt += self.omega_k(k[n])*np.sin(phi[m])*tmp 

                self.cwm_x += tmp1*np.cos(phi[m])
                self.cwm_y += tmp1*np.sin(phi[m])

                progress_bar.update(1)
        progress_bar.close()
        progress_bar.clear()
        heights = [self.surface]
        slopes = [self.surface_s,self.surface_xx, self.surface_yy]
        velocity = [self.surface_v,self.surface_xt, self.surface_yt]
        cwm = [self.cwm_x, self.cwm_y, ]
        return heights,slopes,velocity, cwm

    def surfaces_band(self,r,t):
        N = self.N
        M= self.M
        k = self.k
        phi = self.phi
        A = self.A
        F = self.F
        psi = self.psi

        s = 0
        s_xx = 0
        s_yy = 0

        s1 = 0
        s_xx1 = 0
        s_yy1 = 0

        progress_bar = tqdm( total = N*M,  leave = False )

        for n in range(N):
            for m in range(M):
                kr = k[n]*(r[0]*np.cos(phi[m])+r[1]*np.sin(phi[m]))
                tmp = A[n] * \
                    np.cos( 
                        +kr
                        +psi[n][m]
                        +self.omega_k(k[n])*t) \
                        * F[n][m]

                tmp1 = -A[n] * \
                    np.sin( 
                        +kr
                        +psi[n][m]
                        +self.omega_k(k[n])*t) \
                        * F[n][m]

                if k[n] <= self.k_c.max():
                    s += tmp
                    s_xx += k[n]*np.cos(phi[m])*tmp1
                    s_yy += k[n]*np.sin(phi[m])*tmp1 

                else :
                    s1 += tmp
                    s_xx1 += k[n]*np.cos(phi[m])*tmp1
                    s_yy1 += k[n]*np.sin(phi[m])*tmp1

                progress_bar.update(1)
        progress_bar.close()
        progress_bar.clear()
        band_c = [s, np.sqrt(s_xx**2+s_yy**2), s_xx, s_yy]
        band_ku = [s+s1, np.sqrt((s_xx+s_xx1)**2 + (s_yy+s_yy1)**2), s_xx + s_xx1, s_yy + s_yy1]
        return band_c, band_ku

    def choppy_wave(self, r, t):
        N = self.N
        M= self.M
        k = self.k
        phi = self.phi
        A = self.A
        F = self.F
        psi = self.psi

        self.cwm_x = 0
        self.cwm_y = 0
        for n in range(N):
            for m in range(M):
                kr = k[n]*(r[0]*np.cos(phi[m])+r[1]*np.sin(phi[m]))
                tmp = -A[n] * \
                    np.sin( 
                        +kr
                        +psi[n][m]
                        +self.omega_k(k[n])*t) \
                        * F[n][m]

                self.cwm_x += tmp * np.cos(phi[m])
                self.cwm_y += tmp * np.sin(phi[m])
        return [self.cwm_x, self.cwm_y]




    def choppy_wave_jac(self, r, t):
        N = self.N
        M= self.M
        k = self.k
        phi = self.phi
        A = self.A
        F = self.F
        psi = self.psi

        self.cwm_x_dot = 0
        self.cwm_y_dot = 0
        for n in range(N):
            for m in range(M):
                kr = k[n]*(r[0]*np.cos(phi[m])+r[1]*np.sin(phi[m]))
                tmp = -A[n] * \
                    np.cos( 
                        +kr
                        +psi[n][m]
                        +self.omega_k(k[n])*t) \
                        * F[n][m]

                self.cwm_x_dot += tmp * k[n]*np.cos(phi[m])**2
                self.cwm_y_dot += tmp * k[n]*np.sin(phi[m])**2
        return [self.cwm_x_dot, self.cwm_y_dot]