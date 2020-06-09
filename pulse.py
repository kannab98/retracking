import numpy as np
class Pulse:
    def __init__(self, surface, band, 
                       x, y, r0=[0,0,1e3],
                       gane_width=np.deg2rad(1.5),
                       c =  299792458, timp = 3e-9):

        if len(x.shape) < 2:
            self.x, self.y = np.meshgrid(x,y)
        else:
            self.x, self.y = x, y
        self.r = np.vstack((
                            self.x.flatten(), 
                            self.y.flatten(), 
                            surface[0].flatten()
                        ))

        self.r0 = np.vstack((
                      +r0[0]*np.zeros(self.x.size), 
                      +r0[1]*np.zeros(self.y.size), 
                      -r0[2]*np.ones(surface[0].size)
                    ))

        self.n = np.vstack((
                        surface[1].flatten(),
                        surface[2].flatten(),
                        np.ones(surface[0].size)
                    ))

        if band == 'Ku':
            self.omega = 2*np.pi * 12e9
        elif band == 'C':
            self.omega = 2*np.pi * 4e9
        else:
            self.omega = band

        self.c =  c
        self.k = self.omega/self.c
        self.R = self.r - self.r0

        self.Rabs = self.Rabs_calc(self.R)
        self.Nabs = self.Nabs_calc(self.n)
        self.theta = self.theta_calc(self.R, self.Rabs)

        #!$gane\_width \equiv \theta_{3dB}$!
        gane_width = np.deg2rad(1.5) # Ширина диаграммы направленности в радианах
        self.gamma = 2*np.sin(gane_width/2)**2/np.log(2)
    
    def main(self):
        # self.theta  = self.theta_calc(self.R, self.Rabs)
        self.theta0 = self.theta0_calc(self.R, self.n, self.Rabs, self.Nabs)
        self.index = self.mirror_sort(self.r, self.r0, self.n, self.theta0)
        return self.theta0


    def G(self,theta,G0=1):
            # G -- диаграмма направленности
            # theta -- угол падения
            return G0*np.exp(-2/self.gamma * np.sin(theta)**2)
    
    def Rabs_calc(self, R):
        R = np.array(R)
        if len(R.shape)>1:
            Rabs = np.sqrt(np.sum(R**2,axis=0))
        else:
            Rabs = np.sqrt(np.sum(R**2))
        return Rabs 

    def Nabs_calc(self,n):
        N = np.sqrt(np.sum(n**2,axis=0))
        return N

    def theta_calc(self, R, Rabs):
        theta = R[-1,:]/Rabs
        return np.arccos(theta)

    def theta0_calc(self,R,n,Rabs,Nabs):
        theta0 = np.einsum('ij, ij -> j', R, n)
        theta0 *= 1/Rabs/Nabs
        return np.arccos(theta0)

    def mirror_sort(self,r, r0, n, theta, err = 1):
        index = np.where(theta < np.deg2rad(err))
        return index
    
    def sort(self, arr):
        arr = arr.flatten()
        return arr[self.index]

    def power(self, t, omega ,timp ,R, theta,):
        
        c = self.c 
        G = self.G
        tau = R/c

        index = [ i for i in range(tau.size) if 0 <= t - tau[i] <= timp ]
        theta = theta[index] 
        R = R[index]
        tau = tau[index]
        # Путь к поверхности
        #! $\omega\tau\cos(\theta) = kR$! 
        E0 = G(theta)/R
        e0 = np.exp(1j*omega*(t  - tau*np.cos(theta)) ) 
        # Путь от поверхности
        E0 = E0*e0*G(theta)/R
        e0 = np.exp(1j*omega*(tau + tau*np.cos(theta)) ) 
        
        
        E = np.sum(E0*e0)
        return np.abs(E)**2/2
    