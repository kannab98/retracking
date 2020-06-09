import configparser
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import pandas as pd

class Data():
    def __init__(self, conf_file = None,  N = None, M = None, 
            random_phases = None, kfrag = None, 
            wind = None, U10 = None, x = None, band = None, KT = None):


        if conf_file != None:
            config = configparser.ConfigParser()
            config.read(conf_file)
            config = config['Surface']


        if N == None:
            try:
                self.N = int(config['N'])
            except:
                self.N = 256
        else:
            self.N = N

        if M == None:
            try:
                self.M = int(config['M'])
            except: 
                self.M = 128
        else:
            self.M = M

        if wind == None:
            try:
                self.wind = float(config['WindDirection'])
                self.wind = np.deg2rad(self.wind)
            except:
                self.wind = 0
        else:
            self.wind = np.deg2rad(wind)

        if kfrag == None:
            try:
                self.kfrag  = config['WaveNumberFragmentation']
            except:
                self.kfrag = 'log'
        else:
            self.kfrag = kfrag

        if random_phases == None:
            try:
                self.random_phases = int(config['RandomPhases'])
            except:
                self.random_phases = 1
        else:
            self.random_phases = random_phases


        if conf_file != None:
            config = configparser.ConfigParser()
            config.read(conf_file)
            config = config['Spectrum']


        # скорость ветра на высоте 10 м над уровнем моря.
        if U10 == None:
            try:
                self.U10 = str2num(config['WindSpeed'])
            except:
                self.U10 = 5
        else:
            self.U10 = U10

        if x == None:
            try:
                self.x = str2num(config['WaveEvolution'])
            except:
                self.x = 20170

        self.band = band
        if self.band == None:
            try:
                self.band = config['Band']
            except:
                self.band = 'Ku'


    def surface(self):
        return self.N, self.M, self.random_phases, self.kfrag, self.wind, 

    def spectrum(self):
        return self.U10, self.x, self.band, None
    
    def export(self, x,y, surface, datadir = None, k_ku=None, ku_band=None, k_c=None, c_band=None, theta0_ku = None, theta0_c = None):
        if datadir == None:
            datadir = 'data' + datetime.datetime.now().strftime("%m%d_%H%M")



        outputdir = os.path.join('.',datadir,'output')
        inputdir = os.path.join('.',datadir,'input')
        kudir = os.path.join(outputdir,'ku-band')
        cdir = os.path.join(outputdir,'c-band')
        diffdir = os.path.join(outputdir,'diff-ku-c')


        os.makedirs(datadir)
        os.makedirs(inputdir)
        os.makedirs(outputdir)
        os.makedirs(kudir)
        os.makedirs(cdir)
        os.makedirs(diffdir)



        k = surface.k0
        S  = surface.spectrum
        sigma_h = np.trapz(S(k),k)
        sigma_s = np.trapz(S(k)*k**2, k)
        k_m = surface.k_m
        data = pd.DataFrame({'sigma_h': sigma_h, 'sigma_s':sigma_s, 'U':surface.U10, 'k_m':k_m, 'lambda_m':2*np.pi/k_m}, index=[0])
        data.to_csv(os.path.join(inputdir,'input.tsv'), index=False, sep='\t')


        x,y = np.meshgrid(x,y)
        # fig_spec,ax_spec = plt.subplots()
        def minmax_ticks(cbar, minVal, maxVal):
            # Get the default ticks and tick labels
            ticklabels = cbar.ax.get_ymajorticklabels()
            ticks = list(cbar.get_ticks())

            # Append the ticks (and their labels) for minimum and the maximum value
            cbar.set_ticks([minVal.round(1), maxVal.round(1)] + ticks)
            cbar.set_ticklabels([minVal.round(1), maxVal.round(1)]+ticklabels)

        if type(ku_band) != 'NoneType':

            # ax_spec.loglog(k_ku, spectrum(k_ku),label='Ku')


            fig_h1, ax_h1 = plt.subplots()
            plt.contourf(x,y, ku_band[0],levels=100)
            ax_h1.set_xlabel('x')
            ax_h1.set_ylabel('y')
            bar = plt.colorbar()
            # minmax_ticks(bar, np.min(ku_band[0]), np.max(ku_band[0]))
            bar.set_label('высоты')

            fig_s1, ax_s1 = plt.subplots()
            plt.contourf(x,y, ku_band[1],levels=100)
            ax_s1.set_xlabel('x')
            ax_s1.set_ylabel('y')
            bar = plt.colorbar()
            # minmax_ticks(bar, np.min(ku_band[1]), np.max(ku_band[1]))
            bar.set_label('наклоны')

            fig_sx1, ax_sx1 = plt.subplots()
            plt.contourf(x,y, ku_band[2],levels=100)
            ax_sx1.set_xlabel('x')
            ax_sx1.set_ylabel('y')
            bar = plt.colorbar()
            # minmax_ticks(bar, np.min(ku_band[2]), np.max(ku_band[2]))
            bar.set_label('наклоны X')

            fig_sy1, ax_sy1 = plt.subplots()
            plt.contourf(x,y, ku_band[3],levels=100)
            ax_sy1.set_xlabel('x')
            ax_sy1.set_ylabel('y')
            bar = plt.colorbar()
            # minmax_ticks(bar, np.min(ku_band[3]), np.max(ku_band[3]))
            bar.set_label('наклоны Y')
            

            fig_h1.savefig(os.path.join(kudir,'heights.png'), dpi=300, bbox_inches='tight')
            fig_s1.savefig(os.path.join(kudir,'slopes.png'), dpi=300, bbox_inches='tight')
            fig_sx1.savefig(os.path.join(kudir,'slopesxx.png'), dpi=300, bbox_inches='tight')
            fig_sy1.savefig(os.path.join(kudir,'slopesyy.png'), dpi=300, bbox_inches='tight')

            data = pd.DataFrame({'x': x.flatten().round(2),'y': y.flatten().round(2), 'heights': ku_band[0].flatten().round(4) })
            data.to_csv(os.path.join(kudir,'heights.tsv'), index=False, sep='\t')
            data = pd.DataFrame({'x': x.flatten().round(2),'y': y.flatten().round(2), 'slopes': ku_band[1].flatten().round(4) })
            data.to_csv(os.path.join(kudir,'slopes.tsv'), index=False, sep='\t')
            data = pd.DataFrame({'x': x.flatten().round(2),'y': y.flatten().round(2), 'slopesxx': ku_band[2].flatten().round(4) })
            data.to_csv(os.path.join(kudir,'slopesxx.tsv'), index=False, sep='\t')
            data = pd.DataFrame({'x': x.flatten().round(2),'y': y.flatten().round(2), 'slopesyy': ku_band[3].flatten().round(4) })
            data.to_csv(os.path.join(kudir,'slopesyy.tsv'), index=False, sep='\t')

            if type(theta0_ku) != 'NoneType':
                fig_th1, ax_th1 = plt.subplots()
                ax_th1.set_xlabel('x')
                ax_th1.set_ylabel('y')
                plt.contourf(x, y, np.rad2deg(theta0_ku), levels = 100)
                bar = plt.colorbar()
                bar.set_label('Угол падения на поверхность')
                fig_th1.savefig(os.path.join(kudir,'theta0.png'), dpi=300, bbox_inches='tight')
                data = pd.DataFrame({'x': x.flatten().round(2),'y': y.flatten().round(2), 'theta0': theta0_ku.flatten().round(4) })
                data.to_csv(os.path.join(kudir,'theta0.tsv'), index=False, sep='\t')

        if type(c_band) != 'NoneType':

            # ax_spec.loglog(k_ku, spectrum(k_ku),label='Ku')

            fig_h2, ax_h2 = plt.subplots()
            plt.contourf(x,y, c_band[0],levels=100)
            ax_h2.set_xlabel('x')
            ax_h2.set_ylabel('y')
            bar = plt.colorbar()
            minmax_ticks(bar, np.min(c_band[0]), np.max(c_band[0]))
            bar.set_label('высоты')

            fig_s2, ax_s2 = plt.subplots()
            plt.contourf(x,y, c_band[1],levels=100)
            ax_s2.set_xlabel('x')
            ax_s2.set_ylabel('y')
            bar = plt.colorbar()
            minmax_ticks(bar, np.min(c_band[1]), np.max(c_band[1]))
            bar.set_label('наклоны')

            fig_sx2, ax_sx2 = plt.subplots()
            plt.contourf(x,y, c_band[2],levels=100)
            ax_sx2.set_xlabel('x')
            ax_sx2.set_ylabel('y')
            bar = plt.colorbar()
            minmax_ticks(bar, np.min(c_band[2]), np.max(c_band[2]))
            bar.set_label('наклоны X')

            fig_sy2, ax_sy2 = plt.subplots()
            plt.contourf(x,y, c_band[3],levels=100)
            ax_sy2.set_xlabel('x')
            ax_sy2.set_ylabel('y')
            bar = plt.colorbar()
            minmax_ticks(bar, np.min(c_band[3]), np.max(c_band[3]))
            bar.set_label('наклоны Y')

            fig_h2.savefig(os.path.join(cdir,'heights.png'), dpi=300, bbox_inches='tight')
            fig_s2.savefig(os.path.join(cdir,'slopes.png'), dpi=300, bbox_inches='tight')
            fig_sx2.savefig(os.path.join(cdir,'slopesxx.png'), dpi=300, bbox_inches='tight')
            fig_sy2.savefig(os.path.join(cdir,'slopesyy.png'), dpi=300, bbox_inches='tight')

            data = pd.DataFrame({'x': x.flatten().round(2),'y': y.flatten().round(2), 'heights': c_band[0].flatten().round(4) })
            data.to_csv(os.path.join(cdir,'heights.tsv'), index=False, sep='\t')
            data = pd.DataFrame({'x': x.flatten().round(2),'y': y.flatten().round(2), 'slopes': c_band[1].flatten().round(4) })
            data.to_csv(os.path.join(cdir,'slopes.tsv'), index=False, sep='\t')
            data = pd.DataFrame({'x': x.flatten().round(2),'y': y.flatten().round(2), 'slopesxx': c_band[2].flatten().round(4) })
            data.to_csv(os.path.join(cdir,'slopesxx.tsv'), index=False, sep='\t')
            data = pd.DataFrame({'x': x.flatten().round(2),'y': y.flatten().round(2), 'slopesyy': c_band[3].flatten().round(4) })
            data.to_csv(os.path.join(cdir,'slopesyy.tsv'), index=False, sep='\t')

            if type(theta0_c) != 'NoneType':
                fig_th2, ax_th2 = plt.subplots()
                ax_th2.set_xlabel('x')
                ax_th2.set_ylabel('y')
                plt.contourf(x, y, np.rad2deg(theta0_c), levels = 100)
                bar = plt.colorbar()
                bar.set_label('Угол падения на поверхность')
                fig_th2.savefig(os.path.join(cdir,'theta0.png'), dpi=300, bbox_inches='tight')
                data = pd.DataFrame({'x': x.flatten().round(2),'y': y.flatten().round(2), 'theta0': theta0_c.flatten().round(4) })
                data.to_csv(os.path.join(cdir,'theta0.tsv'), index=False, sep='\t')

        if type(c_band) != 'NoneType' and type(ku_band) != 'NoneType':
            fig_h3, ax_h3 = plt.subplots()
            plt.contourf(x,y, ku_band[0] - c_band[0],levels=100)
            ax_h3.set_xlabel('x')
            ax_h3.set_ylabel('y')
            bar = plt.colorbar()
            bar.set_label('высоты')

            fig_s3, ax_s3 = plt.subplots()
            plt.contourf(x,y, ku_band[1] - c_band[1],levels=100)
            ax_s3.set_xlabel('x')
            ax_s3.set_ylabel('y')
            bar = plt.colorbar()
            bar.set_label('наклоны')

            fig_sx3, ax_sx3 = plt.subplots()
            plt.contourf(x,y, ku_band[2] - c_band[2],levels=100)
            ax_sx3.set_xlabel('x')
            ax_sx3.set_ylabel('y')
            bar = plt.colorbar()
            bar.set_label('наклоны X')

            fig_sy3, ax_sy3 = plt.subplots()
            plt.contourf(x,y, ku_band[3] - c_band[3],levels=100)
            ax_sy3.set_xlabel('x')
            ax_sy3.set_ylabel('y')
            bar = plt.colorbar()
            bar.set_label('наклоны Y')
            

            fig_h3.savefig(os.path.join(diffdir,'heights.png'), dpi=300, bbox_inches='tight')
            fig_s3.savefig(os.path.join(diffdir,'slopes.png'), dpi=300, bbox_inches='tight')
            fig_sx3.savefig(os.path.join(diffdir,'slopesxx.png'), dpi=300, bbox_inches='tight')
            fig_sy3.savefig(os.path.join(diffdir,'slopesyy.png'), dpi=300, bbox_inches='tight')


            if type(theta0_ku) != 'NoneType' and type(theta0_c) != 'NoneType':
                fig_th2, ax_th2 = plt.subplots()
                ax_th2.set_xlabel('x')
                ax_th2.set_ylabel('y')
                plt.contourf(x, y, np.rad2deg(theta0_ku - theta0_c), levels = 100)
                bar = plt.colorbar()
                bar.set_label('Угол падения на поверхность')
                fig_th2.savefig(os.path.join(diffdir,'theta0.png'), dpi=300, bbox_inches='tight')


        # fig_spec.savefig(os.path.join(outputdir,'spectrum.png'), dpi=300, bbox_inches='tight')
        # fig_spec.savefig(os.path.join(outputdir,'spectrum.pdf'), bbox_inches='tight')





    
# par = Parameters(conf_file='config.ini')  
# print(par.surface())