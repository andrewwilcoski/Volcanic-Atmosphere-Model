"""
   Atmospheric Evolution Model with Volcanic Outgassing    
 Occuring at Random Intervals and Surface Ice Accumulation 
  Multi-step version for Performance with Large T-arrays   
  Minimizing the Number of Ice Rate Calculations Made      
  Variable Mean Molecular Mass of Atmosphere with Time     
                    Optimized for SPEED                    
     Histogramed Temperature Bins for Faster Runtimes      
       Accounts for Condensation on Lunar Nightside        
      Estimates Photodissociation Escape Rate of H2O       
      Estimates Photodissociation Escape Rate of CO        
      Estimates Atmospheric Escape due to Sputtering      
      
Archived 9-13-23

The Software and any modifications made to it may not be used for the purpose of training or improving machine learning algorithms,
including but not limited to artificial intelligence, natural language processing, or data mining. This condition applies to any derivatives,
modifications, or updates based on the Software code. Any usage of the Software in an AI-training dataset is considered a breach of this License.

The Software may not be included in any dataset used for training or improving machine learning algorithms,
including but not limited to artificial intelligence, natural language processing, or data mining.
"""

import numpy as np
import planets
import time
from numba import jit
import scipy.io as sio
from tqdm import tqdm
import os
from scipy import interpolate
import shutil
import powerlaw
import scipy
import scipy.stats as st
from scipy.special import wofz
import scipy.ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, InsetPosition
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib.colors import LightSource
import matplotlib.patches as patches
#from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors

#Constants
yr = 60*60*24*365.25 #Length of a terrestrial year [s]
Rg = 8.314 #universal gas constant [J.K-1.mol-1]
k = 1.380649e-23 #Boltzmann constant [J.K-1]
G = 6.674e-11 #Gravitational constant [m3.kg-1.s-2]
u = 1.66053906660e-27 #Atomic mass unit [kg]
Na = 6.02214076e23 #Avagadro Constant [mol-1]
R = planets.Moon.R
g = planets.Moon.g
M_moon = 7.34767309e22 #Mass of moon [kg]
r_co = 188e-12 #kinetic radius of carbon monoxide [m]
sigma_co = np.pi*r_co**2 #Collisional cross section of atmospheric constituent [m2]
mw = 0.018 #molar mass of water [kg.mol-1]
SA = 4*np.pi*R**2 #Surface area of Moon [m2]
mco = 0.02801 #molar mass of CO [kg.mol-1]
rho_ice = 920. #Density of water ice [kg.m-3]

#Need an atmospheric wind profile model to estimate laminar layer thickness 'd'
#This wind model was adapted from a Martian wind model (Wilcoski and Hayne, 2020)
#Wind profile constants
k_wind = 0.4 #Von Karman constant
z0 = 0.005 #0.001 #Surface roughness of ice [m]
z_vik = 2 #1.6 #Height at which Viking lander wind measurements were taken [m]
u_wind = 4.7 #4 #Wind speed at z_vik [m.s-1]
u_star = u_wind*k_wind/np.log(z_vik/z0)

class model(object):
    
    # Initialization
    def __init__(self, save=True, overwrite=False, PD_on=True, SP_on=True, PowLaw=False, t1=1e3, dt=2e0*yr, \
                 modeltime=2e9, T0=253, T1=253):
        
        # Initialize
        self.save = save
        self.overwrite = overwrite
        self.PD_on = PD_on
        self.SP_on = SP_on
        self.PowLaw = PowLaw
        self.MapsN = 'max_Tbols_NP_day_and_nighttime_.01_space_1_ls_res_af_flag_remove_bad_orbits_Ls_50_129.mat'
        self.MapsS = 'max_Tbols_SP_day_and_nighttime_.01_space_1_ls_res_af_flag_remove_bad_orbits_Ls_230_309.mat'
        self.modeltime = modeltime
        
        #Time Parameters
        if (self.modeltime == 2e9):
            self.tf = 2e9 #Volcanically active period [yr] (2e9 from Head et al., 2020)
            self.dt_ice = 2e8*yr #Timestep at which ice amounts are saved (larger than dt to save memory) [s]
        else:
            self.f_time = self.modeltime/2e9
            self.tf = self.f_time*2e9 #Volcanically active period [yr] (2e9 from Head et al., 2020)
            self.dt_ice = self.f_time*2e8*yr #Timestep at which ice amounts are saved (larger than dt to save memory) [s]
        self.t1 = t1 #Extra model run time [yr]
        self.dt = dt #2e0*yr #Smallest timestep [s]
        self.t = np.arange(0, (self.tf+self.t1)*yr+self.dt, self.dt) #Overall time array [s]
        self.t_ice = np.arange(self.dt_ice, self.tf*yr+self.dt_ice, self.dt_ice) #Time array for ice reservoirs [s]
        
        if (self.save == True):
            #Make new directory or update existing directory
            self.path = 'VAMPIRE_tf'+str(self.tf)
            path_num = 0
            if (self.overwrite == False):
                while (os.path.exists(self.path) == True):
                    self.path = 'VAMPIRE_tf'+str(self.tf)+'_'+str(path_num)
                    path_num += 1
            elif ((self.overwrite == True) & (os.path.exists(self.path) == True)):
                shutil.rmtree(self.path)
            os.mkdir(self.path)
        
        #Eruption Frequency Parameters (SFD)
        if (self.PowLaw):
            #Power law distribution
            if (self.modeltime == 2e9):
                #9-13-23 Changed from 5.45e5 to 5e5 along with slope (a_er) as this matches total and mean gas mass better
                self.n_er = int(5e5) #number of eruptions
            else:
                self.n_er = int(self.f_time*5.45e5) #number of eruptions (50,000 from Head et al., 2020)
            #9-13-23 Changed slope (a_er) from 3.2 to 3 as this matches total and mean gas mass better
            a_er = 3 #slope of power law
            xmin_er = 6e10 #Minimum erupted mass [kg] Based on smallest erupted magma volume of 10 km3 (Wilson and Head, 2017)
            # generates random variates of power law distribution
            self.M_er = powerlaw.Power_Law(xmin=xmin_er, parameters=[a_er]).generate_random(self.n_er)
        else:
            #Exponential distribution
            if (self.modeltime == 2e9):
                self.n_er = int(5e4) #number of eruptions (50,000 from Head et al., 2020)
            else:
                self.n_er = int(self.f_time*5e4) #number of eruptions (50,000 from Head et al., 2020)
            self.mean_er = 1.14e12 #mean gas mass released by an individual eruption [kg] (From Head et al., 2020)
                              #mean_er=1.14e12 gives mean of 1.2e12 if Mmin=6e10 
            Mmin, Mmax = 6e10, 1e14 #[kg] Based on smallest erupted magma volume of 10 km3 (Wilson and Head, 2017)
            self.M_er = np.zeros((0,))    # empty for now
            while self.M_er.shape[0] < self.n_er:
                s = np.random.exponential(self.mean_er ,self.n_er)
                accepted = s[(s >= Mmin) & (s <= Mmax)]
                self.M_er = np.concatenate((self.M_er, accepted), axis=0)
            self.M_er = self.M_er[:self.n_er]
        
        #Eruption Timeline Parameters (PDF)
        #Peak volcanic flux at ~3.6 Ga (Whitten and Head, 2015)
        if (self.modeltime == 2e9):
            PDF = Volcanic_PDF(a=0, b=2e9 , name='Volcanic_PDF', shapes='sigma, gamma')
            sigma = 0.05e9 #0.05e7 #0.05e9
            gamma = 0.26e9 #0.26e7 #0.26e9
        else:
            PDF = Volcanic_PDF(a=0, b=self.f_time*2e9 , name='Volcanic_PDF', shapes='sigma, gamma')
            sigma = self.f_time*0.05e9
            gamma = self.f_time*0.26e9
        self.t_er = PDF.rvs(sigma, gamma, size=self.n_er)
        self.t_er = dt * np.around(self.t_er/(self.dt/yr), decimals=0) #Time of each eruption rounded to the timestep [s]
        self.nw_er = 1133/3400 #Water mass fraction of erupted mass (Rutherford et al. (2017) via Head et al. (2020))
        self.nco_er = 1395/3400 #CO fraction of erupted mass (Rutherford et al. (2017) via Head et al. (2020))

        #Escape Model Parameters
        self.T0 = T0 #Exobase Temperature for a non-collisional atmosphere [K]
        self.T1 = T1 #Exobase Temperature for a collisional atmosphere [K]
        
        #Atmospheric Constituents (except water) from Rutherford et al. (2017): [CO, SO2, H2S, COS, F]
        self.mol_mass = np.array([64.1, 34.1, 60.1, 19]) #Molecular masses of atmopsheric constituents [kg.kmol-1]
        self.f_mol = np.array([327/327, 168/327, 327/327, 50/327]) #Mass fractions relative to CO
        self.c_mol = np.sum(self.mol_mass*self.f_mol)/np.sum(self.f_mol) #Useful constant for calculating mean molecular mass of atmosphere
        self.m_mean = 31.4*u
        
        #Load Diviner Max Temperature Maps Made by Dr. Margaret Landis
        data_n1 = sio.loadmat(self.MapsN)
        data_s1 = sio.loadmat(self.MapsS)

        Tmax_n1 = data_n1['T_bol_array']
        Tmax_s1 = data_s1['T_bol_array']
        Tmax_n1 = np.nan_to_num(Tmax_n1)
        Tmax_s1 = np.nan_to_num(Tmax_s1)

        dim = 6000
        Tmax_n1 = Tmax_n1.reshape(dim, dim)
        Tmax_s1 = Tmax_s1.reshape(dim, dim)
        Tmax1 = np.concatenate([Tmax_s1, Tmax_n1])
        Tmax1[Tmax1==0] = np.nan
        
        T_s1 = Tmax1[:int(Tmax1.shape[0]/2),:]
        T_n1 = Tmax1[int(Tmax1.shape[0]/2):,:]
        
        zoom = 7680/6000
        T_s1_zoom = scipy.ndimage.zoom(T_s1, zoom, order=0)
        T_n1_zoom = scipy.ndimage.zoom(T_n1, zoom, order=0)
        self.Tmax1_zoom = np.concatenate([T_s1_zoom, T_n1_zoom])
        
        #Initialize Coldtrap temperatures and areas
        self.fs = 0.75 #Fraction of present day solar flux
        self.Tmax0 = self.Tmax1_zoom*(self.fs**0.25)
        #Tmax0[Tmax==400] = np.nan
        self.T = self.Tmax0.flatten()

        #Mask areas outside of map
        M_max = 1e14 #Maximum atmospheric mass likely to be achieved [kg]
        P_max = g*M_max/SA #Maximum Surface Pressure likely to be achieved [Pa]
        p_sat_max = satVaporPressureIce(self.T)

        self.idx_m = np.where((np.isnan(self.T)==False) & (p_sat_max<self.nw_er*P_max))
        mT = self.T[self.idx_m]

        #Histogram Temperatures to lessen the number of calculations necessary
        num_bin = 1000
        self.hist_T, bin_edges = np.histogram(mT[~np.isnan(mT)], num_bin)
        T_bin = bin_edges[:-1] + np.diff(bin_edges)/2
        self.idx_hist = np.digitize(mT, bin_edges) - 1
        self.idx_hist[np.argmax(self.idx_hist)] -= 1

        #print ('self.idx_hist:', np.amax(self.idx_hist))

        A0 = (250**2) #Default diviner footprint area [m2]
        A_bin = np.ones_like(T_bin)*A0*self.hist_T #Area of each temperature bin [m2]
        
        #Useful calculations to optimize for speed
        self.p_sat = satVaporPressureIce(T_bin) #Saturation vapor pressure [Pa]
        self.rho_sat = self.p_sat*mw/(Rg*T_bin)
        self.f_D = diffusionCoeff_CO_factor(T_bin) #Useful factor for calculating diffusion coefficient
        self.f_visc = kineViscosity_CO_factor(T_bin) #Useful factor for calculating kinematic viscosity
        self.f = A_bin * (101325*self.f_D*u_star)/(30*self.f_visc)
        self.f1 = self.f*(mw*g/(Rg*T_bin*SA))
        self.f2 = self.f*self.rho_sat
        
        #Mass arrays
        self.M_atm = np.zeros_like(self.t) #Total atmospheric mass over time
        self.M_co = np.zeros_like(self.t) #Total atmospheric CO mass over time
        self.M_w = np.zeros_like(self.t) #Total atmospheric water mass over time
        self.rate_w = np.zeros_like(self.t) #Rate of atmospheric water loss due to ice accumulation over time
        self.rate_PDw = np.zeros_like(self.t) #Rate of atmospheric water loss due to photodissociation over time
        self.M_ice = np.zeros([self.t_ice.size, self.T.size]) #Mass of ice in each cold trap over time
        self.M_ice[:,np.isnan(self.T)] = np.nan
        self.M_ice_fine = np.zeros([T_bin.size]) #Keeps track of the ice mass in each reservoir for the fine timestep, dt
        self.rate_ice = np.zeros([T_bin.size]) #np.zeros_like(mT) #Ice growth rate in each ice reservoir
        #self.coll = np.zeros_like(self.t) #Is the atmosphere collisional? #####MAYBE REMOVE AS IT IS UNNECESSARY

        #Account for Nightside accumulation of water ice
        self.f_night = NightSideFrac() #Function to calculate fraction of water vapor available in atmo (not sequestered)

        #Account for escape of H2O and CO due to photodissociation in the exobase region
        self.PDw = PhotoDisRate_H2O()
        self.eff_PDw = 0.5 #Efficiency of photodissociation of H2O as a loss mechanism
        self.PDco = PhotoDisRate_CO()
        self.eff_PDco = 0.5 #Efficiency of photodissociationof CO as a loss mechanism

        #Account for escape due to solar wind induced sputtering
        self.SP = SputRate()
        self.eff_SP = 0.33 #0.09 #Efficiency of sputtering depending on a magnetic field of 5 (0.09) or 0.5 (0.33) muT.
                            #Garrick-Bethell et al., 2019 via Tucker et al., 2021
        
        
    def run(self):
        #Run model and store output
        start = time.time()
        
        #roadus blockus Brian Hood for i in tqdm(range(1, self.t.size)):
            if (self.M_w[i-1] >= 333235.29):
                frac_night = self.f_night(np.log(self.M_w[i-1]))
            else:
                frac_night = self.f_night(np.log(333235.29))

            if (self.M_w[i-1]+self.M_co[i-1] > self.M_atm[i-1]):
                #This can happen if a significant amount of H2O was released from ice reservoirs in the previous timestep
                #We will check if the H2O vapor content is enough to form a collisional atmosphere after calculating 
                #the loss rate in several lines
                self.M_atm[i-1] = self.M_w[i-1] + self.M_co[i-1]
                nw = (self.M_w[i-1]*frac_night)/self.M_atm[i-1]
                nco = self.M_co[i-1]/self.M_atm[i-1]
                m = (mw/Na)*nw + (mco/Na)*nco #Molecular mass of atmosphere [kg] Equal to mw*1e3*nw*u + mco*1e3*nco*u
            elif (self.M_atm[i-1] > 0):
                #Calculate atmospheric water and CO mass ratio (excluding frost on the nightside)
                nw = (self.M_w[i-1]*frac_night)/self.M_atm[i-1]
                nco = self.M_co[i-1]/self.M_atm[i-1]
                #Calculate mean molecular mass of atmosphere [kg]
                m = (self.c_mol + ((mw*1e3) - self.c_mol)*nw + ((mco*1e3) - self.c_mol)*nco) * u
                #Make sure m doesn't become negative
                #This may be redundant after making sure the water mass can't be greater than erupted mass fraction
                if (m < 0): 
                    self.M_atm[i-1] = self.M_w[i-1] + self.M_co[i-1]
                    nw = (self.M_w[i-1]*frac_night)/self.M_atm[i-1]
                    nco = self.M_co[i-1]/self.M_atm[i-1]
                    m = (mw/Na)*nw + (mco/Na)*nco
            else:
                nw = 0
                nco = 0
                m = self.m_mean #31.4*u

            #Calculate rate of atmospheric mass loss to thermal escape (if there is any atmosphere)
            if (self.M_atm[i-1] > 0):
                rate_loss, Pe, P0 = LossRate_Exo((self.M_atm[i-1] + self.M_w[i-1]*(frac_night - 1)), T0=self.T0, T1=self.T1, m=m)
            else:
                rate_loss, Pe, P0 = 0, 1, 0

            #Calculate rate of ice growth in ice reservoirs
            if (P0 >= Pe): #When atmosphere is collisional at surface
                #print ('M_atm:', M_atm[i-1])
                #coll[i] = 1 #####MAYBE REMOVE AS IT IS UNNECESSARY

                #Don't bother computing the ablation rate of points with zero ice mass that are too hot to gain ice
                truth = np.logical_and( self.M_ice_fine<=0, self.p_sat>=nw*P0 )

                idx_calc = np.where(np.logical_not(truth))

                self.rate_ice[:] = 0

                #Rate of ice growth in each ice reservoir
                self.rate_ice[idx_calc] = self.f1[idx_calc]*(self.M_w[i-1]*frac_night) - self.f2[idx_calc]

                #Update ice mass in each ice reservoir
                self.M_ice_fine[idx_calc] = self.M_ice_fine[idx_calc] + self.rate_ice[idx_calc]*self.dt

                idx_ice = np.where(self.M_ice_fine[idx_calc] < 0)

                if (idx_ice[0].size): #This should help fix the mass balance problem (negative ice mass)
                    self.rate_ice[idx_calc][idx_ice] -=  self.M_ice_fine[idx_calc][idx_ice]/self.dt
                    self.M_ice_fine[idx_calc][idx_ice] = 0

                #Calculate rate of atmospheric water vapor loss due to ice
                self.rate_w[i] = -np.sum(self.rate_ice[idx_calc])

            else:
                #print ('M_atm:', M_atm[i-1])
                self.rate_ice[:] = 0
                self.rate_w[i] = 0

            #Save ice map to coarse time array
            if (np.any(self.t[i] == self.t_ice)):
                idx_coarse = np.where(self.t[i] == self.t_ice)
                self.M_ice[idx_coarse,self.idx_m] = self.M_ice_fine[self.idx_hist]/self.hist_T[self.idx_hist]

            #Calculate loss rate of atmosphere due to sputtering
            if (self.SP_on == True):
                if (self.M_atm[i-1] >= 1e7):
                    rate_SP = self.eff_SP * self.SP(np.log(self.M_atm[i-1]))
                else:
                    rate_SP = 0
            else:
                rate_SP = 0

            #Calculate loss rate of atmospheric water vapor and CO due to photodissociation
            if (self.PD_on == True):
                if (self.M_w[i-1] >= 4e7):
                    #roadus blockus Brian Hood self.rate_PDw[i] = self.eff_PDw * self.PDw(np.log(self.M_w[i-1]*frac_night))
                    if (self.M_w[i-1] + self.rate_w[i]*self.dt - nw*rate_loss*self.dt - nw*rate_SP*self.dt < \
                        self.rate_PDw[i]*self.dt):
                        self.rate_PDw[i] = (self.M_w[i-1] + self.rate_w[i]*self.dt - nw*rate_loss*self.dt - \
                                            nw*rate_SP*self.dt)/self.dt
                else:
                    self.rate_PDw[i] = 0
                if (self.M_co[i-1] >= 4e7):
                    rate_PDco = self.eff_PDco * self.PDco(np.log(self.M_co[i-1]))
                    if (self.M_co[i-1]  - nco*rate_loss*self.dt - nco*rate_SP*self.dt < rate_PDco*self.dt):
                        rate_PDco = (self.M_co[i-1]  - nco*rate_loss*self.dt - nco*rate_SP*self.dt)/self.dt
                else:
                    rate_PDco = 0
            else:
                self.rate_PDw[i] = 0
                rate_PDco = 0

            if (np.any(self.t[i] == self.t_er)): #If one or more eruptions occur in this timestep
                #Update total atmospheric mass
                idx_er = np.where(self.t[i] == self.t_er)
                # RB self.M_atm[i] = self.M_atm[i-1] - rate_loss*self.dt + np.sum(self.M_er[idx_er]) + self.rate_w[i]*self.dt - \
                                                        self.rate_PDw[i]*self.dt - rate_PDco*self.dt - rate_SP*self.dt

                #Update total atmospheric water vapor mass
                self.M_w[i] = self.M_w[i-1] + self.rate_w[i]*self.dt - nw*rate_loss*self.dt + \
                              self.nw_er*np.sum(self.M_er[idx_er]) - self.rate_PDw[i]*self.dt - nw*rate_SP*self.dt

                #Update total atmospheric CO mass
                self.M_co[i] = self.M_co[i-1]  - nco*rate_loss*self.dt + self.nco_er*np.sum(self.M_er[idx_er]) - \
                               rate_PDco*self.dt - nco*rate_SP*self.dt

            else: #If no eruptions occur in this timestep
                #Update total atmospheric mass
                self.M_atm[i] = self.M_atm[i-1] - rate_loss*self.dt + self.rate_w[i]*self.dt - self.rate_PDw[i]*self.dt - \
                                rate_PDco*self.dt - rate_SP*self.dt

                #Update total atmospheric water vapor mass
                self.M_w[i] = self.M_w[i-1] + self.rate_w[i]*self.dt - nw*rate_loss*self.dt - self.rate_PDw[i]*self.dt - \
                              nw*rate_SP*self.dt

                #Update total atmospheric CO mass
                self.M_co[i] = self.M_co[i-1]  - nco*rate_loss*self.dt - rate_PDco*self.dt - nco*rate_SP*self.dt

            #Make sure atmosphere can't have negative mass
            if (self.M_atm[i] < 0):
                self.M_atm[i] = 0

            #Make sure atmospheric can't have negative water mass
            if (self.M_w[i] < 0):
                self.M_w[i] = 0

            #Make sure atmospheric can't have negative CO mass
            if (self.M_co[i] < 0):
                self.M_co[i] = 0

            #####MAYBE REMOVE AS IT IS UNNECESSARY
            #Make sure water mass can't be greater than erupted mass fraction
            #Hypothetically, this could happen if all the stored ice suddenly ablated, 
            #but the temps are constant so this shouldn't happen in this model
            #if (M_w[i] > nw_er*M_atm[i]):
            #    #M_w[i] = nw_er*M_atm[i]
            #    print ('This Happened! t =', t[i]/yr)

        
        #Reshape M_ice array to match original Diviner maps
        #M_ice = M_ice.reshape(t_ice.size, Tmax1_zoom.shape[0], Tmax1_zoom.shape[1])
        self.M_ice = self.M_ice.reshape(self.t_ice.size, self.Tmax0.shape[0], self.Tmax0.shape[1])
        
        if (self.save):
            self.save_model()
            
        end = time.time()
        print ('Elapsed Time:', (end-start), '(s), ', (end-start)/60, '(min)', (end-start)/60/60, '(hr)')
    
    def save_model(self):
        #Make ReadMe File
        f = open(self.path+'/ReadMe_'+self.path+'.txt','w+')
        f.write('Volcanic Atmosphere Model Predicting Ice Rates and Escape (VAMPIRE)\n'+\
        'Model Parameters:\n\n'+\
        'tf = '+str(self.tf)+' #Volcanically active period [yr] (2e9 From Head et al., 2020)\n'+\
        't1 = '+str(self.t1)+' #Extra model run time [yr]\n'+\
        'dt = '+str(self.dt/yr)+' #Smallest timestep [yr]\n'+\
        'dt_ice = '+str(self.dt_ice/yr)+' #Timestep at which ice amounts are saved [yr]\n'+\
        'n_er = '+str(self.n_er)+' #number of eruptions (From Head et al., 2020)\n'+\
        'mean_er = '+str(self.mean_er)+' #mean gas mass released by an individual eruption [kg] (From Head et al., 2020)\n'+\
        'nw_er = '+str(self.nw_er)+' #Water mass fraction of total atmospheric mass (Rutherford et al. (2017) via Head et al. \
                                      (2020))\n'+\
        'nco_er = '+str(self.nco_er)+' #CO fraction of erupted mass (Rutherford et al. (2017) via Head et al. (2020))\n'+\
        'T0 = '+str(self.T0)+' #Exobase Temperature for a non-collisional atmosphere [K]\n'+\
        'T1 = '+str(self.T1)+' #Exobase Temperature for a collisional atmosphere [K]\n'+\
        'm_mean = '+str(self.m_mean/u)+' #Mean molecular mass of atmosphere released in an eruption [amu]\n'+\
        'fs = '+str(self.fs)+' #Fraction of present day solar flux\n'+\
        'PD_on = '+str(self.PD_on)+' #Allow H2O and CO to be lost due to photodissociation in the exobase region?\n'+\
        'eff_PDw = '+str(self.eff_PDw)+' #Efficiency of photodissociation of H2O as a loss mechanism\n'+\
        'eff_PDco = '+str(self.eff_PDco)+' #Efficiency of photodissociation of CO as a loss mechanism\n'+\
        'SP_on = '+str(self.SP_on)+' #Allow atmospheric escape due to solar wind sputtering?\n'+\
        'eff_SP = '+str(self.eff_SP)+' #Efficiency of sputtering depending on a magnetic field of 5 (0.09) or 0.5 (0.33) muT\n'+\
        'PowLaw = '+str(self.PowLaw)+' #Use a power law distribution for erupted gas mass, or use the default exponential \
                                        distribution?\n'+\
        'z0 = '+str(z0)+' #Surface roughness of ice [m]\n'+\
        'z_vik = '+str(z_vik)+' #Height at which lander wind measurements were taken [m]\n'+\
        'u_wind = '+str(u_wind)+' #Wind speed at z_vik [m.s-1]\n'+\
        'u_star = '+str(u_star)+' #Wind friction velocity [m.s-1]\n'+\
        'MapsN = '+self.MapsN+' #Northern hemisphere Diviner maps file used'+\
        'MapsS = '+self.MapsS+' #Southern hemisphere Diviner maps file used')

        f.close()

        #Save arrays
        np.save(self.path+'/t', self.t/yr)
        np.save(self.path+'/t_ice', self.t_ice/yr)
        np.save(self.path+'/M_atm', self.M_atm)
        np.save(self.path+'/M_w', self.M_w)
        np.save(self.path+'/M_co', self.M_co)
        np.save(self.path+'/M_ice', self.M_ice)
        np.save(self.path+'/M_er', self.M_er) #Mass of each eruption [kg]
        np.save(self.path+'/rate_w', self.rate_w)
        np.save(self.path+'/rate_PDw', self.rate_PDw)
    
    def plot_atm(self, save=False, step=2):
        #Plot atmospheric mass vs time and water loss rates vs time
        
        fig, ax = plt.subplots(2,1, figsize=(9,8))
        
        #A
        #########################
        ax[0].set_xlim(4.02,1.98)
        if (self.modeltime == 2e9):
            ax[0].plot(np.abs(self.t[0::step]/yr/1e9 - 4), self.M_atm[0::step]/1e12, label='Total Atmo')#, color='indigo')
            ax[0].plot(np.abs(self.t[0::step]/yr/1e9 - 4), self.M_w[0::step]/1e12, label='H$_2$O')#, color='C1')
        else:
            ax[0].plot(np.abs(self.t[0::step]/yr/(self.f_time*1e9)- 4), self.M_atm[0::step]/1e12, label='Total Atmo')#, color='indigo')
            ax[0].plot(np.abs(self.t[0::step]/yr/(self.f_time*1e9) - 4), self.M_w[0::step]/1e12, label='H$_2$O')#, color='C1')
        ax[0].set_ylabel('Vapor Mass (10$^{12}$ kg)', fontsize=16)
        ax[0].grid()
        ax[0].legend(loc=2)
        
        #idx1 = int(1.390746e8)
        #idx2 = int(1.390813e8)
        if (self.modeltime == 2e9):
            idx1 = int(2.26484809e8) #newmap
            idx2 = int(2.26490680e8) #newmap
            x1, x2, y1, y2 = np.abs(self.t[idx1]/yr/1e9 - 4), np.abs(self.t[idx2]/yr/1e9 - 4), -0.1, 1.6
        else:
            idx1 = int(self.f_time*2.26484809e8)
            idx2 = int(self.f_time*2.26490680e8)
            x1, x2, y1, y2 = np.abs(self.t[idx1]/yr/(self.f_time*1e9) - 4), np.abs(self.t[idx2]/yr/(self.f_time*1e9) - 4), -0.1, 1.6
        
        #x1, x2, y1, y2 = np.abs(self.t[idx1]/yr/1e9 - 4), np.abs(self.t[idx2]/yr/1e9 - 4), -0.1, 1.6
        ax2 = plt.axes([0,0,1,1], label='ax2')
        # Manually set the position and relative size of the inset axes within ax1
        ip0 = InsetPosition(ax[0], [0.48,0.48,0.5,0.5])
        ax2.set_axes_locator(ip0)
        # Mark the region corresponding to the inset axes on ax1 and draw lines
        # in grey linking the two axes.
        mark_inset(ax[0], ax2, loc1=2, loc2=4, fc="none", ec='0.0', zorder = 3)
        
        if (self.modeltime == 2e9):
            ax2.plot(np.abs(self.t[idx1:idx2]/yr/1e9 - 4), self.M_atm[idx1:idx2]/1e12, linewidth=1.5)
            ax2.plot(np.abs(self.t[idx1:idx2]/yr/1e9 - 4),self.M_w[idx1:idx2]/1e12, linewidth=1.5)
        else:
            ax2.plot(np.abs(self.t[idx1:idx2]/yr/(self.f_time*1e9) - 4), self.M_atm[idx1:idx2]/1e12, linewidth=1.5)
            ax2.plot(np.abs(self.t[idx1:idx2]/yr/(self.f_time*1e9) - 4),self.M_w[idx1:idx2]/1e12, linewidth=1.5)
        ax2.set_xlim(x1, x2)
        ax2.set_ylim(y1, y2)
        ax2.set_xticks([])
        
        fontprops = fm.FontProperties(size=13)
        scalebar = AnchoredSizeBar(ax2.transData,
                                   0.000005, '5000 yr', 'upper left', 
                                   pad=0.2,
                                   color='k',
                                   frameon=False,
                                   size_vertical=0.078,
                                   fontproperties=fontprops)
        
        ax2.add_artist(scalebar)
        #########################
        
        
        #B
        #########################
        ax[1].set_xlim(4.02,1.98)
        if (self.modeltime == 2e9):
            ax[1].plot(np.abs(self.t[0::step]/yr/1e9 - 4), -self.rate_w[0::step], label='Ice', alpha=0.7, color='indigo')
            ax[1].plot(np.abs(self.t[0::step]/yr/1e9 - 4), self.rate_PDw[0::step], label='PD', alpha=0.7, color='gold')
            ax[1].set_xlabel('Time (Ga)', fontsize=16)
        else:
            ax[1].plot(np.abs(self.t[0::step]/yr/(self.f_time*1e9) - 4), -self.rate_w[0::step], label='Ice', alpha=0.7, color='indigo')
            ax[1].plot(np.abs(self.t[0::step]/yr/(self.f_time*1e9) - 4), self.rate_PDw[0::step], label='PD', alpha=0.7, color='gold')
        ax[1].set_ylabel('Water Removal Rate (kg/s)', fontsize=16)
        ax[1].grid()
        ax[1].legend(loc=2)
        
        ax3 = plt.axes([0,0,1,1], label='ax3')
        # Manually set the position and relative size of the inset axes within ax1
        ip1 = InsetPosition(ax[1], [0.48,0.48,0.5,0.5])
        ax3.set_axes_locator(ip1)
        # Mark the region corresponding to the inset axes on ax1 and draw lines
        # in grey linking the two axes.
        mark_inset(ax[1], ax3, loc1=2, loc2=4, fc="none", ec='0.0', zorder = 3)
        
        if (self.modeltime == 2e9):
            ax3.plot(np.abs(self.t[idx1:idx2]/yr/1e9 - 4), -self.rate_w[idx1:idx2], linewidth=1.5, alpha=0.7, color='indigo')
            ax3.plot(np.abs(self.t[idx1:idx2]/yr/1e9 - 4), self.rate_PDw[idx1:idx2], linewidth=1.5, alpha=0.7, color='gold')
        else:
            ax3.plot(np.abs(self.t[idx1:idx2]/yr/(self.f_time*1e9) - 4), -self.rate_w[idx1:idx2], linewidth=1.5, alpha=0.7, color='indigo')
            ax3.plot(np.abs(self.t[idx1:idx2]/yr/(self.f_time*1e9)- 4), self.rate_PDw[idx1:idx2], linewidth=1.5, alpha=0.7, color='gold')
        y1, y2 = -20, 420
        ax3.set_xlim(x1, x2)
        ax3.set_ylim(y1, y2)
        ax3.set_xticks([])
        
        fontprops = fm.FontProperties(size=13)
        scalebar = AnchoredSizeBar(ax3.transData,
                                   0.000005, '5000 yr', 'upper left', 
                                   pad=0.2,
                                   color='k',
                                   frameon=False,
                                   size_vertical=20,
                                   fontproperties=fontprops)
        
        ax3.add_artist(scalebar)
        #########################
        
        if (save):
            if (self.PowLaw):
                plt.savefig(self.path+'/VAMPIRE_tf'+str("{:.0e}".format(self.tf))+'_Atmo_Mass_Time_Pow', dpi=300, transparent=False)
            else:
                plt.savefig(self.path+'/VAMPIRE_tf'+str("{:.0e}".format(self.tf))+'_Atmo_Mass_Time_Exp', dpi=300, transparent=False)
        
        plt.show()
        
    def plot_maps(self, save=False):
        #####################################
        ##### Big Six Pack Paper Figure #####
        #####################################
        
        #Process Data and Load LOLA Maps
        M_ice = self.M_ice[-1,:,:]
        M_ice[M_ice==0] = np.nan
        
        M_s = M_ice[:int(M_ice.shape[0]/2),:]
        M_n = M_ice[int(M_ice.shape[0]/2):,:]
        
        #LOLA Maps
        input_filename_s = "LDEM_60S_240M.IMG"
        input_filename_n = "LDEM_60N_240M.IMG"
        shape = (7760, 7760) # matrix size
        dtype = np.dtype('<i2') # little-endian unsigned integer (16bit)
        
        fid_s = open(input_filename_s, 'rb')
        data_s = np.fromfile(fid_s, dtype) * 0.5
        image_s = data_s.reshape(shape)
        fid_n = open(input_filename_n, 'rb')
        data_n = np.fromfile(fid_n, dtype) * 0.5
        image_n = data_n.reshape(shape)
        
        #LOLA Maps
        input_filename_s80 = "LDEM_80S_80M.IMG"
        input_filename_n80 = "LDEM_80N_80M.IMG"
        shape80 = (7600, 7600) # matrix size
        dtype = np.dtype('<i2') # little-endian unsigned integer (16bit)
        
        fid_s80 = open(input_filename_s80, 'rb')
        data_s80 = np.fromfile(fid_s80, dtype) * 0.5
        image_s80 = data_s80.reshape(shape80)
        fid_n80 = open(input_filename_n80, 'rb')
        data_n80 = np.fromfile(fid_n80, dtype) * 0.5
        image_n80 = data_n80.reshape(shape80)
        
        M_s80 = M_s[int(M_s.shape[0]/3):int(2*M_s.shape[0]/3), int(M_s.shape[1]/3):int(2*M_s.shape[1]/3)]
        M_n80 = M_n[int(M_n.shape[0]/3):int(2*M_n.shape[0]/3), int(M_n.shape[1]/3):int(2*M_n.shape[1]/3)]
        
        #Determine ice mass assuming 4 Ga of sublimation to space
        A0 = (250**2) #Default diviner footprint area [m2]
        p_sat = satVaporPressureIce(self.Tmax0)
        rate_sub = p_sat*np.sqrt(mw/(2*np.pi*Rg*self.Tmax0))
        
        rate_sub[M_ice[:,:]==0] = np.nan
        M_ice_sub = M_ice[:,:] - (rate_sub*A0*4e9*yr)
        M_ice_sub[M_ice_sub<=0] = np.nan
        
        M_ice_sub_s = M_ice_sub[:int(M_ice.shape[0]/2),:]
        M_ice_sub_n = M_ice_sub[int(M_ice.shape[0]/2):,:]
        
        M_ice_sub_s80 = M_ice_sub_s[int(M_s.shape[0]/3):int(2*M_s.shape[0]/3), int(M_s.shape[1]/3):int(2*M_s.shape[1]/3)]
        M_ice_sub_n80 = M_ice_sub_n[int(M_n.shape[0]/3):int(2*M_n.shape[0]/3), int(M_n.shape[1]/3):int(2*M_n.shape[1]/3)]
        
        
        ##### Start Plotting Figure #####
        fracx = 7759/7679 #1.011
        fracy = 7759/7679 #1.011 #1.034
        xs,ys = np.meshgrid(fracx*np.flip(np.arange(-7680,0)), fracy*np.flip(np.arange(-7680,0)))
        xs_im,ys_im = np.meshgrid(np.flip(np.arange(-7760,0)), np.flip(np.arange(-7760,0)))
        x,y = np.meshgrid(fracx*np.arange(0,7680), fracy*np.arange(0,7680))

        A0 = (250**2) #Default diviner footprint area [m2]
        
        fig, ax = plt.subplots(3, 2, figsize=[10,15.8])
        
        #A
        ##########################
        ########South Pole########
        ls = LightSource(azdeg=-90, altdeg=45)
        #im = ax[0].imshow(ls.hillshade(np.flip(image_s, 0), vert_exag=0.01, fraction=0.1), cmap = "gray")
        im = ax[0,0].pcolormesh(xs_im, ys_im, ls.hillshade(np.flip(image_s, 0), vert_exag=0.01, fraction=0.1), cmap = "gray")
        patch = patches.Circle((int(-image_s.shape[0]/2),int(-image_s.shape[1]/2)), radius=int(image_s.shape[1]/2), \
                               transform=ax[0,0].transData)
        ax[0,0].text(-0.03, 1, 'a', transform=ax[0,0].transAxes, 
                    size=18, weight='bold')
        im.set_clip_path(patch)
        
        #m = Basemap(projection='npstere',boundinglat=60,lon_0=180,resolution=None, ax=ax[0], round=True, \
        #            rsphere=(7590/(np.pi/3)))
        #m.drawparallels(np.arange(60.,81.,10.))
        ##m = Basemap(projection='spstere',boundinglat=-60,lon_0=180,resolution=None, ax=ax[0,0], round=True, \
        ##            rsphere=(7590/(np.pi/3)))
        ##m.drawparallels(np.arange(-80.,-59.,10.), ax=ax[0,0])
        ##m.drawmeridians(np.arange(-135.,181.,45.),labels=[1,1,1,1], labelstyle=None, ax=ax[0,0])
        
        norm = colors.Normalize(vmin=np.nanmin(M_ice[:,:].T/A0/rho_ice), \
                                vmax=np.nanmax(M_ice[:,:].T/A0/rho_ice))
        
        am = ax[0,0].pcolormesh(xs, ys, M_s[:,:].T/A0/rho_ice, edgecolor='none', norm=norm)
        #am.set_clip_path(patch)
        ##########################
        ##########################
        
        #B
        ##########################
        ########North Pole########
        ls = LightSource(azdeg=-90, altdeg=45)
        im = ax[0,1].imshow(ls.hillshade(np.flip(image_n, 0), vert_exag=0.01, fraction=0.1), cmap = "gray")
        patch = patches.Circle((int(image_s.shape[0]/2),int(image_s.shape[1]/2)), radius=int(image_s.shape[1]/2), \
                               transform=ax[0,1].transData)
        ax[0,1].text(-0.03, 1, 'b', transform=ax[0,1].transAxes, 
                    size=18, weight='bold')
        im.set_clip_path(patch)
        
        ##m = Basemap(projection='npstere',boundinglat=60,lon_0=0,resolution=None, ax=ax[0,1], round=True, \
        ##            rsphere=(7590/(np.pi/3)))
        ##m.drawparallels(np.arange(60.,81.,10.), ax=ax[0,1])
        ##m.drawmeridians(np.arange(-135.,181.,45.),labels=[1,1,1,1], labelstyle=None, ax=ax[0,1])
        
        ax[0,1].pcolormesh(x, y, M_n[:,:].T/A0/rho_ice, edgecolor='none', norm=norm)
        ##########################
        ##########################
        
        ax[0,0].set_aspect('equal')
        ax[0,1].set_aspect('equal')
        
        
        
        
        fracx = 7599/2559 #1.011
        fracy = 7599/2559 #1.011 #1.034
        xs,ys = np.meshgrid(fracx*np.flip(np.arange(-2560,0)), fracy*np.flip(np.arange(-2560,0)))
        xs_im,ys_im = np.meshgrid(np.flip(np.arange(-7600,0)), np.flip(np.arange(-7600,0)))
        x,y = np.meshgrid(fracx*np.arange(0,2560), fracy*np.arange(0,2560))
        
        #C
        ##########################
        ######South Pole 80#######
        ls = LightSource(azdeg=-90, altdeg=45)
        #im = ax[1,0].imshow(ls.hillshade(np.flip(image_s80, 0), vert_exag=0.01, fraction=0.1), cmap = "gray")
        im = ax[1,0].pcolormesh(xs_im, ys_im, ls.hillshade(np.flip(image_s80, 0), vert_exag=0.01, fraction=0.1), cmap = "gray")
        patch = patches.Circle((int(-image_s80.shape[0]/2),int(-image_s80.shape[1]/2)), radius=int(image_s80.shape[1]/2), \
                               transform=ax[1,0].transData)
        #patch = patches.Circle((int(image_s80.shape[0]/2),int(image_s80.shape[1]/2)), radius=int(image_s80.shape[1]/2), \
        #                       transform=ax[1,0].transData)
        ax[1,0].text(-0.03, 1, 'c', transform=ax[1,0].transAxes, 
                    size=18, weight='bold')
        im.set_clip_path(patch)
        
        ##m = Basemap(projection='spstere',boundinglat=-80,lon_0=180,resolution=None, ax=ax[1,0], round=True, \
        ##            rsphere=(7590/(np.pi/9)))
        ##m.drawparallels(np.arange(-85.,-79.,10.), ax=ax[1,0])
        ##m.drawmeridians(np.arange(-135.,181.,45.),labels=[1,1,1,1], labelstyle=None, latmax=85, ax=ax[1,0])
        
        ams = ax[1,0].pcolormesh(xs, ys, M_s80.T/A0/rho_ice, edgecolor='none', norm=norm)
        ams.set_clip_path(patch)
        #ax[1,0].plot(xs[M_s80.T/A0/rho_ice>409], ys[M_s80.T/A0/rho_ice>409], 'o', color='red')
        ##########################
        ##########################
        
        #D
        ##########################
        ######North Pole 80#######
        ls = LightSource(azdeg=-90, altdeg=45)
        im = ax[1,1].imshow(ls.hillshade(np.flip(image_n80, 0), vert_exag=0.01, fraction=0.1), cmap = "gray")
        patch = patches.Circle((int(image_n80.shape[0]/2),int(image_n80.shape[1]/2)), radius=int(image_n80.shape[1]/2), \
                               transform=ax[1,1].transData)
        ax[1,1].text(-0.03, 1, 'd', transform=ax[1,1].transAxes, 
                    size=18, weight='bold')
        im.set_clip_path(patch)
        
        ##m = Basemap(projection='npstere',boundinglat=80,lon_0=0,resolution=None, ax=ax[1,1], round=True, \
        ##            rsphere=(7590/(np.pi/9)))
        ##m.drawparallels(np.arange(80.,86.,5.), ax=ax[1,1])
        ##m.drawmeridians(np.arange(-135.,181.,45.),labels=[1,1,1,1], labelstyle=None, latmax=85, ax=ax[1,1])
        
        amn = ax[1,1].pcolormesh(x, y, M_n80.T/A0/rho_ice, edgecolor='none', norm=norm)
        amn.set_clip_path(patch)
        #ax[1,1].plot(x[M_n80.T/A0/rho_ice>319], y[M_n80.T/A0/rho_ice>319], 'o', color='red')
        ##########################
        ##########################
        
        ax[1,0].set_aspect('equal')
        ax[1,1].set_aspect('equal')
        
        
        
        #E
        ##########################
        ######South Pole 80#######
        ls = LightSource(azdeg=-90, altdeg=45)
        #im = ax[1,0].imshow(ls.hillshade(np.flip(image_s80, 0), vert_exag=0.01, fraction=0.1), cmap = "gray")
        im = ax[2,0].pcolormesh(xs_im, ys_im, ls.hillshade(np.flip(image_s80, 0), vert_exag=0.01, fraction=0.1), cmap = "gray")
        patch = patches.Circle((int(-image_s80.shape[0]/2),int(-image_s80.shape[1]/2)), radius=int(image_s80.shape[1]/2), \
                               transform=ax[2,0].transData)
        #patch = patches.Circle((int(image_s80.shape[0]/2),int(image_s80.shape[1]/2)), radius=int(image_s80.shape[1]/2), \
        #                       transform=ax[1,0].transData)
        ax[2,0].text(-0.03, 1, 'e', transform=ax[2,0].transAxes, 
                    size=18, weight='bold')
        im.set_clip_path(patch)
        
        ##m = Basemap(projection='spstere',boundinglat=-80,lon_0=180,resolution=None, ax=ax[1,0], round=True, \
        ##            rsphere=(7590/(np.pi/9)))
        ##m.drawparallels(np.arange(-85.,-79.,10.), ax=ax[2,0])
        ##m.drawmeridians(np.arange(-135.,181.,45.),labels=[1,1,1,1], labelstyle=None, latmax=85, ax=ax[2,0])
        
        ams = ax[2,0].pcolormesh(xs, ys, M_ice_sub_s80.T/A0/rho_ice, edgecolor='none', norm=norm)
        ams.set_clip_path(patch)
        ##########################
        ##########################
        
        #F
        ##########################
        ######North Pole 80#######
        ls = LightSource(azdeg=-90, altdeg=45)
        im = ax[2,1].imshow(ls.hillshade(np.flip(image_n80, 0), vert_exag=0.01, fraction=0.1), cmap = "gray")
        patch = patches.Circle((int(image_n80.shape[0]/2),int(image_n80.shape[1]/2)), radius=int(image_n80.shape[1]/2), \
                               transform=ax[2,1].transData)
        ax[2,1].text(-0.03, 1, 'f', transform=ax[2,1].transAxes, 
                    size=18, weight='bold')
        im.set_clip_path(patch)
        
        ##m = Basemap(projection='npstere',boundinglat=80,lon_0=0,resolution=None, ax=ax[1,1], round=True, \
        ##            rsphere=(7590/(np.pi/9)))
        ##m.drawparallels(np.arange(80.,86.,5.), ax=ax[2,1])
        ##m.drawmeridians(np.arange(-135.,181.,45.),labels=[1,1,1,1], labelstyle=None, latmax=85, ax=ax[2,1])
        
        amn = ax[2,1].pcolormesh(x, y, M_ice_sub_n80.T/A0/rho_ice, edgecolor='none', norm=norm)
        amn.set_clip_path(patch)
        ##########################
        ##########################
        
        ax[2,0].set_aspect('equal')
        ax[2,1].set_aspect('equal')
        
        p0 = ax[0,0].get_position().get_points().flatten()
        p1 = ax[0,1].get_position().get_points().flatten()
        ax_cbar = fig.add_axes([p0[0], 0.1, p1[2]-p0[0], 0.015])
        cbar = plt.colorbar(am, cax=ax_cbar, orientation='horizontal')
        cbar.set_label('Ice Thickness (m)', fontsize=14)
        #plt.tight_layout(rect=[0, 0.12, 1, 0.96], pad=1.5)
        
        if (save):
            if (self.PowLaw):
                plt.savefig(self.path+'/VAMPIRE_tf'+str("{:.0e}".format(self.tf))+'_Ice_Map_Six_Pack_Pow', dpi=300, transparent=False)
            else:
                plt.savefig(self.path+'/VAMPIRE_tf'+str("{:.0e}".format(self.tf))+'_Ice_Map_Six_Pack_Exp', dpi=300, transparent=False)
                
        plt.show()
    
    
#---------------------------------------------------------------------------
"""

The functions defined below are used by the thermal code.

"""
#---------------------------------------------------------------------------
        
    
class Volcanic_PDF(st.rv_continuous):
    def _pdf(self, x, sigma, gamma):
        center = 0.4e9
        
        if (x >= center):
            y = Voigt(x - center, sigma, gamma)
        elif ((x < center) & (x >= -1e9 + center )):
            y = ( -( (x - center)/1e9 )**2 + 1 )*Voigt(0, sigma, gamma)
        else:
            y = 0
        
        return y/0.9007885521899425
    
def Voigt(x, alpha, gamma):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    sigma = alpha / np.sqrt(2 * np.log(2))
    
    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma/np.sqrt(2*np.pi)

@jit(nopython=True)
def satVaporPressureIce(T):
    #(Buck 1981) originally from (Wexler 1977)
    x = np.exp( -5865.3696/T + 22.241033 + 0.013749042*T - \
                      0.34031775e-4*T**2 + 0.26967687e-7*T**3 + 0.6918651*np.log(T) )
    return x

@jit(nopython=True)
def satVaporPressureIce_Fray(T):
    #Fray and Schmitt (2009)
    e0 = 20.9969665107897
    e1 = 3.72437478271362
    e2 = -13.9205483215524
    e3 = 29.6988765013566
    e4 = -40.1972392635944
    e5 = 29.7880481050215
    e6 = -9.13050963547721
    
    Tt = 273.16 #Triple point temperature [K]
    pt = 611.657 #6.11657e-3 # 611.657 #Triple point pressure [Pa]
        
    n = -(e0 + e1*(T/Tt) + e2*(T/Tt)**2 + e3*(T/Tt)**3 + e4*(T/Tt)**4 + e5*(T/Tt)**5 + e6*(T/Tt)**6)
    
    #p = pt*np.exp( 1.5*np.log(T/Tt) + (1-(T/Tt))*n*(T/Tt) ) #pt*np.exp( 1.5*np.log(T/Tt) + (1-(T/Tt))*n )
    p = pt*(T/Tt)**1.5 * np.exp( (1-(T/Tt))*n )
    
    return p

@jit(nopython=True)
def p_SO2(T):
    #Saturation vapor pressure of SO2 [Pa]
    #Fray and Schmitt (2009), Eq. 4, Table 5
    
    A = [1.560e1, -3.508e3, -9.401e4, 4.152e6, -6.946e7]
    
    p = 1e5*np.exp( A[0] + A[1]/(T) + A[2]/(T**2) + A[3]/(T**3) + A[4]/(T**4) ) #1e5 to convert bar to Pa
    
    return p

@jit(nopython=True)
def p_H2S(T):
    #Saturation vapor pressure of SO2 [Pa]
    #Fray and Schmitt (2009), Eq. 4, Table 5
    
    A = [1.298e1, -2.707e3]
    
    p = 1e5*np.exp( A[0] + A[1]/T ) #1e5 to convert bar to Pa
    
    return p

@jit(nopython=True)
def diffusionCoeff_CO_factor(T):
    #Diffusion coefficient for a binary system of CO and H2O as given by Chen and Othmer (1962)
    Vc_w = 56 #Critical point volume of H2O [cm3.gmol-1]
    Vc_co = 93 #Critical point volume of CO [cm3.gmol-1]
    Tc_w = 647.27 #Critical point temperature of H2O [K]
    Tc_co = 132.91 #Critical point temperature of CO [K]
    
    #T: Temperature [K]
    #P: Pressure [Pa]
    
    x = 0.43 * (T/100.)**(1.81) * ( 1/(mw*1e3) + 1/(mco*1e3) )**(0.5)
    y = (Tc_w*Tc_co/1e4)**(0.1406) * ( (Vc_w/100.)**(0.4) + (Vc_co/100.)**(0.4) )**2
    
    D = (x/y) * 1e-4 #Diffusion coefficient [m2.s-1]
    
    return D

@jit(nopython=True)
def kineViscosity_CO_factor(T):
    #Kinematic viscosity for CO based on the Sutherland Equation
    mu0 = 1.66e-5 #reference dynamic viscosity at T0 [Pa.s]
    T0 = 273.15 #reference temperature [K]
    S = 56.34842079493554
    x = ( (Rg*T)/(mco) ) * mu0 * (T/T0)**(1.5) * (T0 + S)/(T + S)
    return x

def NightSideFrac():
    #Make function to calculate fraction of total atmospheric water vapor content 
    #NOT sequestered in ice on lunar nightside
    #Fitting log(M_w0), so proper use of function is: p(np.log(M_w))
    #mass_atm = [1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16]
    M_w0 = [333235.29411764705, 3332352.941176471, 33323529.411764707, 333235294.11764705, 3332352941.1764708, \
            33323529411.764706, 333235294117.6471, 3332352941176.4707, 33323529411764.707, 333235294117647.06, \
            3332352941176470.5]
    frac_w = [0.8772355393355274, 0.8752357545504654, 0.8725329379508105, 0.8689517380946907, 0.8640948907957137, \
              0.8564480962163382, 0.8452862242684541, 0.8308547929354799, 0.8032833135652397, 0.7701634181777177, \
              0.7160300973235072]
    
    z = np.polyfit(np.log(M_w0), frac_w, 6)
    p = np.poly1d(z)
    
    return p

def PhotoDisRate_H2O():
    """
    Estimate photodissociation rate of H2O based on total mass of water vapor in atmosphere.
    Rate estimated assuming an atmosphere completely composed of H2O of total mass M_w.
    This assumes H2O is only dissociated due to solar UV flux on the lunar dayside.
    This also assumes that the recombination rate of H2O is zero.
    Based on equations 7a and 7b in Tucker et al., 2021.
    See jupyter notebook "Moon - Photodissociation of H2O" for further details.
    """
    #M_w = [3332352.941176471, 13329411.764705883, 16661764.705882354, 33323529.411764707, \
    #       133294117.64705883, 333235294.11764705, 1332941176.4705882, 3332352941.1764708, 13329411764.705883, \
    #       33323529411.764706, 133294117647.05882, 333235294117.6471, 1332941176470.5884, 3332352941176.4707, \
    #       13329411764705.883, 33323529411764.707] #[kg]
    """rate = [3.6433431040599156, 14.440484618067154, 15.960193524328957, 17.169623819512644, \
            20.060534728384, 22.411403993095487, 26.88680116138493, 30.66967876911732, 38.2352708387219, \
            45.01664083673167, 59.7176366988792, 74.30978487205262, 111.30978954099074, 156.80096453148184, \
            325.35084527315746, 688.2563425534648] #[kg.s-1] (Rate for Lyman Alpha Only)"""
    
    ##Rate including other UV flux in addition to Lyman Alpha (Crovisier, 1989)
    #rate = [   7.69333385,   30.4927277 ,   33.70176611,   36.25561591, \
    #     42.3601035 ,   47.32423166,   56.77454242,   64.76251927, \
    #     80.73812847,   95.05776346,  126.10059031,  156.91357288, \
    #    235.04329616,  331.10309252,  687.01535955, 1453.3316434 ] #[kg.s-1]
    
    
    #*Assuming all-water atmosphere with Ts=253 K (isothermal) (exobase height defined assuming all water atmo)
    #including other UV wavelengths (Crovisier, 1989) for rate_coeff = 12.6e-6
    M_w = np.array([4e7, 1e8, 4e8, 1e9, 4e9, 1e10, 4e10, 1e11, 4e11, 1e12, 4e12, 1e13, 4e13, 1e14]) #[kg]
    rate = np.array([ 106.73601516,  117.29191926,  136.64776263,  152.3229286,   182.00809955, \
                        206.94725487,  256.42707758,  300.34536856,  394.2144441,   485.64613654, \
                        710.34275139,  974.44204361, 1873.1819911,  3580.84759977,])
    

    z = np.polyfit(np.log(M_w), rate, 12)
    p = np.poly1d(z)
    
    return p

def PhotoDisRate_CO():
    """
    Estimate photodissociation rate of CO based on total mass of CO in atmosphere.
    Rate estimated assuming an atmosphere completely composed of CO of total mass M_co.
    This assumes CO is only dissociated due to solar UV flux on the lunar dayside.
    This also assumes that the recombination rate of CO is zero.
    Based on equations 7a and 7b in Tucker et al., 2021.
    Using PD cross sections from Heays et al., 2017 via:
    https://home.strw.leidenuniv.nl/~ewine/photo/cross_sections.html
    Using TIMED SEE solar irradiance spectrum via https://lasp.colorado.edu/lisird/data/timed_see_ssi_l3/
    See jupyter notebook "Moon - Photodissociation of H2O" for further details.
    """
    #M_co = [16411764.705882352, 41029411.76470588, 164117647.05882353, 410294117.6470588, 1641176470.5882351, \
    #       4102941176.470588, 16411764705.882353, 41029411764.70588, 164117647058.82352, 410294117647.05884, \
    #       1641176470588.2354, 4102941176470.588, 16411764705882.352, 41029411764705.88] #[kg]
    """rate = [3.6433431040599156, 14.440484618067154, 15.960193524328957, 17.169623819512644, \
            20.060534728384, 22.411403993095487, 26.88680116138493, 30.66967876911732, 38.2352708387219, \
            45.01664083673167, 59.7176366988792, 74.30978487205262, 111.30978954099074, 156.80096453148184, \
            325.35084527315746, 688.2563425534648] #[kg.s-1] (Rate for Lyman Alpha Only)"""
    
    ##Rate due to solar UV flux from 88-108 nm
    #rate = [3.1263132371509013, 4.374826362365331, 4.790869232813559, 5.100093749923821, 5.629575619507928, \
    #     6.027364839129611, 6.7168547026175, 7.241536315360763, 8.1643439986486, 8.877909172493219, \
    #     10.156292512039238, 11.084861646303498, 12.923515140107545, 14.413814701085542] #[kg.s-1]

    #*Assuming only CO is photodissociating For m = 28.01*u, Ts = 253, Te = 253
    M_co = np.array([4e7, 1e8, 4e8, 1e9, 4e9, 1e10, 4e10, 1e11, 4e11, 1e12, 4e12, 1e13, 4e13, 1e14]) #[kg]
    rate = np.array([ 7.54633399,  7.97295872,  8.69217304,  9.22353435, 10.12736747, 10.80152277, 11.96054156, \
                  12.83492244, 14.35745909, 15.52229874, 17.58292915, 19.18661461, 22.0813258, 24.38451513])
    
    z = np.polyfit(np.log(M_co), rate, 9)
    p = np.poly1d(z)
    
    return p

def SputRate():
    """
    Estimate solar wind induced sputtering escape rate.
    Based on sputter yields for H+ and He+ solar wind ions reported in Table 4 of Tucker et al., 2021.
    """
    flux_H = 16.5e12 #Solar wind hydrogen flux [atoms.m-2.s-1]
    flux_He = 0.04*flux_H #Solar wind helium flux [atoms.m-2.s-1]

    Y_H = 1 #Hydrogen sputtering yield (Tucker et al., 2021)
    Y_He = 3.6 #Helium sputtering yield (Tucker et al., 2021)

    ##For m = 34.1*u, Ts = 270
    #M_atm = [1e7, 4e7, 1e8, 4e8, 1e9, 4e9, 1e10, 4e10, 1e11, 4e11, 1e12, 4e12, 1e13, 4e13, 1e14] #[kg]
    #Rx = np.array([1737400.0, 1753467.0, 1794220.0, 1859711.0, 1905762.0, 1980070.0, 2032548.0, 2117625.0, \
    #               2178007.0, 2276436.0, 2346703.0, 2461984.0, 2544852.0, 2681855.0, 2781160.0])

    #*For m = 31.4*u, Ts = 253
    M_atm = [1e7, 4e7, 1e8, 4e8, 1e9, 4e9, 1e10, 4e10, 1e11, 4e11, 1e12, 4e12, 1e13, 4e13, 1e14] #[kg]
    Rx = np.array([1737400.0, 1757425.3523025583, 1799146.1896679858, 1866282.3958143285, 1913557.1697394275, \
               1989955.3302840001, 2043996.9348483337, 2131764.7512490973, 2194175.159585287, 2296121.2461707457, \
               2369061.6957455557, 2489023.152090012, 2575485.729759463, 2718859.4620919093, 2823119.318694129])
    
    m = 31.4*u

    #SW_eff = 1 #0.09

    rate_sp = m * (Y_H*flux_H + Y_He*flux_He) * np.pi * (Rx)**2

    p = interpolate.interp1d(np.log(M_atm), rate_sp, fill_value='extrapolate')
    
    return p

def LossRate_Exo(M_atm, T0=270, T1=800, M_atm0=1e5, M_atm1=1e7, m=31.4*u):
    mu = m*Na #molar mass of atmosphere [kg.mol-1]
    
    if (M_atm <= M_atm0):
        T = T0
    elif ((M_atm > M_atm0) & (M_atm <= M_atm1)):
        T = T0*(M_atm/M_atm0)**(np.log(T1/T0)/np.log(M_atm1/M_atm0))
    else:
        T = T1
    
    ####################################################
    ##### Variable Pe and Re with atmospheric mass #####
    ####################################################
    #Pe vs mass was fit to a second order polynomial in log10(Pe)-log10(mass) space for different exobase temps
    if ((T1 == 253) & (T0 == 253)):
        z_Pe = [-6.76368812e-07,  4.17659228e-05, -1.65955288e-03, -1.62967589e-02, -1.31255379e+01]
    elif ((T1 == 253) & (T0 == 253) & (m == 18.02*u)):
        z_Pe = [-7.96084182e-05,  7.08652614e-03, -2.39303501e-01,  3.50081524e+00, -3.28825191e+01] #Pure water
    elif ((T1 == 400) & (T0 == 270)):
        z_Pe = [-6.18794884e-07,  3.81140122e-05, -1.54542551e-03, -1.71261338e-02, -1.26517766e+01]
    elif ((T1 == 400) & (T0 == 400)):
        z_Pe = [-8.72130394e-06,  7.01058735e-04, -2.31611994e-02,  2.79691493e-01, -1.44222949e+01]
    elif ((T1 == 270) & (T0 == 270)):
        z_Pe = [-6.18794884e-07,  3.81140122e-05, -1.54542551e-03, -1.71261338e-02, -1.30448192e+01]
    elif (T1 == 600):
        z_Pe = [-1.54798917e-03,  1.38924430e-02, -1.34005684e+01]
    elif (T1 == 700):
        z_Pe =  [-1.54828807e-03,  1.39112446e-02, -1.32467055e+01]
    else: #Default for T1 == 800
        z_Pe = [-1.54798917e-03,  1.38924430e-02, -1.31128863e+01]
        
    p_Pe = np.poly1d(z_Pe)
    
    Pe = np.exp(p_Pe(np.log(M_atm))) #Exobase pressure [Pa]
    #if (Pe > mu*g/(Na*sigma_co*np.sqrt(2))):
    #    Pe = mu*g/(Na*sigma_co*np.sqrt(2))
    
    if (T0 == 253):
        z_Re = [ 3.41792381e-07, -2.12498659e-05,  8.43588768e-04,  7.92061549e-03, 1.40641299e+01]
    elif ((T0 == 253) & (m == 18.02*u)):
        z_Re = [ 3.98049239e-05, -3.54333684e-03,  1.19654570e-01, -1.75045485e+00, 2.36638645e+01] #Pure water
    elif (T0 == 270):
        z_Re = [ 3.05295570e-07, -1.86543946e-05,  7.57978734e-04,  8.80147892e-03, 1.40621918e+01]
    elif (T0 == 400):
        z_Re = [ 4.36159845e-06, -3.50623772e-04,  1.15840096e-02, -1.39898600e-01, 1.47526647e+01]
    else:
        #Re vs mass was fit to a second order polynomial in log10(Pe)-log10(mass) space. Independent of exobase temp
        z_Re = [ 7.74167984e-04, -6.95707481e-03,  1.42910557e+01]
    p_Re = np.poly1d(z_Re)
    Re = np.exp(p_Re(np.log(M_atm))) #Exobase Radius [m]
    ####################################################
    ####################################################
    ####################################################
    
    P0 = M_atm*g/(4*np.pi*R**2) #surface pressure

    if (Pe < P0):
        E = G*M_moon*m/(k*T*Re)
    else:
        E = m*g*R/(k*T)

    #Escape Flux
    if (Pe < P0):
        rhoe = (Pe*mu)/(Rg*T) #Exobase density
    else:
        rhoe = (P0*mu)/(Rg*T) #Exobase density (surface density)

    phi = (1/(2*np.sqrt(np.pi))) * rhoe * ( ((2*k*T)/m)**0.5 ) * ( (1+E)/np.exp(E) ) #escape flux [kg.m-2.s-1]

    if (Pe < P0):
        rate = (4*np.pi*Re**2) * phi #escape rate [kg.s-1]
    else:
        rate = (4*np.pi*R**2) * phi #escape rate [kg.s-1]
        
    return rate, Pe, P0#, E