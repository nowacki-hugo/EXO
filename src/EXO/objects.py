import numpy as np
import matplotlib.pyplot as plt 
from joblib import Parallel, delayed
import multiprocessing
import EXO.utils as utXO
import EXO.funcs as fuXO


# =============================================================================
# Definition of the Star class
# =============================================================================
class Star:
    def __init__(self, data):
        self.name = data.get("Name", "Unknown_Star")                                   ### Name
        self.Ms, self.e_Ms = data.get("Ms", None), data.get("e_Ms", None)              ### Stellar mass [Solar mass]
        self.Rs, self.e_Rs = data.get("Rs", None), data.get("e_Rs", None)              ### Stellar radius [Solar radius]
        self.Rhos, self.e_Rhos = data.get("Rhos", None), data.get("e_Rhos", None)      ### Stellar density [Solar density]
        self.LDD, self.e_LDD = data.get("LDD", None), data.get("e_LDD", None)          ### Limb-darkened angular diameter [mas]
        self.plx, self.e_plx = data.get("plx", None), data.get("e_plx", None)          ### Gaia parallax [mas]
        self.Fbol, self.e_Fbol = data.get("Fbol", None), data.get("e_Fbol", None)      ### Bolometric flux [erg/s/cm2]
        self.Ls, self.e_Ls = data.get("Ls", None), data.get("e_Ls", None)              ### Stellar luminosity [Solar luminosity]
        
        ### Compute the stellar radius [Rsun]
        if self.LDD and self.plx :
            self.Rs = utXO.au2Rsun( self.LDD / self.plx ) / 2
            self.e_Rs = self.Rs * np.sqrt( (self.e_plx/self.plx)**2 + (self.e_LDD/self.LDD)**2 )
        ### Compute the stellar effective temperature [Kelvins]
        if self.Ls and self.Rs:
                self.Teff = utXO.TSUN*(self.Ls)**(1/4)*(self.Rs)**(-1/2)
                self.e_Teff = self.Teff*np.sqrt( (self.e_Ls/self.Ls/4 )**2 + (self.e_Rs/self.Rs/2)**2 )
        
    def compute_radius_pdf( self, R_grid, new_params=True, plot=True ):
        plx, e_plx = self.plx/1000/utXO.PARSEC, self.e_plx/1000/utXO.PARSEC
        LDD, e_LDD = utXO.mas2rad(self.LDD), utXO.mas2rad(self.e_LDD)
        p_min, p_max, Np = plx-5*e_plx, plx+5*e_plx, 5000
        p_grid = np.geomspace(p_min, p_max, Np)
        params = plx, e_plx, LDD, e_LDD
        pdf_R = np.zeros(R_grid.shape)
        for i,r in enumerate(R_grid) :
            pdf_R[i] = fuXO.f_R( p_grid, r*utXO.RSUN, params )
        best_R = R_grid[np.nanargmax(pdf_R)]
        zone = pdf_R >= np.nanmax(pdf_R)/np.sqrt(np.e)
        up_R = np.nanmax(R_grid[zone]) - best_R
        low_R = best_R - np.nanmin(R_grid[zone])
        
        if plot :
            plt.figure()
            plt.plot(R_grid, pdf_R)
            plt.xlabel(r'Radius ($R_\odot$)'), plt.yticks([])
            plt.scatter(best_R, np.nanmax(pdf_R), marker='+', color='red')
            plt.hlines( np.nanmax(pdf_R)/np.sqrt(np.e), best_R-low_R, best_R+up_R, ls='dotted', color='r' )
            plt.vlines( [best_R-low_R, best_R, best_R+up_R], 0, [np.nanmax(pdf_R)/np.sqrt(np.e), np.nanmax(pdf_R), np.nanmax(pdf_R)/np.sqrt(np.e)], ls=['dashed'], color='r' )
            plt.xlim(R_grid.min(), R_grid.max()), plt.ylim(0 ,1.05*np.nanmax(pdf_R))
            plt.show()
        
        if new_params :
            self.Rs, self.ep_Rs, self.em_Rs, self.e_Rs = best_R, up_R, low_R, (up_R+low_R)/2
        
        return pdf_R
    
    def compute_density(self, a_Rs, e_a_Rs, P, e_P, new_params=True):
        ### P = Period of an orbiting planet [days]
        rho = a_Rs**3. * (utXO.Rsun2Au(self.Rs)/self.Rs)**3. / fuXO.daysToYear(P)**2.
        e_rho = rho*np.sqrt((3.*e_a_Rs/a_Rs)**2.+(2.*e_P/P)**2.)
        if new_params :
            self.Rhos, self.e_Rhos = rho, e_rho
        return rho, e_rho
        
    
    def compute_mass(self, Rhos, e_Rhos, Rs, e_Rs):
        ### Rhos = stellar density [solar density]
        ### Rs = stellar radius [solar radii]
        self.Ms = Rhos * Rs**3.
        self.e_Ms = self.Ms * np.sqrt( (e_Rhos/Rhos)**2. + (e_Rs/Rs)**2. )
        return self.Ms, self.e_Ms
        
    
    def compute_HZ(self, T=None, e_T=None, R=None, e_R=None, L=None, e_L=None, method='optimistic'):
        
        if method == 'optimistic':
            S_inf, S_sup = 0.32, 1.78
        elif method == 'conservative':
            S_inf, S_sup = 0.35, 1.0
        else :
            return 'Unkown method, chose between "conservative" and "optimistic".'
        
        if self.Ls or L :
            Ls, e_Ls = self.Ls, self.e_Ls
            if not Ls :
                Ls, e_Ls = L, e_L
        elif (self.Rs and self.Teff) or (R and T) :
            Ts, e_Ts = self.Teff, self.e_Teff
            Rs, e_Rs = self.Rs, self.e_Rs
            if not Ts :
                Ts, e_Ts = T, e_T
            if not Rs :
                Rs, e_Rs = R, e_R
            Ls = Rs**2 * (Ts/utXO.TSUN)**4
            e_Ls = Ls*np.sqrt( 2*e_Rs/Rs + 4*e_Ts/Ts )
            
        else :
            return 'Cannot evaluate the HZ. Need either Ls, or Teff+Rs'
        
        HZinf, HZsup = np.sqrt(Ls/S_sup), np.sqrt(Ls/S_inf)
        e_HZinf, e_HZsup = HZinf/2*e_Ls/Ls, HZsup/2*e_Ls/Ls
        
        return HZinf, e_HZinf, HZsup, e_HZsup
    
    def computeMLF_LT( self, L_grid, T_grid, new_params=True, plot=True, ncpulim=None, verbose=False, savefig=False ):
        # ============================================================
        # Computes the joint maximum likelihood function of luminosity
        # and temperature from bolometric flux, apparent diameter and 
        # parallax
        # ============================================================
        if verbose :
            print("COMPUTING MLF FOR STELLAR LUMINOSITY AND TEMPERATURE")
        if not (self.Fbol and self.plx and self.LDD) :
            return print( "Missing parameters among Fbol, plx and LDD (or their error)." )
            
        else :
            Fbol, e_Fbol = utXO.ergs_s_cm2ToWatts(self.Fbol), utXO.ergs_s_cm2ToWatts(self.e_Fbol)
            plx, e_plx = self.plx/1000/utXO.PARSEC, self.e_plx/1000/utXO.PARSEC
            LDD, e_LDD = utXO.mas2rad( self.LDD ), utXO.mas2rad( self.e_LDD )
            
        t_min, t_max, Nt = Fbol-5*e_Fbol, Fbol+5*e_Fbol, 5000
        t_grid = np.geomspace(t_min, t_max, Nt)
        
        L_grid = L_grid * utXO.LSUN
        points = [(Lval, Tval) for Lval in L_grid for Tval in T_grid]
        params = (Fbol, e_Fbol, plx, e_plx, LDD, e_LDD)
        
        def _compute_point(point):
            Lval, Tval = point
            return fuXO.likelihood_LT(t_grid, Lval, Tval, params)
        
        if ncpulim :
            n_jobs = min(ncpulim, multiprocessing.cpu_count())
        else :
            n_jobs = multiprocessing.cpu_count()
            
        results = Parallel( n_jobs=n_jobs, verbose=False )( delayed(_compute_point)(p) for p in points )
        likelihood_grid = np.array(results).reshape(len(L_grid),len(T_grid)) 
        likelihood_grid /= np.nanmax(likelihood_grid)
        
        # ============================================================
        # Extracting best values for Luminosity-Temperature
        # ============================================================
        idx_l, idx_t = np.unravel_index(np.nanargmax(likelihood_grid), likelihood_grid.shape)
        best_L, best_T = L_grid[idx_l]/utXO.LSUN, T_grid[idx_t]
        crit = (likelihood_grid >= 1/np.sqrt(np.e))
        LL, TT = np.meshgrid(L_grid,T_grid)
        up_L, low_L = np.nanmax(LL[crit.T])/utXO.LSUN - best_L, best_L - np.nanmin(LL[crit.T])/utXO.LSUN
        up_T, low_T = np.nanmax(TT[crit.T]) - best_T, best_T - np.nanmin(TT[crit.T])
        if new_params :
            self.Ls, self.ep_Ls, self.em_Ls, self.e_Ls = best_L, up_L, low_L, (up_L+low_L)/2
            self.Teff, self.ep_Teff, self.em_Teff, self.e_Teff = best_T, up_T, low_T, (up_T+low_T)/2
    
        if plot :
            # =============================================================================
            # Plot joint likelihood function
            # =============================================================================
            fig, axs = plt.subplot_mosaic([['sum_T', '.'], ['dens', 'sum_L']], figsize=(3.5, 3.5), width_ratios=(5, 1), height_ratios=(1, 5), layout='constrained')
            axs['dens'].pcolormesh(T_grid, L_grid/utXO.LSUN, likelihood_grid, cmap='CMRmap_r')
            axs['dens'].contour(T_grid, L_grid/utXO.LSUN, likelihood_grid, levels=[1/np.sqrt(np.e**9),1/np.e**2,1/np.sqrt(np.e)], colors=['k'], linestyles=["dotted", "dashed", "solid"])
            axs['dens'].scatter(best_T, best_L, marker='+',color='white')
            axs['dens'].set_xlabel(r'Effective temperature (K)'), axs['dens'].set_ylabel(r'Bolometric luminosity ($L_\odot$)')
            axs['dens'].set_xlim(np.nanmin(T_grid), np.nanmax(T_grid)), axs['dens'].set_ylim(np.nanmin(L_grid/utXO.LSUN), np.nanmax(L_grid/utXO.LSUN))
            axs['sum_T'].plot(T_grid, np.nansum(likelihood_grid,axis=0), color='k', lw=0.7)
            axs['sum_T'].set_xlim(np.nanmin(T_grid), np.nanmax(T_grid))
            axs['sum_T'].set_xticks(ticks=[]), axs['sum_T'].set_yticks(ticks=[])
            axs['sum_L'].plot(np.nansum(likelihood_grid,axis=1), L_grid/utXO.LSUN,color='k', lw=0.7)
            axs['sum_L'].set_ylim(np.nanmin(L_grid/utXO.LSUN), np.nanmax(L_grid/utXO.LSUN))
            axs['sum_L'].set_xticks(ticks=[]), axs['sum_L'].set_yticks(ticks=[])
            axs['dens'].text(0.1, 0.85, self.name, horizontalalignment='left', verticalalignment='center',  transform = axs['dens'].transAxes )
            if savefig :
                if type(savefig) is str :
                    fig.savefig(savefig,dpi=250)
                else :
                    fig.savefig("Likelihood_LT_"+self.name+".jpg",dpi=250)
                
            plt.show()
        # self.MLFLT_comp = True
        return likelihood_grid
    
    
    def computeMLF_MR( self, M_grid, R_grid, new_params=True, plot=True, savefig=False, ncpulim=None, verbose=False ):
        # ============================================================
        # Computes the joint maximum likelihood function of mass
        # and radius from stellar density, apparent diameter and 
        # parallax
        # ============================================================
        if verbose :
            print("COMPUTING MLF FOR STELLAR MASS AND RADIUS")
        if not (self.Rhos and self.plx and self.LDD) :
            return print("WARNING : Missing parameters among stellar density, plx and LDD. \n CANNOT COMPUTE MLF.")
        else :
            rhos, e_rhos = self.Rhos*utXO.RHOSUN, self.e_Rhos*utXO.RHOSUN
            plx, e_plx = self.plx/1000/utXO.PARSEC, self.e_plx/1000/utXO.PARSEC
            LDD, e_LDD = utXO.mas2rad(self.LDD), utXO.mas2rad(self.e_LDD)
        
        p_min, p_max, Np = plx-5*e_plx, plx+5*e_plx, 5000
        p_grid = np.geomspace(p_min, p_max, Np)
        
        M_grid, R_grid = M_grid*utXO.MSUN, R_grid*utXO.RSUN
        points = [(Rval, Mval) for Rval in R_grid for Mval in M_grid]
        params = (rhos, e_rhos, plx, e_plx, LDD, e_LDD)
        
        def _compute_point(point):
            Rval, Mval = point
            return fuXO.likelihood_RM(p_grid, Rval, Mval, params)
        
        if ncpulim :
            n_jobs = min(ncpulim, multiprocessing.cpu_count())
        else :
            n_jobs = multiprocessing.cpu_count()
            
        results = Parallel( n_jobs=n_jobs, verbose=False )( delayed(_compute_point)(p) for p in points )
        likelihood_grid = np.array(results).reshape(len(R_grid),len(M_grid))    
        likelihood_grid /= np.nanmax(likelihood_grid)
        
        # ============================================================
        # Extracting best values for Mass-Radius
        # ============================================================
        idx_r, idx_m = np.unravel_index(np.nanargmax(likelihood_grid), likelihood_grid.shape)
        best_M, best_R = M_grid[idx_m]/utXO.MSUN, R_grid[idx_r]/utXO.RSUN
        crit = (likelihood_grid >= 1/np.sqrt(np.e))
        RR, MM = np.meshgrid(R_grid,M_grid)
        up_M, low_M = np.nanmax(MM[crit.T])/utXO.MSUN - best_M, best_M - np.nanmin(MM[crit.T])/utXO.MSUN
        up_R, low_R = np.nanmax(RR[crit.T])/utXO.RSUN - best_R, best_R - np.nanmin(RR[crit.T])/utXO.RSUN
        
        if new_params :
            self.Rs, self.ep_Rs, self.em_Rs, self.e_Rs = best_R, up_R, low_R, (up_R+low_R)/2
            self.Ms, self.ep_Ms, self.em_Ms, self.e_Ms = best_M, up_M, low_M, (up_M+low_M)/2
            self.Rhos, self.e_Rhos = best_M/best_R**3, best_M/best_R**3*np.sqrt((self.e_Ms/self.Ms)**2 + (self.e_Rs/self.Rs/3)**2)
        
        if plot :
            fig, axs = plt.subplot_mosaic([['sum_M', '.'], ['dens', 'sum_R']], figsize=(3.5, 3.5), width_ratios=(5, 1), height_ratios=(1, 5), layout='constrained')
            axs['dens'].pcolormesh(M_grid/utXO.MSUN, R_grid/utXO.RSUN, likelihood_grid, cmap='CMRmap_r')
            axs['dens'].contour(M_grid/utXO.MSUN, R_grid/utXO.RSUN, likelihood_grid, levels=[1/np.sqrt(np.e**9),1/np.e**2,1/np.sqrt(np.e)], colors=['k'], linestyles=["dotted", "dashed", "solid"])
            axs['dens'].scatter(best_M, best_R, marker='+',color='white')
            axs['dens'].set_xlabel(r'Stellar mass ($M_\odot$)'), axs['dens'].set_ylabel(r'Stellar radius ($R_\odot$)')
            axs['sum_M'].plot(M_grid/utXO.MSUN, np.nansum(likelihood_grid,axis=0), color='k', lw=0.7)
            axs['sum_M'].set_xlim(np.nanmin(M_grid/utXO.MSUN), np.nanmax(M_grid/utXO.MSUN))
            axs['sum_M'].set_xticks(ticks=[]), axs['sum_M'].set_yticks(ticks=[])
            axs['sum_R'].plot(np.nansum(likelihood_grid,axis=1), R_grid/utXO.RSUN,color='k', lw=0.7)
            axs['sum_R'].set_ylim(np.nanmin(R_grid/utXO.RSUN), np.nanmax(R_grid/utXO.RSUN))
            axs['sum_R'].set_xticks(ticks=[]), axs['sum_R'].set_yticks(ticks=[])
            axs['dens'].text(0.1, 0.85, self.name, horizontalalignment='left', verticalalignment='center',  transform = axs['dens'].transAxes )
            if savefig :
                if type(savefig) is str :
                    fig.savefig(savefig,dpi=250)
                else :
                    fig.savefig("Likelihood_MR_"+self.name+".jpg",dpi=250)
            plt.show()
        
        self.MLFMR_comp = True
        
        return likelihood_grid
    
    
    def __repr__(self):
        return (
            f"===========================================\n"
            f"Star object (as defined in EXO) \n"
            f"Main properties :\n"
            f"M= {self.Ms:.3f} Msun | R= {self.Rs:.3f} Rsun \n"
            f"===========================================\n"
        )






# =============================================================================
# Definition of the Planet class
# =============================================================================
class Planet :
    def __init__( self, data={} ):#, *args, **kwargs ):
        # super().__init__(*args, **kwargs)
        self.name = data.get("Name", None)                                   ### Name
        self.dF, self.e_dF = data.get("dF", None), data.get("e_dF", None)           ### Relative transit depth [ppm]
        self.Mp, self.e_Mp = data.get("Mp", None), data.get("e_Mp", None)           ### Planetary mass [earth mass]
        self.Rp, self.e_Rp = data.get("Rp", None), data.get("e_Rp", None)           ### Planetary earth [earth radius]
        self.Rhop, self.e_Rhop = data.get("Rhop", None), data.get("e_Rhop", None)   ### Planetary density [earth density]
        self.transit_computed, self.RV_computed = False, False
        self.P, self.e_P = data.get("Porb",None), data.get("e_Porb",0)
        self.K, self.e_K = data.get("K",None), data.get("e_K",0)
        self.inc, self.e_inc = utXO.degToRad(data.get("inc",np.pi/2)), utXO.degToRad(data.get("e_inc",0))
         
    def compute_inclination(self, a_Rs, e_a_Rs, b, e_b, new_params=True):
        ### b = impact parameter
        inc = np.arccos(b / a_Rs)
        e_inc = np.sqrt( (e_b/a_Rs)**2 + (b*e_a_Rs/(a_Rs)**2)**2 ) / np.sqrt(1-(b/a_Rs)**2)
        if new_params :
            self.inc = inc
            self.e_inc = e_inc
        return inc, e_inc
    
    def compute_radius(self, Rp_Rs, e_Rp_Rs, Rs, e_Rs, new_params=True):
        Rp = utXO.RsunToRearth(Rp_Rs*Rs)
        e_Rp = self.Rp*np.sqrt( (e_Rp_Rs/Rp_Rs)**2 + (e_Rs/Rs)**2 )
        if new_params :
            self.Rp = Rp
            self.e_Rp = e_Rp
        return Rp, e_Rp
        
    
    def distance_from_RV(self, star, P_days=None, e_P_days=None, M_Msun=None, e_M_Msun=None, new_params=True):
        if not P_days :
            P_days = self.P
            e_P_days = self.e_P
        if not M_Msun :
            M_Msun = star.Ms
            e_M_Msun = star.e_Ms
        T, e_T = utXO.daysToSec(P_days), utXO.daysToSec(e_P_days)
        M, e_M = M_Msun*utXO.MSUN, e_M_Msun*utXO.MSUN
        a = (utXO.G*M*T**2/4/np.pi**2)**(1./3.)
        err_a = a * np.hypot( e_M/M/3 , 2*e_T/T/3 )
        if new_params :
            self.a_Rs, self.e_a_Rs = a, err_a
        return a/star.Rs/utXO.RSUN, err_a/star.Rs/utXO.RSUN
        
    def compute_Teq( self, star, a=None, e_a=None, alb=0, eps=1, new_params=True, method='L' ):
        ### Equilibrium temperature of the dayside
        ### Typical values : 1/4 = dense atm but tidally locked ; 2/3 = thin atm and tidally locked ; 1 = perfect redistribution ; 2 no redistribution. 
        if not star.Teff and not star.Ls :
            raise ValueError("Need stellar temperature or Luminosity")
        
        Rs, e_Rs = star.Rs, star.e_Rs
        
        if not a :
            a_Rs = self.a_Rs
            e_a_Rs = self.e_a_Rs
            if not a :
                raise ValueError("If no semi-major axis is provided, need to compute transit parameters (a/Rs)")
        else :
            a_Rs = a/Rs * utXO.AU/utXO.RSUN
            e_a_Rs = a_Rs * np.hypot( e_Rs/Rs, e_a/a )
                
        if method == 'L' :
            Ls, e_Ls = star.Ls, star.e_Ls
            Ts = utXO.TSUN * Ls**0.25 * Rs**-0.5
            e_Ts = Ts * np.hypot(e_Ls/Ls/4, e_Rs/Rs/2)
        elif method == 'T' :
            Ts, e_Ts = star.Teff, star.e_Teff
        else:
            raise ValueError("Unknown method, either 'L' for luminosity or 'T' for temperature.")
        
        T0 = Ts*a_Rs**-0.5
        
        Td = T0 * (1-alb)**0.25 * (2/3 - 5*eps/12)**0.25
        e_Td = Td * np.sqrt( (e_Ts/Ts/2)**2 + (e_a_Rs/a_Rs/2)**2 )
        
        Tn = T0 * (1-alb)**0.25 * (eps/4)**0.25
        e_Tn = Tn * np.sqrt( (e_Ts/Ts/2)**2 + (e_a_Rs/a_Rs/2)**2 )
        
        Teq = ( (Tn**4 + Td**4)/2 )**0.25
        if eps == 0 :
            e_Teq = Teq * e_Td/Td
        else :
            e_Teq = Teq * np.hypot( e_Td/Td, e_Tn/Tn )
        
        if not Teq :
            raise ValueError("Need to compute either temperature or (luminosity + radius) of the star.")
        
        if new_params :
            self.Td, self.e_Td = Td, e_Td
            self.Tn, self.e_Tn = Tn, e_Tn
            self.Teq, self.e_Teq = Teq, e_Teq
        
        
        return Teq, e_Teq
        
    def transit( self, star, params_transit, new_params=True, recover=False, force=False ) :
        ### Computes Planetary and Stellar parameters from transit observations
        ### dF [ppm]
        ### tauT/tauF [hours]
        ### Porb [days]
        ### ecc
        ### omega [degrees]
        
        if ((not self.transit_computed) or (self.transit_computed and force) ) :
            self.dF, self.e_dF = params_transit.get("dF",0), params_transit.get("e_dF",0)               ### Transit depth [ppm]
            self.tauT, self.e_tauT = params_transit.get("tauT",0)/24, params_transit.get("e_tauT",0)/24       ### Transit total duration [hours]
            self.tauF, self.e_tauF = params_transit.get("tauF",1)/24, params_transit.get("e_tauF",1)/24       ### Transit flat duration [hours]
            self.P, self.e_P = params_transit.get("Porb",np.inf), params_transit.get("e_Porb",0)  ### Transit Period [days]
            ### Compute planet radius [Stellar radii]
            self.Rp_Rs, self.e_Rp_Rs = np.sqrt(self.dF*1E-6), self.e_dF*1E-6/2/np.sqrt(self.dF*1E-6)
            Rp = utXO.RsunToRearth(self.Rp_Rs*star.Rs)
            e_Rp = Rp*np.sqrt( (self.e_Rp_Rs/self.Rp_Rs)**2 + (star.e_Rs/star.Rs)**2 )
            if new_params :
                self.Rp = Rp
                self.e_Rp = e_Rp            
            
            ### Compute the perturbation due to eccentricity
            if (not self.RV_computed) :
                print("!!! WARNING !!!")
                print("With RV not computed, eccentricity = 0 if not in parameters")    
                self.ecc, self.e_ecc = params_transit.get("ecc",0), params_transit.get("e_ecc",0)
                self.ome, self.e_ome = params_transit.get("ome",0), params_transit.get("e_ome",0)
            else :
                print("!!! WARNING !!!")
                print("RV computed, taking eccentricity ONLY from there")
            
            tauTo, e_tauTo = self.tauT, self.e_tauT
            tauFo, e_tauFo = self.tauF, self.e_tauF
            
            self.tauT *= np.sqrt(1-self.ecc**2)/( 1 + self.ecc*np.sin(utXO.degToRad(self.ome)) )
            self.tauF *= np.sqrt(1-self.ecc**2)/( 1 + self.ecc*np.sin(utXO.degToRad(self.ome)) )
            
            self.e_tauT = self.tauT*np.sqrt( (e_tauTo/tauTo)**2 
                                            + ((self.ecc+np.sin(utXO.degToRad(self.ome)))/(1+self.ecc*np.sin(utXO.degToRad(self.ome))))**2*self.e_ecc**2/(1-self.ecc**2)**2 
                                            + (self.ecc*np.cos(utXO.degToRad(self.ome))/(1+self.ecc*np.sin(utXO.degToRad(self.ome))))**2*utXO.degToRad(self.e_ome)**2 )
            self.e_tauF = self.tauF*np.sqrt( (e_tauFo/tauFo)**2 
                                            + ((self.ecc+np.sin(utXO.degToRad(self.ome)))/(1+self.ecc*np.sin(utXO.degToRad(self.ome))))**2*self.e_ecc**2/(1-self.ecc**2)**2 
                                            + (self.ecc*np.cos(utXO.degToRad(self.ome))/(1+self.ecc*np.sin(utXO.degToRad(self.ome))))**2*utXO.degToRad(self.e_ome)**2 )
            
            ### Compute the impact parameter [Stellar radii]
            SR = np.sin(np.pi*self.tauF/self.P)/np.sin(np.pi*self.tauT/self.P)
            tanF, tanT = np.tan(np.pi*self.tauF/self.P), np.tan(np.pi*self.tauT/self.P)
            e_SR = np.pi*SR/self.P * np.sqrt( (self.e_tauF/tanF)**2 + (self.e_tauT/tanT)**2
                                  + (self.e_P/self.P)**2*(self.tauT/tanT-self.tauF/tanF)**2 )
            ### Debug point
            # print(( (1-self.Rp_Rs)**2 - (1+self.Rp_Rs)**2 * SR**2 ) / ( 1 - SR**2 ) )
            self.b = np.sqrt( abs( (1-self.Rp_Rs)**2 - (1+self.Rp_Rs)**2 * SR**2 ) / ( 1 - SR**2 ) )
            self.e_b = np.sqrt( self.e_Rp_Rs**2 *((1-self.Rp_Rs+SR**2*(1+self.Rp_Rs))/self.b/(1-SR**2))**2
                                                           + (e_SR*4*SR*self.Rp_Rs/self.b/(1-SR**2)**2)**2 )
            
            ### Compute the semi-major axis [Stellar radii]
            cosT, sinT = np.cos(np.pi*self.tauT/self.P), np.sin(np.pi*self.tauT/self.P)
            self.a_Rs = np.sqrt( ((1+self.Rp_Rs)**2-self.b**2*(1-sinT**2)) / sinT**2 )
            self.e_a_Rs = 1/self.a_Rs/sinT**2 * np.sqrt( (1+self.Rp_Rs)**2 *self.e_Rp_Rs**2
                                + (self.b*cosT**2*self.e_b)**2
                                + (np.pi/self.P/tanT)**2 
                                    *(self.b**2-(1+self.Rp_Rs)**2)**2
                                    *(self.e_tauT**2 + self.tauT**2/self.P**2*self.e_P**2) )
            
            ### Compute the inclination of the orbit [rad]
            self.inc = np.arccos(self.b/self.a_Rs)
            self.e_inc = np.cos(self.inc)/np.sqrt(1+(self.b/self.a_Rs)**2) * np.sqrt((self.e_b/self.b)**2 
                                                                             + (self.e_a_Rs/self.a_Rs)**2 )
            ### Compute stellar density [solar density]
            Rhos = self.a_Rs**3 * (utXO.Rsun2Au(star.Rs)/star.Rs)**3 / utXO.daysToYear(self.P)**2
            e_Rhos = star.Rhos*np.sqrt((3*self.e_a_Rs/self.a_Rs)**2+(2*self.e_P/self.P)**2)
            if new_params :
                star.Rhos = Rhos
                star.e_Rhos = e_Rhos
            
            return Rp, e_Rp, Rhos, e_Rhos
            ### Transit has been computed
            self.transit_computed = True
            
            if recover :
                #Make a dictionary with all computed intermediate parameters
                print('Sorry, part not implemented yet, set recover=False')
                
        else :
            return 'Transit already computed, pass "force=True" to overrule.'


    def RV( self, star, params_RV, force=False ):
        ### Computes Planetary and Stellar parameters from radial velocity observations
        ### K [m/s]
        ### inc [deg]
        ### Porb [days]
        ### ecc
        ### omega [deg]
        if ((not self.RV_computed) or (self.RV_computed and force) ) :
            if not self.transit_computed :
                print("!!! WARNING !!!")
                print("Without transits, Mass values will be an upper limit, expressed in stellar masses")
                self.inc, self.e_inc = utXO.degToRad(params_RV.get("inc",90)), utXO.degToRad(params_RV.get("e_inc",0))  ### Transit inclination [rads]
                star.Ms, star.e_Ms = 1, 0
            else :
                print("!!! WARNING !!!")
                print('Transit computed : taking inclination and period from there.')
            
            # Extract radial velocities parameters
            self.P, self.e_P = params_RV.get("Porb",None), params_RV.get("e_Porb",0)  ### Transit Period [days]
            self.K, self.e_K = params_RV.get('K',0), params_RV.get('e_K',0)
            self.ecc, self.e_ecc = params_RV.get("ecc",0), params_RV.get("e_ecc",0)
            # self.ome, self.e_ome = params_RV.get("ome",0), params_RV.get("e_ome",0)
            
            # Derive parameters from radial velocities
            # Debug
            # print(self.P, self.K, self.inc, self.ecc, star.Ms)
            self.Mp = 1.564 * self.P**(1/3)*self.K/np.sin(self.inc) * np.sqrt(1-self.ecc**2) * (star.Ms)**(2/3)
            self.e_Mp = self.Mp*np.sqrt( (self.e_P/self.P/3)**2 + (2*star.e_Ms/star.Ms/3)**2 + (self.e_K/self.K)**2 \
                                        + (self.ecc*self.e_ecc/(1-self.ecc**2))**2 + (self.e_inc/np.tan(self.inc))**2 )
            self.RV_computed = True
            if not self.transit_computed :
                star.Ms, star.e_Ms = None, None
            else :
                self.Rhop = self.Mp/self.Rp**3
                self.e_Rhop = self.Rhop*np.sqrt( (self.e_Mp/self.Mp)**2 + (3*self.e_Rp/self.Rp)**2 )
        else :
            print('Transit already computed, pass "force=True" to overrule.')
            
            
    def PDF_MR(self, Mp_grid, Rp_grid, Ms_grid, Rs_grid, L_MR_grid, new_params=True, plot=True, savefig=False):
        # Formating the entry parameters for the function
        params = ( self.K, self.e_K, self.dF*1e-6, self.e_dF*1e-6, utXO.daysToSec(self.P), self.inc )
        # Computing the grid
        pdf_MR_grid = fuXO.compute_fp( Mp_grid, Rp_grid, Ms_grid, Rs_grid, L_MR_grid.T, params, block_size=64 ).T
        normfac = np.nanmax(pdf_MR_grid)
        pdf_MR_grid /= normfac
        
        # ============================================================
        # Extracting best values for Mass-Radius
        # ============================================================
        idx_m, idx_r = np.unravel_index(np.nanargmax(pdf_MR_grid.T), pdf_MR_grid.T.shape)
        best_M, best_R = Mp_grid[idx_m], Rp_grid[idx_r]
        crit = (pdf_MR_grid >= 1/np.sqrt(np.e))
        RR, MM = np.meshgrid(Rp_grid,Mp_grid)
        up_M, low_M = np.nanmax(MM[crit.T])- best_M, best_M - np.nanmin(MM[crit.T])
        up_R, low_R = np.nanmax(RR[crit.T]) - best_R, best_R - np.nanmin(RR[crit.T])
        
        if new_params :
            self.Rp, self.ep_Rp, self.em_Rp, self.e_Rp = best_R, up_R, low_R, (up_R+low_R)/2
            self.Mp, self.ep_Mp, self.em_Mp, self.e_Mp = best_M, up_M, low_M, (up_M+low_M)/2
            self.Rhop, self.e_Rhop = best_M/best_R**3, best_M/best_R**3*np.sqrt((self.e_Mp/self.Mp)**2 + (self.e_Rp/self.Rp/3)**2)
        
        if plot :
            fig, axs = plt.subplot_mosaic([['sum_M', '.'], ['dens', 'sum_R']], figsize=(3.5, 3.5), width_ratios=(5, 1), height_ratios=(1, 5), layout='constrained')
            axs['dens'].pcolormesh(Mp_grid, Rp_grid, pdf_MR_grid, cmap='CMRmap_r')
            axs['dens'].contour(Mp_grid, Rp_grid, pdf_MR_grid, levels=[1/np.sqrt(np.e**9),1/np.e**2,1/np.sqrt(np.e)], colors=['k'], linestyles=["dotted", "dashed", "solid"])
            axs['dens'].scatter(best_M, best_R, marker='+',color='white', zorder=15)
            axs['dens'].set_xlabel(r'Planetary mass ($M_\oplus$)'), axs['dens'].set_ylabel(r'Planetary radius ($R_\oplus$)')
            axs['sum_M'].plot(Mp_grid, np.nansum(pdf_MR_grid,axis=0), color='k', lw=0.7)
            axs['sum_M'].set_xlim(np.nanmin(Mp_grid), np.nanmax(Mp_grid))
            axs['sum_M'].set_xticks(ticks=[]), axs['sum_M'].set_yticks(ticks=[])
            axs['sum_R'].plot(np.nansum(pdf_MR_grid,axis=1), Rp_grid, lw=0.7, color='k')
            axs['sum_R'].set_ylim(np.nanmin(Rp_grid), np.nanmax(Rp_grid))
            axs['sum_R'].set_xticks(ticks=[]), axs['sum_R'].set_yticks(ticks=[])
            axs['dens'].text(0.1, 0.85, self.name, horizontalalignment='left', verticalalignment='center',  transform = axs['dens'].transAxes )
            if savefig :
                if type(savefig) is str :
                    fig.savefig(savefig,dpi=250)
                else :
                    fig.savefig("PDF_MR_planet_"+self.name+".jpg",dpi=250)
            plt.show( )
               
        return( pdf_MR_grid )
    
            
    
    def __repr__(self):
        if ( self.transit_computed and self.RV_computed ) :
            return (
            f"===========================================\n"
            f"{self.name}\n"
            f"Planet object (as defined in EXO) \n"
            f"Main properties :\n"
            f"M= {self.Mp:.3f} (Mearth) | R= {self.Rp:.3f} (Rearth) | density= {self.Rhop:.3f} (earth)\n"
            f"Transit and RV computed \n"
            f"===========================================\n")
        elif self.transit_computed :
            return (
            f"===========================================\n"
            f"{self.name}\n"
            f"Planet object (as defined in EXO) \n"
            f"Main properties :\n"
            f"R= {self.Rp:.3f} (Rearth) | density= {self.Rhop:.3f} (earth)\n"
            f"Transit computed \n"
            f"===========================================\n")
        elif self.RV_computed :
            return (
            f"===========================================\n"
            f"{self.name}\n"
            f"Planet object (as defined in EXO) \n"
            f"Main properties :\n"
            f"Msini= {self.Mp:.3f} (Mearth)\n"
            f"RV computed \n"
            f"===========================================\n")
        else :
            return (
            "===========================================\n"
            f"{self.name}\n"
            "Planet object (as defined in EXO) \n"
            "Main properties unknown for the moment \n"
            "===========================================\n")
