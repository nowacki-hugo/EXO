import numpy as np
import scipy.constants as csts
from scipy.stats import norm
from scipy.integrate import quad
import scipy.optimize as opt

G = csts.G


def Gaussian(x,mu,sigma):
    output = np.exp( -(x-mu)**2/2/sigma**2 )
    return output

def Integral(sigma):
    solution = quad( Gaussian, 0, np.inf, args=(0, sigma) )
    return solution

def tauF( Rp_Rs, e_Rp_Rs, P, e_P, b, e_b, tauT, e_tauT ):
    tauF = P / np.pi * np.arcsin( np.sin(np.pi*tauT/P) * np.sqrt((1-Rp_Rs)**2-b**2) / np.sqrt((1+Rp_Rs)**2-b**2) )
    X = np.sin( np.pi*tauF/P )
    sinT, tanT = np.sin(np.pi*tauT/P), np.tan(np.pi*tauT/P)
    e_tauF = np.sqrt( (np.arcsin(X) + X/np.sqrt(1-X**2)*tauT/P/tanT)**2/np.pi**2 * e_P**2 
                     + (X/np.sqrt(1-X**2)/tanT)**2 * e_tauT**2
                     + (2*P*sinT**2/np.pi/X/np.sqrt(1-X**2)*(1-Rp_Rs**2)/((1+Rp_Rs)**2-b**2)**2)**2 * e_Rp_Rs**2
                     + (4*b*P*sinT**2/np.pi/X/np.sqrt(1-X**2)*Rp_Rs/((1+Rp_Rs)**2-b**2)**2)**2 * e_b**2 )
    return tauF, e_tauF


def RsunToAu(Rsun):
    au = Rsun/215.032
    return au
def auToRsun(au):
    Rsun = au*215.032
    return Rsun

def radToDeg(rad):
    deg = rad*180/np.pi
    return deg

def degToRad(deg):
    rad = deg/180*np.pi
    return rad

def dayToYear(days):
    years = days/365.25
    return years

def yearToDays(years):
    days = years*365.25
    return days

def Gaussian_2D(xy, amplitude, xo, yo, sigma_x, sigma_y, theta):
    x, y = xy
    # xo = float(xo)
    # yo = float(yo)
    theta = degToRad( theta )    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)) )
    return g.ravel()


class Star:
    def __init__(self, data):
        self.name = data["Name"]                               ### Name
        self.Ms, self.e_Ms = None, None                        ### Stellar mass [Solar mass]
        self.Rs, self.e_Rs = None, None                        ### Stellar radius [Solar radius]
        self.Rhos, self.e_Rhos = None, None                    ### Stellar density [Solar density]
        self.LDD, self.e_LDD = data.get("LDD", 1), data.get("e_LDD", 0.01)      ### Limb-darkened angular diameter [mas]
        self.plx, self.e_plx = data.get("plx", 10), data.get("e_plx", 0.1)      ### Gaia parallax [mas]
        
        ### Compute the stellar radius [Rsun]
        self.Rs = auToRsun( self.LDD / self.plx /2 )
        self.e_Rs = self.Rs * np.sqrt( (self.e_plx/self.plx)**2 + (self.e_LDD/self.LDD)**2 )
    def set_param( self, param, value ):
        if param == "Ms":
            self.Ms = value
        elif param == "e_Ms":
            self.e_Ms = value
        elif param == "Rs":
            self.Rs = value
        elif param == "e_Rs":
            self.e_Rs = value
        elif param == "Rhos":
            self.Rhos = value
        elif param == "e_Rhos":
            self.e_Rhos = value
        else :
            print('Unknown parameter to set. Possible values are "Ms", "Rs" or "Rhos" and "e_..." for their error.')
    
    def __repr__(self):
        return (
            f"===========================================\n"
            f"Star object (as defined in EXO) \n"
            f"Main properties :\n"
            f"M= {self.Ms} Msun | R= {self.Rs} Rsun \n"
            f"RV computed ? --> {self.RV_computed}\n"
            f"===========================================\n"
        )

class Planet :
    def __init__( self  ):#, *args, **kwargs ):
        # super().__init__(*args, **kwargs)
        self.Mp, self.e_Mp = None, None     ### Planetary mass [earth mass]
        self.Rp, self.e_Rp = None, None     ### Planetary earth [earth radius]
        self.Rhop, self.e_Rhop = None, None ### Planetary density [earth density]
        self.transit_computed, self.RV_computed = False, False
        
    def set_param( self, param, value ):
        if param == "Mp":
            self.Mp = value
        elif param == "e_Mp":
            self.e_Mp = value
        elif param == "Rp":
            self.Rp = value
        elif param == "e_Rp":
            self.e_Rp = value
        elif param == "Rhop":
            self.Rhop == value
        elif param == "e_Rhop":
            self.e_Rhop == value
        else :
            print('Unknown parameter to set. Possible values are "Mp", "Rp" or "Rhop" and "e_..." for their error.')
    
    
    def transit( self, star, params_transit, force=False ) :
        ### Computes Planetary and Stellar parameters from transit observations
        ### Need to update formulas and errors accounting for 
        ### dF [ppm]
        ### tauT/tauF [min]
        ### Porb [days]
        ### ecc
        ### omega [degrees]
        
        if ((not self.transit_computed) or (self.transit_computed and force) ) :
            self.dF, self.e_dF = params_transit.get("dF",0), params_transit.get("e_dF",0)                       ### Transit depth [ppm]
            self.tauT, self.e_tauT = params_transit.get("tauT",0)*60/86400, params_transit.get("e_tauT",0)*60/86400   ### Transit total duration [minutes]
            self.tauF, self.e_tauF = params_transit.get("tauF",0)*60/86400, params_transit.get("e_tauF",0)*60/86400   ### Transit flat duration [minutes]
            self.P, self.e_P = params_transit.get("Porb",np.inf), params_transit.get("e_Porb",0)                ### Transit Period [days]
            ### Compute planet radius [Stellar radii]
            self.Rp_Rs, self.e_Rp_Rs = np.sqrt(self.dF*1E-6), self.e_dF*1E-6/2/np.sqrt(self.dF*1E-6)
            
            ### Compute the perturbation due to eccentricity
            if (not self.RV_computed) :
                print("!!! WARNING !!!")
                print("With RV not computed, eccentricity taken from inputs or assumed = 0")    
                self.ecc, self.e_ecc = params_transit.get("ecc",0), params_transit.get("e_ecc",0)
                self.ome, self.e_ome = params_transit.get("ome",0), params_transit.get("e_ome",0)
            
            tauTo, e_tauTo = self.tauT, self.e_tauT
            tauFo, e_tauFo = self.tauF, self.e_tauF
            
            self.tauT *= np.sqrt(1-self.ecc**2)/( 1 + self.ecc*np.sin(self.ome) )
            self.tauF *= np.sqrt(1-self.ecc**2)/( 1 + self.ecc*np.sin(self.ome) )
            
            self.e_tauT = self.tauT*np.sqrt( (e_tauTo/tauTo)**2 + ((self.ecc+np.sin(self.ome))/(1+self.ecc*np.sin(self.ome)))**2*self.e_ecc**2/(1-self.ecc**2)**2 + (self.ecc*np.cos(self.ome)/(1+self.ecc*np.sin(self.ome)))**2*self.e_ome**2 )
            self.e_tauF = self.tauF*np.sqrt( (e_tauFo/tauFo)**2 + ((self.ecc+np.sin(self.ome))/(1+self.ecc*np.sin(self.ome)))**2*self.e_ecc**2/(1-self.ecc**2)**2 + (self.ecc*np.cos(self.ome)/(1+self.ecc*np.sin(self.ome)))**2*self.e_ome**2 )
            
            ### Compute the impact parameter [Stellar radii]
            SR = np.sin(np.pi*self.tauF/self.P)/np.sin(np.pi*self.tauT/self.P)
            tanF, tanT = np.tan(np.pi*self.tauF/self.P), np.tan(np.pi*self.tauT/self.P)
            e_SR = np.pi*SR/self.P * np.sqrt( (self.e_tauF/tanF)**2 + (self.e_tauT/tanT)**2
                                  + (self.e_P/self.P)**2*(self.tauT/tanT-self.tauF/tanF)**2 )
            self.b = np.sqrt( ( (1-self.Rp_Rs)**2 - (1+self.Rp_Rs)**2 * SR**2 ) / ( 1 - SR**2 ) )
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
            star.Rhos = self.a_Rs**3 * (RsunToAu(star.Rs)/star.Rs)**3 / dayToYear(self.P)**2
            star.e_Rhos = star.Rhos*np.sqrt((3*self.e_a_Rs/self.a_Rs)**2+(2*self.e_P/self.P)**2)
            
            ### Transit has been computed
            self.transit_computed = True
        else :
            print('Transit parameters already computed, pass "force=True" to overrule.')


    def RV( self, star, params_RV, force=False):
        if not self.transit_computed :
            print("!!! WARNING !!!")
            print("Without transits, Mass values will be an upper limit")    
            self.inc, self.e_inc = np.pi/2, 0
        # Derive parameters from radial velocities
        self.K, self.e_K = params_RV.get('K',0), params_RV.get('e_K',0)

    
    def __repr__(self):
        return (
            f"===========================================\n"
            f"Planet object (as defined in EXO) \n"
            f"Main properties :\n"
            f"M= {self.Mp} (Mearth) | R= {self.Rp} (Rearth) | density= {self.Rhop} (earth)\n"
            f"transit computed ? --> {self.transit_computed}\n"
            f"===========================================\n"
        )

class Likelihood:
    def __init__( self, obj, objtype ):
        self.obj = objtype
        ### Compute the radius and density Probability Density Functions (Gaussians)
        if objtype == "Star":
            if not (obj.Rs and obj.e_Rs):
                print("Missing Star radius or error !")
                print("Wil not be able to compute MLF.")
            else :
                self.f_R = norm( loc=obj.Rs, scale=np.sqrt((obj.e_LDD/obj.LDD)**2+(obj.e_plx/obj.plx)**2) )
                self.f_Rho = norm( loc=obj.Rhos, scale=obj.e_Rhos )
        
        elif objtype == "Planet":
            if not (obj.Rp and obj.e_Rp) :
                print("Missing Planet radius or error !")
                print("Wil not be able to compute MLF.")
            else :
                self.f_R = norm( loc=obj.Rp, scale=obj.e_Rp )
                self.f_Rho = norm( loc=obj.Rhop, scale=obj.e_Rhop )
    
        else :
            print('Invalid objtype Please re-init. Possible values are "Star" or "Planet".')
        
    # def funcR( p, R0, R, sigma_p, sigma_theta ):
    #     theta = p*R/R0
    #     solution = theta*Gaussian(R0*theta/R,0,sigma_p)*Gaussian(theta,0,sigma_theta)
    #     return solution
    
    
    def computeMLF_MR( self, obj, M_range, R_range, Nb_pts=101 ):
        # Computes a Maximum Likelihood Function (MLF) grid as a function of mass and radius
        # obj = Star or Planet object
        # M_range [Msun] = (M_min, M_max) : the range of masses explored
        # R_range [Rsun] = (R_min, R_max) : the range of radii explored
        # Nb_pts = Number of points per axis 
        
        ### Set the grid and relevant parameters
        xMin, xMax = M_range
        yMin, yMax = R_range
        self.M_axis = np.linspace( xMin, xMax, Nb_pts )
        self.R_axis = np.linspace( yMin, yMax, Nb_pts )
        self.M_grid, self.R_grid = np.meshgrid( self.M_axis, self.R_axis )
        self.density_grid = self.M_grid / self.R_grid**3
        if self.obj == "Star":
            self.r_noCorr = (self.M_axis/obj.Rhos)**(1/3)
        elif self.obj == "Planet":
            self.r_noCorr = (self.M_axis/obj.Rhop)**(1/3)
        else :
            print('Invalid objtype, cannot derive mass !')
        
        
        ### Compute the MLF
        self.MLF = 3/4/np.pi/self.R_grid**3 * self.f_R.pdf(self.R_grid) * self.f_Rho.pdf(self.density_grid)
        
        ### We use the statistics of the log-likelihood
        logMLF = np.log(self.MLF)
        ### Use the maximum as the central value
        idx_R, idx_M = np.unravel_index(np.argmax(self.MLF), self.MLF.shape)
        obj.Ms, obj.Rs = self.M_axis[idx_M], self.R_axis[idx_R]
        ### Use log-likelihood properties for uncertainties log(L(sigma)) = log(Lmax) - 1/2
        sigmaR_interv = self.R_grid[logMLF >= np.log(self.MLF.max())-1/2]
        obj.em_Rs, obj.ep_Rs = sigmaR_interv.min(), sigmaR_interv.max()
        sigmaM_interv = self.M_grid[logMLF >= np.log(self.MLF.max())-1/2]
        obj.em_Ms, obj.ep_Ms = sigmaM_interv.min(), sigmaM_interv.max()        























