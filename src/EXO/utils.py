import numpy as np

# =============================================================================
# Constants (SI), values set by IAU 2015 resolution B3 from Prsa (2016)
# =============================================================================
G = 6.67428e-11
RSUN = 6.957e8  # at the Equator
MSUN = 1.3271244e20 / G
RHOSUN = 3/4/np.pi*MSUN/RSUN**3
REARTH = 6.3781e6 # at the Equator
MEARTH = 3.986004e14 / G
TSUN = 5772.0  # +/- 0.8
RJUP = 7.1492e7 # at the Equator
MJUP = 1.2668653e17 / G
LSUN = 3.828e26  # in watts
### Astronomical unit from IAU definition of 2012
AU = 149597870700
PARSEC = AU * 648000/np.pi
### erg/s in watts
ERG_S = 1e-7
# =============================================================================
# Conversion functions
# =============================================================================
def Rsun2Au(Rsun):
    au = Rsun * (RSUN/AU)
    return au
def au2Rsun(au):
    Rsun = au * (AU/RSUN)
    return Rsun
def au2Mas(au, plx):
    mas = 2*au*plx
    return mas
def Mas2Au(mas, plx):
    # plx in mas
    au = mas/plx/2
    return au
def RsunToRad(Rsun, plx):
    Rad = mas2rad(au2Mas(Rsun2Au(Rsun), plx))
    return Rad
def RsunToRearth( Rsun ):
    Rearth = Rsun * (RSUN/REARTH)
    return Rearth
def RearthToRsun( Rearth ) :
    Rsun = Rearth * (REARTH/RSUN)
    return Rsun
def MsunToMearth( Msun ):
    Mearth = Msun * (MSUN/MEARTH)
    return (Mearth)
def MearthToMsun( Mearth ):
    Msun = Mearth * (MEARTH/MSUN)
    return (Msun)
    
def radToDeg(rad):
    deg = rad*180/np.pi
    return deg
def degToRad(deg):
    rad = deg/180*np.pi
    return rad
def mas2rad(mas):
    rad = mas /1000 /3600 *np.pi/180
    return(rad)
def rad2mas(rad):
    mas = rad *180/np.pi *3600 *1000
    return(mas)
def daysToYear(days):
    years = days/365.25
    return years
def yearToDays(years):
    days = years*365.25
    return days
def daysToSec(days):
    sec = days*86400
    return(sec)
def ergs_s_cm2ToWatts(ergs):
    Watts = ergs * ERG_S / (10**-2)**2
    return Watts
def ergs_s_cm2ToLsun(ergs):
    Lsun = ergs * ERG_S / (10**-2)**2 / LSUN
    return Lsun

def BIC(RSS, Ndata, Nfree):
    # RSS = Sum_i^Ndata ( D_i - M_i )**2 Residual sum of squares
    # Ndata = Number of data points
    # Nfree = Number of free parameters
    return Ndata*np.log(RSS/Ndata) + Nfree*np.log(Ndata)

# =============================================================================
# Basic functions
# =============================================================================
def Logg(M_Msun, e_M_Msun, R_Rsun, e_R_Rsun):
    logg = np.log10( G*M_Msun*MSUN/(R_Rsun*RSUN)**2 * 100)
    e_logg = np.sqrt( (e_M_Msun/M_Msun)**2 + (e_R_Rsun/R_Rsun/2)**2 )/np.log(10)
    return(logg, e_logg)

