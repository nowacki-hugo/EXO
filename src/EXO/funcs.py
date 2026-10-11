import numpy as np
import scipy.constants as csts
import EXO.utils as utXO
from scipy.interpolate import RegularGridInterpolator



def f_den(t, X, e_X):
    return np.exp(-(t-X)**2/2/e_X**2) / np.sqrt(2*np.pi*e_X**2)



def f_R( p_grid, R, params ):
    R0 = 0.5 # Value from Crida+ (2018) adapted to the units used
    plx, e_plx, ldd, e_ldd = params
    
    arg_theta = p_grid * R/R0
    plx_term = f_den( p_grid, plx, e_plx )
    th_term = f_den( arg_theta, ldd, e_ldd )
    
    integrand = ( p_grid * plx_term * th_term )
    itg = np.trapz(integrand, p_grid)
    return itg/R0



def likelihood_LT( t_grid, Lval, Tval, params ):
    # Joint likelihood function as developed in Crida+ (2018)
    m = 1000 * utXO.PARSEC/utXO.AU
    prefactor = ( 4 * np.sqrt(np.pi) * m / ( Tval**3 * np.sqrt(csts.sigma * Lval**3) ) )
    arg_plx = np.sqrt( 4 * np.pi * t_grid / Lval )
    arg_theta = np.sqrt( 4 * t_grid / (csts.sigma * Tval**4) )
    
    fbol, e_fbol, plx, e_plx, ldd, e_ldd = params
    
    integrand = ( t_grid
        * f_den(t_grid, fbol, e_fbol)
        * f_den(arg_plx, plx, e_plx)
        * f_den(arg_theta, ldd, e_ldd) )

    integral = np.trapz(integrand, t_grid)
    return prefactor * integral



def likelihood_RM( p_grid, Rval, Mval, params):
    # Joint likelihood function as developed in Crida+ (2018)
    rho, e_rho, plx, e_plx, ldd, e_ldd = params
    
    prefactor = 0.75 / (np.pi*Rval**3)
    arg_rho = 0.75 * Mval / (np.pi*Rval**3)
    den_term = f_R( p_grid, Rval, (plx, e_plx, ldd, e_ldd) ) * f_den(arg_rho, rho, e_rho)

    return prefactor * den_term



def compute_fp( Mp_grid, Rp_grid, Ms_grid, Rs_grid, L_MR_grid, params, block_size=32 ):
    """
    Computes the joint probability density function of planetary mass and radius
    based on the formulae developped in Crida et al. (2018) adapted to avoid 
    discontinuities due to too sharp distributions

    Parameters
    ----------
    Mp_grid [M_earth]: Grid of planetary masses to test
    Rp_grid [R_earth]: Grid of planetary radii to test
    Ms_grid [M_sun]: Grid of stellar masses
    Rs_grid [R_sun]: Grid of stellar radii
    L_MR_grid [--]: Joint distribution L_M*R*(Ms,Rs)
    Ke [m/s]: Observed RV semi-amplitude
    sigmaK [m/s]: Uncertainty on Ke
    TDe [--]: Observed relative transit depth
    sigmaTD [--]: Uncertainty on TDe
    P [seconds]: Orbit period
    inc [rad]: Orbit inclination
    ------
    """
    Ke, sigmaK, TDe, sigmaTD, P, inc = params
    
    
    
    # =============================================================================
    #     Checks
    # =============================================================================
    Mp_grid = np.asarray(Mp_grid*utXO.MEARTH, dtype=np.float64)
    Rp_grid = np.asarray(Rp_grid*utXO.REARTH, dtype=np.float64)
    Ms_grid = np.asarray(Ms_grid*utXO.MSUN, dtype=np.float64)
    Rs_grid = np.asarray(Rs_grid*utXO.RSUN, dtype=np.float64)
    L_MR_grid = np.asarray(L_MR_grid, dtype=np.float64)

    if L_MR_grid.shape != (len(Ms_grid), len(Rs_grid)):
        raise ValueError(
            f"L_MR_grid should be "
            f"({len(Ms_grid)}, {len(Rs_grid)})" )



    # =============================================================================
    #   Converting to log scales
    # =============================================================================   
    C = (2. * np.pi * np.sin(inc)**3 * utXO.G / P) ** (1. / 3.)
    
    K_min = C* np.min(Mp_grid) / np.max(Ms_grid)**(2/3)
    K_max = C* np.max(Mp_grid) / np.min(Ms_grid)**(2/3)
    NlogK = len(Ms_grid)
    logK_grid = np.linspace( np.log(K_min), np.log(K_max), NlogK )
    
    TD_min = ( np.min(Rp_grid) / np.max(Rs_grid) )**2
    TD_max = ( np.max(Rp_grid) / np.min(Rs_grid) )**2
    NlogTD = len(Rs_grid)
    logTD_grid = np.linspace( np.log(TD_min), np.log(TD_max), NlogTD )

    dlogK = np.gradient(logK_grid)
    dlogTD = np.gradient(logTD_grid)



    # =============================================================================
    #     Converting MLF in observables space
    # =============================================================================
    LOGK, LOGTD = np.meshgrid( logK_grid, logTD_grid, indexing="ij" )
    logLobs = -0.5 * ((np.exp(LOGK)-Ke)/sigmaK)**2 - 0.5*((np.exp(LOGTD)-TDe)/sigmaTD)**2
    logLobs -= np.max(logLobs)
    Lobs = np.exp( logLobs )
    Wobs = ( Lobs * dlogK[:,None] * dlogTD[None,:] )

    
    
    # =============================================================================
    #     Interpolation on grids pre-computed to match Mp/Rp grids
    # =============================================================================
    interp = RegularGridInterpolator( (Ms_grid, Rs_grid), L_MR_grid, bounds_error=False, fill_value=0.0 )

    
    
    # =============================================================================
    #     Integration by blocks
    # =============================================================================
    fp = np.zeros( (len(Mp_grid), len(Rp_grid)) )
    
    for i0 in range(0, len(Mp_grid), block_size):
        i1 = min( i0+block_size, len(Mp_grid) )
        for j0 in range(0, len(Rp_grid), block_size):
            j1 = min( j0+block_size, len(Rp_grid) )

            Mp = Mp_grid[i0:i1]
            Rp = Rp_grid[j0:j1]

            Ms_eff = ( C**1.5 * (Mp[:,None,None,None])**1.5 * np.exp( -1.5*LOGK[None,None,:,:] ) )
            Rs_eff = ( (Rp[None,:,None,None]) * np.exp( -0.5*LOGTD[None,None,:,:] ) )

            Ms_eff, Rs_eff = np.broadcast_arrays( Ms_eff, Rs_eff)
            pts = np.stack( ( Ms_eff.ravel(), Rs_eff.ravel() ), axis=-1 )
            # (nMp,nRp,nK,nTD,2)
            pts = pts.reshape(-1,2)
            L = interp(pts).reshape( Ms_eff.shape )
            # L = np.exp(-()/2/)
            
            J = ( 0.75 * Ms_eff * Rs_eff )
                        
            fp[i0:i1,j0:j1] = np.sum( L * J * Wobs[None,None,:,:], axis=(2,3) )
    
    return fp
