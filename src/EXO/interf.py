import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 
from joblib import Parallel, delayed
import multiprocessing
import oimodeler as oim
from oimodeler.oimUtils import hdulistDeepCopy
import scipy.special as sp
from scipy.integrate import cumulative_trapezoid

mpl.rc('text', usetex=True)
mpl.rcParams.update({'font.size': 15})



class FitterGrid:
    """
    1D regular-grid fitter

    """
    def __init__( self, data, model, params, min, max, steps, dataTypes=None, bands=None, progress=True ):
        self.data = data
        self.model = model
        self.params = params
        self.gridMin = min
        self.gridMax = max
        self.gridSteps = steps
        self.dataTypes = dataTypes
        self.progress = progress
        self.bands = bands # Names of the datasets/instruments. If None, automatically detected from INSNAME.
        self._prepare_grid()
        self._split_data()
        
        
        
    # ------------------------------------------------------------------
    # GRID
    # ------------------------------------------------------------------
    def _prepare_grid( self ):
        if len(self.params) != 1:
            raise ValueError( "This fitter is intended for a 1D grid: "
                "exactly one parameter is required." )
        if not ( len(self.gridMin) == len(self.gridMax) == len(self.gridSteps) == 1 ):
            raise ValueError( "min, max and steps must contain one value each." )
        self.grid = np.arange( self.gridMin[0], self.gridMax[0] + self.gridSteps[0], self.gridSteps[0] )
        
        
        
    # ------------------------------------------------------------------
    # DATA SPLITTING
    # ------------------------------------------------------------------
    def _get_file_instruments( self, hdulist ):
        instruments = set()
        for hdu in hdulist:
            if hasattr( hdu, "header" ) and "INSNAME" in hdu.header:
                instruments.add( hdu.header["INSNAME"] )
        return instruments
    
    def _split_data( self ):
        groups = {}
        for hdulist in self.data.data:
            instruments = self._get_file_instruments( hdulist )
            if len(instruments) == 0:
                raise ValueError( "Could not find INSNAME in one of the OIFITS files." )
            if len(instruments) > 1:
                raise ValueError( "One OIFITS file contains several INSNAME values. " 
                                 "The automatic band splitting is ambiguous." )
            instrument = list(instruments)[0]
            if instrument not in groups:
                groups[instrument] = []
            groups[instrument].append( hdulistDeepCopy(hdulist) )
        if self.bands is None:
            self.bands = list( groups.keys() )
        missing = [b for b in self.bands if b not in groups]
        if missing:
            raise ValueError( f"The following requested bands were not found: {missing}. "
                f"Available bands are: {list(groups.keys())}" )
        self.band_data = { band: oim.oimData( groups[band] ) for band in self.bands }
        
        
        
    # ------------------------------------------------------------------
    # SINGLE-BAND METHODS
    # ------------------------------------------------------------------
    def _compute_band_grid( self, band ):
        data = self.band_data[band]
        fitter = oim.oimFitterRegularGrid( data, self.model, dataTypes=self.dataTypes, )
        fitter.prepare( params=self.params, min=self.gridMin, max=self.gridMax, steps=self.gridSteps, )
        fitter.run( progress=self.progress )
        return fitter
    
    @staticmethod
    def _get_uncertainties_from_chi2grid( grid, chi2_map ):
        imin = np.argmin( chi2_map )
        chi2min = chi2_map[imin]
        theta_best = grid[imin]
        # chi2_norm = chi2 / chi2_min  and  uncertainty region: chi2_norm <= 2  =>  chi2 <= 2xchi2_min
        accepted = chi2_map <= 2.0 * chi2min
        theta_accepted = grid[accepted]
        if theta_accepted.size == 0:
            raise RuntimeError( "No points found in the chi2 <= 2*chi2_min interval." )
        theta_min = theta_accepted.min()
        theta_max = theta_accepted.max()
        err_minus = theta_best - theta_min
        err_plus = theta_max - theta_best
        delta_theta = 0.5 * ( err_minus + err_plus )
        return ( theta_best, err_minus, err_plus, delta_theta, chi2min )

    def _fit_bands( self ):
        self.band_results = {}
        for band in self.bands:
            if self.progress:
                print( f"\nProcessing {band}" )
            fitter = self._compute_band_grid( band )
            chi2r = np.asarray( fitter.chi2rMap )
            theta_best, err_minus, err_plus, delta_theta, chi2r_min = self._get_uncertainties_from_chi2grid( self.grid, chi2r )
            # Number of measurements entering the chi2
            fitter.getResults()
            simulator = fitter.simulator
            chi2_min = simulator.chi2
            ndata = simulator.degFree + len( self.model.getFreeParameters() )
            chi2r_best = chi2_min / simulator.degFree
            
            self.band_results[band] = { "fitter": fitter, "chi2r_grid": chi2r, "chi2_grid": chi2r * simulator.degFree,
                "theta_best": theta_best, "err_minus": err_minus, "err_plus": err_plus, "delta_theta": delta_theta,
                "chi2_min": chi2_min, "chi2r_min": chi2r_best, "ndata": ndata, "dof": simulator.degFree }
    
    
    
    
    # ------------------------------------------------------------------
    # COMBINED GRID
    # ------------------------------------------------------------------
    def _compute_combined_grid( self ):
        self._fit_bands()
        chi2s = { band: self.band_results[band]["chi2_grid"] for band in self.bands }
        self.chi2_weighted = sum( chi2s.values() )
        nfree = len( self.bands ) * len( self.model.getFreeParameters() )
        ndata_total = sum( result["ndata"] for result in self.band_results.values() )
        self.dof_total = ndata_total - nfree
        self.chi2r_weighted = self.chi2_weighted / self.dof_total

    # ------------------------------------------------------------------
    # COMBINED GRID
    # ------------------------------------------------------------------
    def _compute_best_model( self ):
        best_model=123456
        return best_model


    # ------------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------------
    def plotChi2Maps( self, normalized=False ):
        nplots = len( self.bands ) + 1
        fig, axes = plt.subplots( nplots, 1, figsize=(8, 3.5 * nplots), sharex=True )
        # Make axes iterable also when there is only one plot
        axes = np.atleast_1d( axes )
        # --------------------------------------------------------------
        # Individual bands
        # --------------------------------------------------------------
        for ax, band in zip( axes[:-1], self.bands ):
            result = self.band_results[band]
            chi2 = result["chi2r_grid"]
            theta_best = result["theta_best"]
            delta_theta = result["delta_theta"]
            chi2_min = result["chi2r_min"]
            if normalized:
                chi2_plot = chi2 / chi2_min
                threshold = 2.0
                ylabel = r"$\chi^2_r / \chi^2_{r,\min}$"
            else:
                chi2_plot = chi2
                threshold = 2.0 * chi2_min
                ylabel = r"$\chi^2_r$"
            ax.plot( self.grid, chi2_plot, label=band )
            # Best-fit value
            ax.axvline( theta_best, linestyle="--" )
            ax.axvline( self.theta_best, linestyle="dotted", color='red' )
            # Uncertainty region
            if normalized:
                ax.axhline( threshold, linestyle=":" )
            else:
                ax.axhline( threshold, linestyle=":" )
            ax.axvspan( result["theta_best"] - result["err_minus"], result["theta_best"] + result["err_plus"], alpha=0.15 )
            ax.set_ylabel( ylabel ), ax.set_yscale( 'log' )
            ax.set_title( f"{band}  " rf"($\theta_{{\rm best}}$ = "
                         f"${theta_best:.3g}\pm{delta_theta:.3g}$ ; $\chi^2_r={chi2_min:.3f}$)" )
            ax.grid( alpha=0.3 )
        # --------------------------------------------------------------
        # Combined map
        # --------------------------------------------------------------
        ax = axes[-1]
        chi2 = self.chi2r_weighted
        theta_best = self.theta_best
        delta_theta = self.delta_theta
        chi2_min = self.chi2r_min
        if normalized:
            chi2_plot = chi2 / chi2_min
            threshold = 2.0
            ylabel = r"$\chi^2_r / \chi^2_{r,\min}$"
        else:
            chi2_plot = chi2
            threshold = 2.0 * chi2_min
            ylabel = r"$\chi^2_r$"
        ax.plot( self.grid, chi2_plot, linewidth=2 )
        ax.axvline( theta_best, linestyle="--" )
        ax.axhline( threshold, linestyle=":" )
        ax.axvspan( theta_best - self.err_minus, theta_best + self.err_plus, alpha=0.15 )
        ax.set_xlabel( 'LDD (mas)' ), ax.set_ylabel( ylabel ), ax.set_yscale( 'log' )
        ax.set_title( "Combined  " rf"($\theta_{{\rm best}}$ = "
                     f"${theta_best:.3g}\pm{delta_theta:.3g}$ ; $\chi^2_r={chi2_min:.3f}$)" )
        ax.grid( alpha=0.3 ), plt.tight_layout( )
        plt.show( )

    def plot_V2_curve():
        fig, ax = plt.subplots()
        return fig

    # ------------------------------------------------------------------
    # EXTRACTING RESULTS
    # ------------------------------------------------------------------
    def getResults( self ):
        theta = self.theta_best
        ep_theta, em_theta = self.err_plus, self.err_minus
        return theta, ep_theta, em_theta

    def printResults( self, format=".3f" ):
        print( "-" * 70 )
        print( "Individual datasets" )
        for band in self.bands:
            r = self.band_results[band]
            print( f"{band:12s} : "
                f"{self.params[0].name} = "
                f"{r['theta_best']:{format}} "
                f"+{r['err_plus']:{format}} "
                f"-{r['err_minus']:{format}} "
                f"{self.params[0].unit}"
                f"{'':5s} chi2 = {r['chi2_min']:.2f}"
                )
        print("-" * 70)
        comb='Combined'
        print( f"{comb:12s} : "
            f"{self.params[0].name} = "
            f"{self.theta_best:{format}} "
            f"+{self.err_plus:{format}} "
            f"-{self.err_minus:{format}} "
            f"{self.params[0].unit}"
            f"{'':5s} chi2r = {self.chi2r_min:.2f}")
        
        
        
        
    # ------------------------------------------------------------------
    # RUN
    # ------------------------------------------------------------------
    def run( self, plot=False ):
        self._compute_combined_grid()
        theta_best, err_minus, err_plus, delta_theta, chi2r_min = self._get_uncertainties_from_chi2grid( self.grid, self.chi2r_weighted )
        self.theta_best = theta_best
        self.err_minus = err_minus
        self.err_plus = err_plus
        self.delta_theta = delta_theta
        self.chi2r_min = chi2r_min
        if plot :
            self.plotChi2Maps( normalized=True )
            self.plot_V2_curve()
        self.printResults()
        return self
    
    
            
        
        