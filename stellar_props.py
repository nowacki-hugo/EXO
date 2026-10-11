#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example showcasing the capabilities of the EXO code
in deriving stellar parameters from a diversity of datasets

@author: Hugo Nowacki
"""

# =============================================================================
# First step : create a 'star' object with the known properties
# =============================================================================
import EXO as XO

STAR = {
    "Name":"14 And",
    "LDD" : 1.323,          # Lim-darkened diameter [mas]
    "e_LDD" : 0.024,        # error [mas]
    "plx" : 13.17,          # Gaia parallax [mas]
    "e_plx": 0.07,          # error [mas]
    "Fbol": 32.85e-8,       # Bolometric flux [erg/cm2/s]
    "e_Fbol": 0.75e-8}      # error [erg/cm2/s]

star = XO.Star( STAR )

# =============================================================================
# Second step : Derive some stellar properties
# =============================================================================
import numpy as np
# EXO relies on a probabilistic approach, and therefore requires a range of 
# values to explore. The code will compute the maximum likelihood function /
# probability density function associated to the quantities provided.

# Luminosity/Temperature --> requires LDD, plx and Fbol
Ls, Ts = np.linspace( 20, 100, 501), np.linspace( 4500, 5200, 201 )
star.computeMLF_LT( Ls, Ts, savefig='14And_LT.jpg' )
best_L = star.Ls                                # Best luminosity [Lsun]
unc_L = 0.5 * ( star.ep_Ls + star.em_Ls )       # Uncertainty [Lsun]
best_T = star.Teff                              # Best Teff [K]
unc_T = 0.5 * ( star.ep_Teff + star.em_Teff )   # Uncertainty [Lsun]

# Radius --> requires LDD, plx
Rs = np.linspace( 5, 15, 501 )
star.compute_radius_pdf( Rs, plot=True )

# Habitable zone --> requires either luminosity or radius + temperature
# Luminosity, radius and temperature can be computed with previous functions
# or be set manually with star.Ls = XX [Lsun] or star.Rs = YY [Rsun]
ainf, e_ainf, asup, e_asup = star.compute_HZ( method='optimistic' ) # Possible methods = 'optimistic'/'conservative'
# Returns the inferior edge, the superior edge, and their uncertainties


