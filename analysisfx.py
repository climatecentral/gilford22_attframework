"""
This module code contains the analysis functions that we use throughout the attribution framework codebase, by first appending the directory that contains this set of analysis functions, and then calling the particular function we want to use.

Author: [Daniel Gilford](https://github.com/dgilford)
Last updated: 5/31/2022 by Daniel Gilford
"""

# import modules to use in functions
import numpy as np
import xarray as xr

# Function to calculate the regression between a global xarray and a timeseries
def global_xr_regression(yx,ts,missnum=-999):
    
    # import the linear regression module from scipy
    from scipy.stats import linregress
    
    # regress the global annual DataArray, yx(t), against the timeseries, ts
    # note that y should lead x (lat before lon)
    
    # create the output arrays
    slope_yx = xr.full_like(yx.groupby('time.month').mean().squeeze().copy(deep='True'),np.nan,dtype='float')
    stderr_yx=slope_yx.copy(deep=True)
    
    
    # find the valid locations
    validt=yx.where(yx>missnum)
    
    # loop over latitudes and longitude
    for xi in range(len(yx.lon)):
        for yi in range(len(yx.lat)):
            # find the locations where we have valid years
            # if at least one, calculated the regression and parameters
            validi=~np.isnan(validt[:,yi,xi])
            if np.sum(validi)>0:
                slope_yx[yi,xi],_,_,_,stderr_yx[yi,xi] = linregress(ts[validi.drop('time')],validt[validi.drop('time'),yi,xi])
            del validi
    
    # go back to the above program level
    return(slope_yx,stderr_yx)