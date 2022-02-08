"""
This module code contains functions that are used to bias adjust model
runs from the Climate Model Intercomparison Project 5 (CMIP5), using methods published by 
[Lange (2019)](doi.org/10.5194/gmd-12-3055-2019). Some functions are adapted from code found 
in the [isimip3 Git repository](https://github.com/ssobie/isimip3). As such, this module is 
licenced following [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.en.html).
This license compatible with the MIT [LICENSE] employed in this repository. Source code, from 
which this software has been modified, is provided in the [/isimip3-source/](./isimip3-source/) directory.

Author: [Daniel Gilford](https://github.com/dgilford)
Last updated: 2/8/2022 by Daniel Gilford
"""

# import modules to use in functions
import numpy as np
import xarray as xr

### ------------------- XARRAY WRAPPER ENABLING GLOBALLY-RESOLVED BA ------------------- ###

# Function using apply_ufunc to calculate a month's 
def xr_bias_adjust_singlemonth(obstrain,simtrain,simfuture,options=None):
    """ 
    This function calculates the bias adjustment for a single month
    using xarray.
    
    obstrain,simtrain,simfuture are all datasets with only a single
    calendar month's data timeseries (over all years considered)
    
    options are the user defined choices for the bias correction code
    """
    
    # rename the time index for the simulated future (in case it doesn't match the training)
    simfuture=simfuture.rename({'time': 'time2'})
    
    # get the year grids
    train_yrgrid=obstrain['time.year']
    future_yrgrid=simfuture['time2.year']
    
    # apply the bias correction code over all lat/lon locations
    # with xr.apply_ufunc
    result = xr.apply_ufunc(
        
        bias_correct_singleloc,
        
        obstrain, simtrain, simfuture,
        train_yrgrid, future_yrgrid,
        kwargs=dict(options=options),
        
        input_core_dims=[
            ['time'], ['time'], ['time2'],
            ['time'], ['time2'],
        ],
        output_core_dims=[
            ['time2'],
        ],
        vectorize=True
    )
    
    
    # rename the time index back to just 'time'
    result=result.rename({'time2': 'time'})
    
    return result

### ------------------- BIAS ADJUSTMENT AT A SINGLE LOCATION ------------------- ###
    
    
def bias_correct_singleloc(obstrain_loc,simtrain_loc,simfuture_loc,train_yrgrid,future_yrgrid,options):
    """ 
    This function calculates the bias adjustment for a single location.
    
    obstrain_loc,simtrain_loc,simfuture_loc are datasets with data at 
    a single location and calendar month's data timeseries (over all years considered)
    
    train_yrgrid, future_yrgrid are the grids of year values related to the two input
    timeseries (time and time2)
    
    If there is only missing data at this lat/lon, bias correction is skipped
    and the simfuture_loc is returned, with all missing values (np.nan)
    
    options are the user defined choices for the bias correction code
    """
    
    # Check for whether there is only missing data at this lat/lon
    # or if there are any missing years in the observations
    #print(np.count_nonzero(~np.isnan(simtrain_loc)))
    if np.count_nonzero(~np.isnan(simtrain_loc))<=0 \
        or np.count_nonzero(~np.isnan(obstrain_loc))<len(obstrain_loc) \
        or np.count_nonzero(~np.isnan(simfuture_loc))<len(simfuture_loc):
#         print('skip this lat/lon')
        # if we have all missing data, return the bias correction as missing data
        return(np.copy(simfuture_loc))
#     else:
#         print('i='+str(int(np.count_nonzero(~np.isnan(obstrain_loc)))))
    
    # get the slopes, and trend+detrended timeseries at this location
    slopes,detrend_ts,trend_ts=uf.detrend_each_ts_ufunc(obstrain_loc, simtrain_loc, simfuture_loc, train_yrgrid, future_yrgrid)
    
    # do the bias correction for this month and location
    simfutureBA_DT=ba.map_quantiles_parametric_trend_preserving(
        detrend_ts['x_obs_train_detrend'], detrend_ts['x_sim_train_detrend'], detrend_ts['x_sim_future_detrend'], 
        distribution=options['distribution'], trend_preservation=options['trend_preservation'],
    )
    
    # restore the trend to the bias adjusted distribution at this location
    simfutureBA_loc = uf.subtract_or_add_trend(simfutureBA_DT, future_yrgrid, trend_ts['x_sim_future_trend'])
    
    return(simfutureBA_loc)
    