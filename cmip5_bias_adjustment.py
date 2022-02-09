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
def xr_bias_adjust_singlemonth(xobscal,xsimcal,xsimadj,options=None):
    """ 
    This function is an xarray wrapper that enables bias adjustment for a single month
    of climate model data over all space (lat/lon). The model data should be in an xarray
    dataset format. Needed are the observations over the calibration period (xobscal),
    and model run to be calibrated over the calibration period (xsimcal) and bias adjusted
    over the adjustment period (xsimadj)
    
    xobscal, xsimcal, and xsimadj are all xarray datasets with only a single
    calendar month's data timeseries (over all years considered)
    
    options are the user defined choices for the bias adjustment code from Lange
    """
    
    # rename the time index temporarily for the simulated adjustment dataset to work with xarray
    # (in case it doesn't match the calibration period, which it often will not)
    xsimadj=xsimadj.rename({'time': 'time_tmp'})
    
    # store the year grids for the calibration and adjustment periods
    cal_yrgrid=xobscal['time.year']
    adj_yrgrid=xsimadj['time_tmp.year']
    
    # apply the bias adjustment code over all lat/lon locations
    # using the xr.apply_ufunc feature
    result = xr.apply_ufunc(
        
        # extend the function written to apply at a single location
        # to be applied at all input geospatial locations
        bias_adjust_singleloc,
        
        # define the input datasets
        xobscal, xsimcal, xsimadj,

        # define the time grids that the datasets lie along
        cal_yrgrid, adj_yrgrid,

        # pass the options dictionary as an argument to the "singleloc" function
        kwargs=dict(options=options),
        
        # define the time dimensions of the inputs and outputs
        input_core_dims=[
            ['time'], ['time'], ['time_tmp'],
            ['time'], ['time_tmp'],
        ],
        output_core_dims=[
            ['time_tmp'],
        ],

        # vectorize the function manually
        vectorize=True
    )
    
    # after receiving the geospatial output from apply_ufunc
    # defined along the temporary time grid, 
    # rename the time index back to just 'time'
    result=result.rename({'time_tmp': 'time'})
    
    # go back to the above program level, returning the result
    return result

### ------------------- BIAS ADJUSTMENT AT A SINGLE LOCATION ------------------- ###
    
def bias_adjust_singleloc(xobscal_loc,xsimcal_loc,xsimadj_loc,cal_yrgrid,adj_yrgrid,options):
    """ 
    This function calculates the bias adjustment for a single location
    
    xobscal_loc,xsimcal_loc,xsimadj_loc are datasets with data at 
    a single location and calendar month's data timeseries (over all years considered)
    
    cal_yrgrid, adj_yrgrid are the grids of year values related to the two input
    timeseries (time and time_tmp), respectively

    Results are returned in the xsimadj_BA xarray dataset, and share the same shape
    as xsimadj_loc
    
    If there is only missing data at this geolocation, bias adjustment is skipped
    and the xsimadj_BA is returned, with all missing values (np.nan)
    
    options are the user defined choices for the bias adjustment code from Lange
    """
    
    # Check for whether there is only missing data at this lat/lon
    # or if there are any missing years in the observations
    if np.count_nonzero(~np.isnan(xsimcal_loc))<=0 \
        or np.count_nonzero(~np.isnan(xobscal_loc))<len(xobscal_loc) \
        or np.count_nonzero(~np.isnan(xsimcal_loc))<len(xsimcal_loc):
        # if we have all missing data, return the bias adjustment as missing data
        return(np.copy(xsimadj_loc))
    
    # detrend the timeseries at this location for input to the Lange BA
    _,detrend_ts,trend_ts=detrend_each_ts_ufunc(xobscal_loc, xsimcal_loc, xsimadj_loc, cal_yrgrid, adj_yrgrid)
    
    # do the bias adjustment for this month and location
    simfutureBA_DT=map_quantiles_parametric_trend_preserving(
        detrend_ts['x_obs_train_detrend'], detrend_ts['x_sim_train_detrend'], detrend_ts['x_sim_future_detrend'], 
        distribution=options['distribution'], trend_preservation=options['trend_preservation'],
    )
    
    # restore the trend to the bias adjusted distribution at this location
    xsimadj_BA = uf.subtract_or_add_trend(simfutureBA_DT, future_yrgrid, trend_ts['x_sim_future_trend'])
    
    return(xsimadj_BA)
    
### ------------------- BIAS ADJUSTMENT UTILITIES ------------------- ###

# Function to detrend each timeseries of the three we are concerned with
def detrend_each_ts_ufunc(x_obs_train, x_sim_train, x_sim_fut,train_yrgrid,future_yrgrid):

    x_obs_train_DETREND,x_obs_train_TREND=subtract_or_add_trend(x_obs_train,train_yrgrid, trend=None) 
    x_sim_train_DETREND,x_sim_train_TREND=subtract_or_add_trend(x_sim_train, train_yrgrid, trend=None)
    x_sim_fut_DETREND,x_sim_fut_TREND=subtract_or_add_trend(x_sim_fut, future_yrgrid, trend=None)

    # store detrend and trend timeseries
    detrend_ts={
        'x_obs_train_detrend':x_obs_train_DETREND,
        'x_sim_train_detrend':x_sim_train_DETREND,
        'x_sim_future_detrend':x_sim_fut_DETREND,
    }
    trend_ts={
        'x_obs_train_trend':x_obs_train_TREND,
        'x_sim_train_trend':x_sim_train_TREND,
        'x_sim_future_trend':x_sim_fut_TREND,
    }
    
    # calculate and store the trends
    trends={
        'obs_train_CperDecade':(x_obs_train_TREND[1]-x_obs_train_TREND[0])*10,
        'sim_train_CperDecade':(x_sim_train_TREND[1]-x_sim_train_TREND[0])*10,
        'sim_future_CperDecade':(x_sim_fut_TREND[1]-x_sim_fut_TREND[0])*10,
    }
    
    # go back to the above program
    return(trends,detrend_ts,trend_ts)

def subtract_or_add_trend(x, years, trend=None):
    """
    Subtracts or adds trend from or to x.

    Parameters
    ----------
    x : array
        Time series.
    years : array
        Years of time points of x used to subtract or add trend at annual
        temporal resolution.
    trend : array, optional
        Trend line. If provided then this is the trend line added to x.
        Otherwise, a trend line is computed and subtracted from x

    Returns
    -------
    y : array
        Result of trend subtraction or addition from or to x.
    trend : array, optional
        Trend line. Is only returned if the parameter trend is None.

    """
    assert x.size == years.size, 'size of x != size of years'
    unique_years = np.unique(years)

    # compute trend
    if trend is None:
        annual_means = np.array([np.mean(x[years == y]) for y in unique_years])
        a, b = linlstsq(unique_years, annual_means)
        trend = a * (unique_years - np.mean(unique_years))
        return_trend = True
    else:
        msg = 'size of trend array != number of unique years'
        assert trend.size == unique_years.size, msg
        trend = -trend
        return_trend = False

    # subtract or add trend
    y = np.empty_like(x)
    for i, year in enumerate(unique_years):
        is_year = years == year
        y[is_year] = x[is_year] - trend[i]

    # return result(s)
    if return_trend:
        return y, trend
    else:
        return y