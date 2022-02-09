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
    _,detrend_ts,trend_ts=detrend_each_ts(xobscal_loc, xsimcal_loc, xsimadj_loc, cal_yrgrid, adj_yrgrid)
    
    # do the bias adjustment for this month and location
    simadjBA_DT=map_quantiles_parametric_trend_preserving(
        detrend_ts['x_obs_cal_detrend'], detrend_ts['x_sim_cal_detrend'], detrend_ts['x_sim_adj_detrend'], 
        distribution=options['distribution'], trend_preservation=options['trend_preservation'],
    )

    # restore the trend to the bias adjusted distribution at this location
    xsimadj_BA = subtract_or_add_trend(simadjBA_DT, adj_yrgrid, trend_ts['x_sim_adj_trend'])
    
    # go back to the above program level, returning the bias adjusted dataset
    return(xsimadj_BA)


### ------------------- UNDERLYING LANGE (2019) BIAS ADJUSTMENT MODULE ------------------- ###

# reference the utility functions file directly from the Lange source
import sys
sys.path.append("./isimip3-source/")
import utility_functions as uf

# import other functions necessary for the Lange bias adjustment module
import scipy.stats as sps
import warnings

# Function to perform bias adjustment calculations
def map_quantiles_parametric_trend_preserving(
        x_obs_hist, x_sim_hist, x_sim_fut, 
        distribution='normal', trend_preservation='additive',
        n_quantiles=50, p_value_eps=1e-10,
        max_change_factor=100., max_adjustment_factor=9.,
        adjust_p_values=False,
        lower_bound=None, lower_threshold=None,
        upper_bound=None, upper_threshold=None):
    """
    Adjusts biases using the trend-preserving parametric quantile mapping
    method described in Lange (2019) <https://doi.org/10.5194/gmd-2019-36>.

    Parameters
    ----------
    x_obs_hist : array
        Time series of observed climate data representing the historical or
        training time period.
    x_sim_hist : array
        Time series of simulated climate data representing the historical or
        training time period.
    x_sim_fut : array
        Time series of simulated climate data representing the future or
        application time period.
    distribution : str, optional
        Kind of distribution used for parametric quantile mapping:
        ['normal', 'weibull', 'gamma', 'beta', 'rice'].
    trend_preservation : str, optional
        Kind of trend preservation used for non-parametric quantile mapping:
        ['additive', 'multiplicative', 'mixed', 'bounded'].
    n_quantiles : int, optional
        Number of quantile-quantile pairs used for non-parametric quantile
        mapping.
    p_value_eps : float, optional
        In order to keep p-values with numerically stable limits, they are
        capped at p_value_eps (lower bound) and 1 - p_value_eps (upper bound).
    max_change_factor : float, optional
        Maximum change factor applied in non-parametric quantile mapping with
        multiplicative or mixed trend preservation.
    max_adjustment_factor : float, optional
        Maximum adjustment factor applied in non-parametric quantile mapping
        with mixed trend preservation.
    adjust_p_values : boolean, optional
        Adjust p-values for a perfect match in the reference period.
    lower_bound : float, optional
        Lower bound of values in x_obs_hist, x_sim_hist, and x_sim_fut.
    lower_threshold : float, optional
        Lower threshold of values in x_obs_hist, x_sim_hist, and x_sim_fut.
        All values below this threshold are replaced by lower_bound in the end.
    upper_bound : float, optional
        Upper bound of values in x_obs_hist, x_sim_hist, and x_sim_fut.
    upper_threshold : float, optional
        Upper threshold of values in x_obs_hist, x_sim_hist, and x_sim_fut.
        All values above this threshold are replaced by upper_bound in the end.

    Returns
    -------
    x_sim_fut_ba : array
        Result of bias adjustment.

    """
    lower = lower_bound is not None and lower_threshold is not None
    upper = upper_bound is not None and upper_threshold is not None

    # determine extreme value probabilities of future obs
    if lower:
        p_lower_obs_hist = np.mean(x_obs_hist < lower_threshold)
        p_lower_sim_hist = np.mean(x_sim_hist < lower_threshold)
        p_lower_sim_fut = np.mean(x_sim_fut < lower_threshold)
        p_lower_target = uf.ccs_transfer_sim2obs(
            p_lower_obs_hist, p_lower_sim_hist, p_lower_sim_fut)
    if upper:
        p_upper_obs_hist = np.mean(x_obs_hist > upper_threshold)
        p_upper_sim_hist = np.mean(x_sim_hist > upper_threshold)
        p_upper_sim_fut = np.mean(x_sim_fut > upper_threshold)
        p_upper_target = uf.ccs_transfer_sim2obs(
            p_upper_obs_hist, p_upper_sim_hist, p_upper_sim_fut)
    if lower and upper:
        p_lower_or_upper_target = p_lower_target + p_upper_target
        if p_lower_or_upper_target > 1 + 1e-10:
            msg = 'sum of p_lower_target and p_upper_target exceeds one'
            warnings.warn(msg)
            p_lower_target /= p_lower_or_upper_target
            p_upper_target /= p_lower_or_upper_target

    # use augmented quantile delta mapping to transfer the simulated
    # climate change signal to the historical observation
    x_target = uf.map_quantiles_non_parametric_trend_preserving(
        x_obs_hist, x_sim_hist, x_sim_fut,
        trend_preservation, n_quantiles,
        max_change_factor, max_adjustment_factor,
        True, lower_bound, upper_bound)

    # do a parametric quantile mapping of the values within thresholds
    x_source = x_sim_fut
    y = x_source.copy()

    # determine indices of values to be mapped
    i_fit_obs_hist = np.ones(x_obs_hist.shape, dtype=bool)
    i_fit_sim_hist = np.ones(x_sim_hist.shape, dtype=bool)
    i_fit_source = np.ones(x_source.shape, dtype=bool)
    i_fit_target = np.ones(x_target.shape, dtype=bool)
    if lower:
        i_fit_obs_hist = np.logical_and(i_fit_obs_hist,
                                        x_obs_hist > lower_threshold)
        i_fit_sim_hist = np.logical_and(i_fit_sim_hist,
                                        x_sim_hist > lower_threshold)
        # make sure that lower_threshold_source < x_source 
        # because otherwise sps.beta.ppf does not work
        lower_threshold_source = \
            np.percentile(x_source, 100.*p_lower_target) \
            if p_lower_target > 0 else lower_bound if not upper else \
            lower_bound - 1e-10 * (upper_bound - lower_bound)
        i_lower = x_source <= lower_threshold_source
        i_fit_source = np.logical_and(i_fit_source, np.logical_not(i_lower))
        i_fit_target = np.logical_and(i_fit_target, x_target > lower_threshold)
        y[i_lower] = lower_bound
    if upper:
        i_fit_obs_hist = np.logical_and(i_fit_obs_hist,
                                        x_obs_hist < upper_threshold)
        i_fit_sim_hist = np.logical_and(i_fit_sim_hist,
                                        x_sim_hist < upper_threshold)
        # make sure that x_source < upper_threshold_source
        # because otherwise sps.beta.ppf does not work
        upper_threshold_source = \
            np.percentile(x_source, 100.*(1.-p_upper_target)) \
            if p_upper_target > 0 else upper_bound if not lower else \
            upper_bound + 1e-10 * (upper_bound - lower_bound)
        i_upper = x_source >= upper_threshold_source
        i_fit_source = np.logical_and(i_fit_source, np.logical_not(i_upper))
        i_fit_target = np.logical_and(i_fit_target, x_target < upper_threshold)
        y[i_upper] = upper_bound

    # map quantiles
    while np.any(i_fit_source):
        x_source_fit = x_source[i_fit_source]
        x_target_fit = x_target[i_fit_target]
        spsdotwhat = sps.norm if distribution == 'normal' else \
                     sps.weibull_min if distribution == 'weibull' else \
                     sps.gamma if distribution == 'gamma' else \
                     sps.beta if distribution == 'beta' else \
                     sps.rice if distribution == 'rice' else \
                     None

        # fix location and scale parameters for fitting
        floc = lower_threshold if lower else None
        floc_source = lower_threshold_source if lower else None
        fscale = upper_threshold - lower_threshold if lower and upper else None
        fscale_source = upper_threshold_source - lower_threshold_source \
                        if lower and upper else None

        # because sps.rice.fit and sps.weibull_min.fit cannot handle fscale=None
        if distribution in ['rice', 'weibull']:
            fwords = {'floc': floc}
            fwords_source = {'floc': floc_source}
        else:
            fwords = {'floc': floc, 'fscale': fscale}
            fwords_source = {'floc': floc_source, 'fscale': fscale_source}

        # fit distributions to x_source and x_target
        shape_loc_scale_source = uf.fit(spsdotwhat, x_source_fit, fwords_source)
        shape_loc_scale_target = uf.fit(spsdotwhat, x_target_fit, fwords)

        # do non-parametric or no quantile mapping if fitting failed
        if shape_loc_scale_source is None or shape_loc_scale_target is None:
            if x_target_fit.size:
                msg = 'unable to do parametric quantile mapping' \
                    + ': doing non-parametric quantile mapping instead'
                warnings.warn(msg)
                p_percent = np.linspace(0., 100., n_quantiles + 1)
                q_source_fit = np.percentile(x_source_fit, p_percent)
                q_target_fit = np.percentile(x_target_fit, p_percent)
                y[i_fit_source] = \
                    uf.map_quantiles_non_parametric_with_constant_extrapolation(
                    x_source_fit, q_source_fit, q_target_fit)
                break
            else:
                msg = 'unable to do any quantile mapping' \
                    + ': leaving %i value(s) unadjusted'%x_source_fit.size
                warnings.warn(msg)
                y[i_fit_source] = x_source_fit
                break

        # compute source p-values
        p_source = np.maximum(p_value_eps,
                   np.minimum(1-p_value_eps,
                   spsdotwhat.cdf(x_source_fit,
                   *shape_loc_scale_source)))

        # compute target p-values
        if adjust_p_values:
            x_obs_hist_fit = x_obs_hist[i_fit_obs_hist]
            x_sim_hist_fit = x_sim_hist[i_fit_sim_hist]
            shape_loc_scale_obs_hist = uf.fit(spsdotwhat,
                                       x_obs_hist_fit, fwords)
            shape_loc_scale_sim_hist = uf.fit(spsdotwhat,
                                       x_sim_hist_fit, fwords)
            if shape_loc_scale_obs_hist is None \
            or shape_loc_scale_sim_hist is None:
                msg = 'unable to adjust p-values: leaving them unadjusted'
                warnings.warn(msg)
                p_target = p_source
            else:
                p_obs_hist = np.maximum(p_value_eps,
                             np.minimum(1-p_value_eps,
                             spsdotwhat.cdf(x_obs_hist_fit,
                             *shape_loc_scale_obs_hist)))
                p_sim_hist = np.maximum(p_value_eps,
                             np.minimum(1-p_value_eps,
                             spsdotwhat.cdf(x_sim_hist_fit,
                             *shape_loc_scale_sim_hist)))
                p_target = np.maximum(p_value_eps,
                           np.minimum(1-p_value_eps,
                           uf.transfer_odds_ratio(
                           p_obs_hist, p_sim_hist, p_source)))
        else:
            p_target = p_source

        # map quantiles
        y[i_fit_source] = spsdotwhat.ppf(p_target, *shape_loc_scale_target)
        break

    return y


### ------------------- BIAS ADJUSTMENT UTILITIES ------------------- ###

# Function to detrend each timeseries of the three we are concerned with
def detrend_each_ts(x_obs_cal, x_sim_cal, x_sim_fut, cal_yrgrid, adj_yrgrid):
    """ 
    This function is a wrapper that enables detrending of xarray datasets
    used in the Lange bias adjustment calculations
    
    x_obs_cal, x_sim_cal, x_sim_fut, are single timeseries to be detrended
    cal_yrgrid, adj_yrgrid are the yearly grids the timeseries lie on

    Outputs are stored into dictionaries and returned to the user for the remainder
    of the bias adjustment process.
    """

    # calculate the trend and detrended timeseries for each dataset
    # noting that TREND represents just the stationary year-over-year linear change
    x_obs_cal_DETREND,x_obs_cal_TREND=subtract_or_add_trend(x_obs_cal,cal_yrgrid, trend=None) 
    x_sim_cal_DETREND,x_sim_cal_TREND=subtract_or_add_trend(x_sim_cal, cal_yrgrid, trend=None)
    x_sim_fut_DETREND,x_sim_fut_TREND=subtract_or_add_trend(x_sim_fut, adj_yrgrid, trend=None)

    # store detrend and trend timeseries
    detrend_ts={
        'x_obs_cal_detrend':x_obs_cal_DETREND,
        'x_sim_cal_detrend':x_sim_cal_DETREND,
        'x_sim_adj_detrend':x_sim_fut_DETREND,
    }
    trend_ts={
        'x_obs_cal_trend':x_obs_cal_TREND,
        'x_sim_cal_trend':x_sim_cal_TREND,
        'x_sim_adj_trend':x_sim_fut_TREND,
    }
    
    # calculate and store the trends from their slopes
    # and put into units of Celsius per decade
    trends={
        'obs_cal_CperDecade':(x_obs_cal_TREND[1]-x_obs_cal_TREND[0])*10,
        'sim_cal_CperDecade':(x_sim_cal_TREND[1]-x_sim_cal_TREND[0])*10,
        'sim_adj_CperDecade':(x_sim_fut_TREND[1]-x_sim_fut_TREND[0])*10,
    }
    
    # go back to the above program
    return(trends,detrend_ts,trend_ts)

# Function (by Lange) to add or subtract a trend from a single timeseries
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

def linlstsq(x, y):
    """
    Applies a least-square linear fit y = ax + b, and returns a and b.

    Parameters
    ----------
    x : array
        Array of x values.
    y : array
        Array of y values.

    Returns
    -------
    a : float
        Slope of regression line.
    b : float
        Offset of regression line.

    """
    A = np.vstack([x, np.ones(x.shape)]).T
    return np.linalg.lstsq(A, y, rcond=None)[0]