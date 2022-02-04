# (C) 2019 Potsdam Institute for Climate Impact Research (PIK)
# 
# This file is part of ISIMIP3BASD.
#
# ISIMIP3BASD is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ISIMIP3BASD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ISIMIP3BASD. If not, see <http://www.gnu.org/licenses/>.



"""
Bias adjustment
===============

Provides functions for bias adjustment of climate simulation data using climate
observation data with the same spatial and temporal resolution.

The following variable-specific parameter values (variable units in brackets)
were used to produce the results presented in Stefan Lange: Trend-preserving
bias adjustment and statistical downscaling with ISIMIP3BASD (v1.0),
Geoscientific Model Development Discussions, 2019.

hurs (%)
    --halfwin-upper-bound-climatology 0
    --lower-bound 0
    --lower-threshold .01
    --upper-bound 100
    --upper-threshold 99.99
    --distribution beta
    --trend-preservation bounded
    --adjust-p-values

pr (mm day-1)
    --halfwin-upper-bound-climatology 0
    --lower-bound 0
    --lower-threshold .1
    --distribution gamma
    --trend-preservation mixed
    --adjust-p-values

prsnratio (1)
    --halfwin-upper-bound-climatology 0
    --lower-bound 0
    --lower-threshold .0001
    --upper-bound 1
    --upper-threshold .9999
    --distribution beta
    --trend-preservation bounded
    --if-all-invalid-use 0.
    --adjust-p-values

psl (Pa)
    --halfwin-upper-bound-climatology 0
    --distribution normal
    --trend-preservation additive
    --adjust-p-values
    --detrend

rlds (W m-2)
    --halfwin-upper-bound-climatology 0
    --distribution normal
    --trend-preservation additive
    --adjust-p-values
    --detrend

rsds (W m-2)
    --halfwin-upper-bound-climatology 15
    --lower-bound 0
    --lower-threshold .0001
    --upper-bound 1
    --upper-threshold .9999
    --distribution beta
    --trend-preservation bounded
    --adjust-p-values

sfcWind (m s-1)
    --halfwin-upper-bound-climatology 0
    --lower-bound 0
    --lower-threshold .01
    --distribution weibull
    --trend-preservation mixed
    --adjust-p-values

tas (K)
    --halfwin-upper-bound-climatology 0
    --distribution normal
    --trend-preservation additive
    --detrend

tasrange (K)
    --halfwin-upper-bound-climatology 0
    --lower-bound 0
    --lower-threshold .01
    --distribution rice
    --trend-preservation mixed
    --adjust-p-values

tasskew (1)
    --halfwin-upper-bound-climatology 0
    --lower-bound 0
    --lower-threshold .0001
    --upper-bound 1
    --upper-threshold .9999
    --distribution beta
    --trend-preservation bounded
    --adjust-p-values

"""



import dask
import iris
import warnings
import numpy as np
import scipy.stats as sps
import utility_functions as uf
import iris.coord_categorisation as icc
import multiprocessing as mp
from optparse import OptionParser
from functools import partial



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
    method described in Stefan Lange: Trend-preserving bias adjustment and
    statistical downscaling with ISIMIP3BASD (v1.0), Geoscientific Model
    Development Discussions, 2019.

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



def adjust_bias_one_time_series(
        x_obs_hist, x_sim_hist, x_sim_fut,
        years_obs_hist, years_sim_hist, years_sim_fut,
        lower_bound=None, lower_threshold=None,
        upper_bound=None, upper_threshold=None,
        randomization_seed=None, detrend=False,
        **kwargs):
    """
    First, detrends time series if desired. Secondly, replaces values beyond
    thresholds by random numbers. Thirdly, adjusts biases. Fourthly, replaces
    values beyond thresholds by the respective bound. Fithly, restores trends.

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
    years_obs_hist : array
        Year of every time step of the time series x_obs_hist used for
        detrending.
    years_sim_hist : array
        Year of every time step of the time series x_sim_hist used for
        detrending.
    years_sim_fut : array
        Year of every time step of the time series x_sim_fut used for
        detrending.
    lower_bound : float, optional
        Lower bound of values in x_obs_hist, x_sim_hist, and x_sim_fut.
    lower_threshold : float, optional
        Lower threshold of values in x_obs_hist, x_sim_hist, and x_sim_fut.
        All values below this threshold are replaced by random numbers between
        lower_bound and lower_threshold before bias adjustment.
    upper_bound : float, optional
        Upper bound of values in x_obs_hist, x_sim_hist, and x_sim_fut.
    upper_threshold : float, optional
        Upper threshold of values in x_obs_hist, x_sim_hist, and x_sim_fut.
        All values above this threshold are replaced by random numbers between
        upper_threshold and upper_bound before bias adjustment.
    randomization_seed : int, optional
        Used to seed the random number generator before replacing values beyond
        the specified thresholds.
    detrend : boolean, optional
        Detrend time series before bias adjustment and put trend back in
        afterwards.

    Returns
    -------
    x_sim_fut_ba : array
        Result of bias adjustment.

    Other Parameters
    ----------------
    **kwargs : Passed on to map_quantiles_parametric_trend_preserving.
    
    """
    # subtract trend
    if detrend:
        x_obs_hist, trend_obs_hist = uf.subtract_or_add_trend(
            x_obs_hist, years_obs_hist)
        x_sim_hist, trend_sim_hist = uf.subtract_or_add_trend(
            x_sim_hist, years_sim_hist)
        x_sim_fut, trend_sim_fut = uf.subtract_or_add_trend(
            x_sim_fut, years_sim_fut)
    else:
        x_obs_hist = x_obs_hist.copy()
        x_sim_hist = x_sim_hist.copy()
        x_sim_fut = x_sim_fut.copy()

    # randomize censored values
    # use high powers to create many values close to the bounds as this
    # alleviates kinks in the empirical CDFs at the thresholds if formerly
    # censored values need to be uncensored
    uf.randomize_censored_values(x_obs_hist, 
        lower_bound, lower_threshold, upper_bound, upper_threshold, True, False,
        randomization_seed, 10., 10.)
    uf.randomize_censored_values(x_sim_hist,
        lower_bound, lower_threshold, upper_bound, upper_threshold, True, False,
        randomization_seed, 10., 10.)
    uf.randomize_censored_values(x_sim_fut,
        lower_bound, lower_threshold, upper_bound, upper_threshold, True, False,
        randomization_seed, 10., 10.)

    # adjust distribution and de-randomize censored values
    x_sim_fut_ba = map_quantiles_parametric_trend_preserving(
        x_obs_hist, x_sim_hist, x_sim_fut,
        lower_bound=lower_bound, lower_threshold=lower_threshold,
        upper_bound=upper_bound, upper_threshold=upper_threshold,
        **kwargs)

    # add trend
    if detrend:
        x_sim_fut_ba = uf.subtract_or_add_trend(
            x_sim_fut_ba, years_sim_fut, trend_sim_fut)

    # make sure there are no invalid values
    uf.assert_no_infs_or_nans(x_sim_fut, x_sim_fut_ba)

    return x_sim_fut_ba



def adjust_bias_one_month(
        obs_hist, sim_hist, sim_fut,
        restore_invalid_values=False,
        randomization_seed=None,
        if_all_invalid_use=None,
        invalid_value_warnings=False,
        **kwargs):
    """
    Extracts data arrays from iris cubes, replaces invalid values in these, 
    passes resulting arrays to adjust_bias_one_time_series, restores invalid
    values in resulting array if desired, returns result as an iris cube.

    Parameters
    ----------
    obs_hist : iris cube
        Cube of observed climate data representing the historical or training
        time period.
    sim_hist : iris cube
        Cube of simulated climate data representing the historical or training
        time period.
    sim_fut : iris cube
        Cube of simulated climate data representing the future or application
        time period.
    restore_invalid_values : boolean, optional
        Restore invalid values in input data after bias adjustment.
    randomization_seed : int, optional
        Used to seed the random number generator before replacing invalid
        values.
    if_all_invalid_use : float, optional
        Used to replace invalid values if there are no valid values. An error
        is raised if there are no valid values and this parameter is None.
    invalid_value_warnings : boolean, optional
        Raise user warnings when invalid values are replaced bafore bias
        adjustment.

    Returns
    -------
    sim_fut_ba : iris cube
        Result of bias adjustment.

    Other Parameters
    ----------------
    **kwargs : Passed on to adjust_bias_one_time_series.

    """
    x_sim_fut_masked = sim_fut.data

    # load data from iris cubes and remove invalid values
    x_obs_hist, m_obs_hist = uf.sample_invalid_values(obs_hist.data,
        randomization_seed, if_all_invalid_use, invalid_value_warnings)
    x_sim_hist, m_sim_hist = uf.sample_invalid_values(sim_hist.data,
        randomization_seed, if_all_invalid_use, invalid_value_warnings)
    x_sim_fut, m_sim_fut = uf.sample_invalid_values(x_sim_fut_masked,
        randomization_seed, if_all_invalid_use, invalid_value_warnings)

    # adjust bias and restore formerly invalid values
    x_sim_fut_ba = adjust_bias_one_time_series(
        x_obs_hist, x_sim_hist, x_sim_fut,
        randomization_seed=randomization_seed, **kwargs)
    if restore_invalid_values:
        if m_sim_fut is not None:
            x_sim_fut_ba[m_sim_fut] = x_sim_fut_masked.data[m_sim_fut]
        m_sim_fut_ba = x_sim_fut_masked.mask
    else:
        m_sim_fut_ba = np.zeros_like(x_sim_fut_masked.mask)

    # create iris cube
    sim_fut_ba = sim_fut.copy()
    sim_fut_ba.data = np.ma.array(x_sim_fut_ba, mask=m_sim_fut_ba,
        fill_value=x_sim_fut_masked.fill_value)

    return sim_fut_ba



def adjust_bias_one_location(
        cubes_tuple, months=[], halfwin_upper_bound_climatology=0, **kwargs):
    """
    Adjusts biases in climate data representing one grid cell calendar month by
    calendar month.

    Parameters
    ----------
    cubes_tuple : (3,) tuple
        Tuple of iris cubes obs_hist, sim_hist, and sim_fut containing climate
        data.
    months : list, optional
        List of ints from {1,...,12} representing calendar months for which 
        results of statistical downscaling are to be returned.
    halfwin_upper_bound_climatology : int, optional
        Determines the length of running windows used in the calculations of
        climatologies of upper bounds that are used to scale values of obs_hist,
        sim_hist, and sim_fut to the interval [0,1] before bias adjustment. The
        window length is set to halfwin_upper_bound_climatology * 2 + 1 time
        steps. If halfwin_upper_bound_climatology == 0 then no rescaling is
        done.

    Returns
    -------
    sim_fut_ba.data : array
        Result of bias adjustment.

    Other Parameters
    ----------------
    **kwargs : Passed on to adjust_bias_one_month.

    """
    # prevent dask from opening new threads every time lazy data are realized
    # as this results in RuntimeError: can't start new thread
    # see <http://docs.dask.org/en/latest/scheduler-overview.html>
    dask.config.set(scheduler='single-threaded')

    # put local iris cubes into dictionary
    cubes = {
    'obs_hist': cubes_tuple[0],
    'sim_hist': cubes_tuple[1],
    'sim_fut': cubes_tuple[2]
    }

    # load iris cube data into memory
    for key, cube in cubes.items():
        d = cube.data

    # scale to values in [0, 1]
    if halfwin_upper_bound_climatology:
        upper_bound_climatologies = {}

        # scale obs_hist, sim_hist, sim_fut
        for key, cube in cubes.items():
            upper_bound_climatologies[key] = uf.get_upper_bound_climatology(
                cube, halfwin_upper_bound_climatology)
            uf.scale_by_upper_bound_climatology(
                cube, upper_bound_climatologies[key], divide=True)

        # prepare scaling of sim_fut_ba
        upper_bound_climatologies['sim_fut_ba'] = \
            uf.ccs_transfer_sim2obs_upper_bound_climatology(
            upper_bound_climatologies['obs_hist'],
            upper_bound_climatologies['sim_hist'],
            upper_bound_climatologies['sim_fut'])

    # do bias adjustment calendar month by calendar month
    cubes_this_month = {}
    years_this_month = {}
    cubes_adjusted = []
    for month in months:
        for key, cube in cubes.items():
            cubes_this_month[key] = \
                cube.extract(iris.Constraint(month_number=month))
            years_this_month[key] = cubes_this_month[key].coord('year').points
        sim_fut_ba_this_month = adjust_bias_one_month(
            cubes_this_month['obs_hist'],
            cubes_this_month['sim_hist'],
            cubes_this_month['sim_fut'],
            years_obs_hist=years_this_month['obs_hist'],
            years_sim_hist=years_this_month['sim_hist'],
            years_sim_fut=years_this_month['sim_fut'],
            **kwargs)

        # only store results for actual days of this month
        # we cannot just do cubes_adjusted.append(
        # sim_fut_ba_this_month.extract(iris.Constraint(
        # month_number=month))) here because then the concatenate_cube
        # operation below would not work because iris does not yet
        # support tiled concatenation
        years = list(np.unique(years_this_month['sim_fut']))
        cubes_adjusted.extend([sim_fut_ba_this_month.extract(
                               iris.Constraint(year=year))
                               for year in years])
    
    # merge results across calendar months
    sim_fut_ba = iris.cube.CubeList(cubes_adjusted).concatenate_cube()

    # scale from values in [0, 1]
    if halfwin_upper_bound_climatology:
        uf.scale_by_upper_bound_climatology(
            sim_fut_ba, upper_bound_climatologies['sim_fut_ba'], divide=False)
    
    # save memory by only returning cube data and not the whole cube (the cube
    # also contains the time axis, which in a typical application is of double
    # precision while the data are in single precision, which means that the
    # time array occupies twice as much memory as the data array)
    return sim_fut_ba.data



def adjust_bias(
        obs_hist, sim_hist, sim_fut,
        realize_cubes=False, anonymous_dimension_name=None,
        halfwin_upper_bound_climatology=0, n_processes=1,
        **kwargs):
    """
    Adjusts biases grid cell by grid cell.

    Parameters
    ----------
    obs_hist : iris cube
        Cube of observed climate data representing the historical or training
        time period.
    sim_hist : iris cube
        Cube of simulated climate data representing the historical or training
        time period.
    sim_fut : iris cube
        Cube of simulated climate data representing the future or application
        time period.
    realize_cubes : boolean, optional
        Realize data of obs_hist, sim_hist, and sim_fut before beginning the
        bias adjustment grid cell by grid cell.
    anonymous_dimension_name : str, optional
        Used to name the first anonymous dimension of obs_hist, sim_hist, and
        sim_fut.
    halfwin_upper_bound_climatology : int, optional
        Determines the length of running windows used in the calculations of
        climatologies of upper bounds that is used to rescale all values of
        obs_hist, sim_hist, and sim_fut to values <= 1 before bias adjustment.
        The window length is set to halfwin_upper_bound_climatology * 2 + 1
        time steps. If halfwin_upper_bound_climatology == 0 then no rescaling
        is done.
    n_processes : int, optional
        Number of processes used for parallel processing.

    Returns
    -------
    sim_fut_ba : iris cube
        Result of bias adjustment.

    Other Parameters
    ----------------
    **kwargs : Passed on to adjust_bias_one_location.

    """
    # put iris cubes into dictionary
    cubes = {
    'obs_hist': obs_hist,
    'sim_hist': sim_hist,
    'sim_fut': sim_fut
    }

    space_shape = None
    for key, cube in cubes.items():
        # get cube shape beyond time axis
        if space_shape is None: space_shape = cube.shape[1:]
        else: assert space_shape == cube.shape[1:], 'cube shapes not compatible'
        # load iris cube data into memory
        if realize_cubes: d = cube.data
        # make sure the proleptic gregorian calendar is used in all input files
        uf.assert_calendar(cube, 'proleptic_gregorian')
        # make sure that time is the leading coordinate
        uf.assert_coord_axis(cube, 'time', 0)
        # name the first anonymous dimension
        uf.name_first_anonymous_dimension(cube, anonymous_dimension_name)
        # prepare scaling by upper bound climatology
        if halfwin_upper_bound_climatology: icc.add_day_of_year(cube, 'time')
        # prepare bias adjustment calendar month by calendar month
        icc.add_month_number(cube, 'time')
        # prepare detrending and cube concatenation
        icc.add_year(cube, 'time')
    
    # adjust every location individually using multiprocessing
    print('adjusting at location ...')
    abol = partial(adjust_bias_one_location,
        halfwin_upper_bound_climatology=halfwin_upper_bound_climatology,
        **kwargs)
    pool = mp.Pool(n_processes, maxtasksperchild=1000)
    time_series_adjusted = pool.imap(abol, zip(
        obs_hist.slices('time'),
        sim_hist.slices('time'),
        sim_fut.slices('time')))
    pool.close()

    # replace time series in sim_fut by the adjusted time series
    sim_fut_ba = sim_fut
    d = sim_fut_ba.data
    for i_location, tsa in zip(np.ndindex(space_shape), time_series_adjusted):
        d[(slice(None, None),) + i_location] = tsa
        print(i_location)

    # remove auxiliary coordinates
    sim_fut_ba.remove_coord('year')
    sim_fut_ba.remove_coord('month_number')
    if halfwin_upper_bound_climatology: sim_fut_ba.remove_coord('day_of_year')

    return sim_fut_ba



def main():
    """
    Prepares and concludes bias adjustment.

    """
    # parse command line options and arguments
    parser = OptionParser()
    parser.add_option('-o', '--obs-hist', action='store',
        type='string', dest='obs_hist', default=None,
        help='path to input netcdf file with historical observation')
    parser.add_option('-s', '--sim-hist', action='store',
        type='string', dest='sim_hist', default=None,
        help='path to input netcdf file with historical simulation')
    parser.add_option('-f', '--sim-fut', action='store',
        type='string', dest='sim_fut', default=None,
        help='path to input netcdf file with future simulation')
    parser.add_option('-b', '--sim-fut-ba', action='store',
        type='string', dest='sim_fut_ba', default=None,
        help='path to output netcdf file with bias-adjusted future simulation')
    parser.add_option('-v', '--variable', action='store',
        type='string', dest='variable', default=None,
        help=('standard name of variable to be adjusted in netcdf files '
              '(has to be the same in all files)'))
    parser.add_option('-m', '--months', action='store',
        type='string', dest='months', default=None,
        help=('comma-separated list of integers from {1,...,12} representing '
              'calendar months that shall be bias-adjusted'))
    parser.add_option('--n-processes', action='store',
        type='int', dest='n_processes', default=1,
        help='number of processes used for multiprocessing (default: 1)')
    parser.add_option('-w', '--halfwin-upper-bound-climatology', action='store',
        type='int', dest='halfwin_upper_bound_climatology', default=0,
        help=('half window length used to compute climatologies of upper '
              'bounds used to scale values before and after bias adjustment '
              '(default: 0, which is interpreted as do not scale)'))
    parser.add_option('-a', '--anonymous-dimension-name', action='store',
        type='string', dest='anonymous_dimension_name', default=None,
        help=('if loading into iris cubes results in the creation of one or '
              'multiple anonymous dimensions, then the first of those will be '
              'given this name if specified'))
    parser.add_option('--o-time-range', action='store',
        type='string', dest='obs_hist_time_range', default=None,
        help=('time constraint for data extraction from input netcdf file with '
              'historical observation of format %Y%m%dT%H%M%S-%Y%m%dT%H%M%S '
              '(if not specified then no time constraint is applied)'))
    parser.add_option('--s-time-range', action='store',
        type='string', dest='sim_hist_time_range', default=None,
        help=('time constraint for data extraction from input netcdf file with '
              'historical simulation of format %Y%m%dT%H%M%S-%Y%m%dT%H%M%S '
              '(if not specified then no time constraint is applied)'))
    parser.add_option('--f-time-range', action='store',
        type='string', dest='sim_fut_time_range', default=None,
        help=('time constraint for data extraction from input netcdf file with '
              'future simulation of format %Y%m%dT%H%M%S-%Y%m%dT%H%M%S '
              '(if not specified then no time constraint is applied)'))
    parser.add_option('--b-time-range', action='store',
        type='string', dest='sim_fut_ba_time_range', default=None,
        help=('time constraint for data extraction from iris cube with bias-'
              'adjusted future simulation of format %Y%m%dT%H%M%S-%Y%m%dT%H%M%S'
              ' (if not specified then no time constraint is applied)'))
    parser.add_option('--lower-bound', action='store',
        type='float', dest='lower_bound', default=None,
        help=('lower bound of variable that has to be respected during bias '
              'adjustment (default: not specified)'))
    parser.add_option('--lower-threshold', action='store',
        type='float', dest='lower_threshold', default=None,
        help=('lower threshold of variable that has to be respected during '
              'bias adjustment (default: not specified)'))
    parser.add_option('--upper-bound', action='store',
        type='float', dest='upper_bound', default=None,
        help=('upper bound of variable that has to be respected during '
              'bias adjustment (default: not specified)'))
    parser.add_option('--upper-threshold', action='store',
        type='float', dest='upper_threshold', default=None,
        help=('upper threshold of variable that has to be respected during '
              'bias adjustment (default: not specified)'))
    parser.add_option('--randomization-seed', action='store',
        type='int', dest='randomization_seed', default=None,
        help=('seed used during randomization to generate reproducible results '
              '(default: not specified)'))
    parser.add_option('--distribution', action='store',
        type='string', dest='distribution', default='normal',
        help=('distribution family used for parametric quantile mapping '
              '(default: normal, alternatives: gamma, weibull, beta, rice)'))
    parser.add_option('-t', '--trend-preservation', action='store',
        type='string', dest='trend_preservation', default='additive',
        help=('kind of trend preservation (default: additive, alternatives: '
              'multiplicative, mixed, bounded)'))
    parser.add_option('-q', '--n-quantiles', action='store',
        type='int', dest='n_quantiles', default=50,
        help=('number of quantiles used for non-parametric quantile mapping '
              '(default: 50)'))
    parser.add_option('-e', '--p-value-eps', action='store',
        type='float', dest='p_value_eps', default=1.e-10,
        help=('lower cap for p-values during parametric quantile mapping '
              '(default: 1.e-10)'))
    parser.add_option('--max-change-factor', action='store',
        type='float', dest='max_change_factor', default=100.,
        help=('cap for change factor for non-parametric quantile mapping '
              '(default: 100.)'))
    parser.add_option('--max-adjustment-factor', action='store',
        type='float', dest='max_adjustment_factor', default=9.,
        help=('cap for adjustment factor for non-parametric quantile mapping '
              '(default: 9.)'))
    parser.add_option('--if-all-invalid-use', action='store',
        type='float', dest='if_all_invalid_use', default=None,
        help=('replace missing values, infs and nans by this value before '
              'biases adjustment if there are no other values in a time series '
              '(default: None)'))
    parser.add_option('-p', '--adjust-p-values', action='store_true',
        dest='adjust_p_values', default=False,
        help=('adjust p-values during parametric quantile mapping for a '
              'perfect adjustment of the reference period distribution '
              '(default: do not)'))
    parser.add_option('-d', '--detrend', action='store_true',
        dest='detrend', default=False,
        help=('subtract trend before bias adjustment, add it back afterwards '
              '(default: do not)'))
    parser.add_option('--realize-cubes', action='store_true',
        dest='realize_cubes', default=False,
        help=('realize iris cube data right after loading '
              '(this can reduce run time, default: do not)'))
    parser.add_option('--repeat-warnings', action='store_true',
        dest='repeat_warnings', default=False,
        help='repeat warnings for the same source location (default: do not)')
    parser.add_option('--invalid-value-warnings', action='store_true',
        dest='invalid_value_warnings', default=False,
        help=('raise warning when missing values, infs or nans are replaced by '
              'sampling from all other values before bias adjustment '
              '(default: do not)'))
    parser.add_option('--restore-invalid-values', action='store_true',
        dest='restore_invalid_values', default=False,
        help=('restore missing values, infs and nans after bias adjustment '
              '(note that missing values, infs and nans are always replaced by '
              'sampling from all other values before bias adjustment; '
              'default: do not)'))
    parser.add_option('--limit-time-dimension', action='store_true',
        dest='limit_time_dimension', default=False,
        help=('save output netcdf file with a limited time dimension (default: '
              'save output netcdf file with an unlimited time dimension)'))
    (options, args) = parser.parse_args()
    if options.repeat_warnings: warnings.simplefilter('always', UserWarning)

    # set multiprocessing's way to start a process to forkserver
    # because the default (fork) leads to memory errors
    # see <https://github.com/SALib/SALib/issues/140>
    mp.set_start_method('forkserver')

    # do some preliminary checks
    months = list(np.sort(np.unique(np.array(
        options.months.split(','), dtype=int))))
    uf.assert_validity_of_months(months)
    uf.assert_consistency_of_bounds_and_thresholds(
        options.lower_bound, options.lower_threshold,
        options.upper_bound, options.upper_threshold)
    uf.assert_consistency_of_distribution_and_bounds(options.distribution,
        options.lower_bound, options.lower_threshold,
        options.upper_bound, options.upper_threshold)

    # process time constraints
    obs_hist_time_constraint = uf.time_range_to_iris_constraints(
                               options.obs_hist_time_range)
    sim_hist_time_constraint = uf.time_range_to_iris_constraints(
                               options.sim_hist_time_range)
    sim_fut_time_constraint = uf.time_range_to_iris_constraints(
                              options.sim_fut_time_range)
    sim_fut_ba_time_constraint = uf.time_range_to_iris_constraints(
                                 options.sim_fut_ba_time_range)

    # load input data
    obs_hist = iris.load_cube(options.obs_hist, options.variable
               if obs_hist_time_constraint is None else
               options.variable & obs_hist_time_constraint)
    sim_hist = iris.load_cube(options.sim_hist, options.variable
               if sim_hist_time_constraint is None else
               options.variable & sim_hist_time_constraint)
    sim_fut = iris.load_cube(options.sim_fut, options.variable
              if sim_fut_time_constraint is None else
              options.variable & sim_fut_time_constraint)

    # do bias adjustment
    sim_fut_ba = adjust_bias(
                 obs_hist, sim_hist, sim_fut,
                 options.realize_cubes,
                 options.anonymous_dimension_name,
                 options.halfwin_upper_bound_climatology,
                 options.n_processes,
                 months=months,
                 lower_bound=options.lower_bound,
                 lower_threshold=options.lower_threshold,
                 upper_bound=options.upper_bound,
                 upper_threshold=options.upper_threshold,
                 randomization_seed=options.randomization_seed,
                 distribution=options.distribution,
                 trend_preservation=options.trend_preservation,
                 n_quantiles=options.n_quantiles,
                 p_value_eps=options.p_value_eps,
                 max_change_factor=options.max_change_factor,
                 max_adjustment_factor=options.max_adjustment_factor,
                 if_all_invalid_use=options.if_all_invalid_use,
                 adjust_p_values=options.adjust_p_values,
                 detrend=options.detrend,
                 restore_invalid_values=options.restore_invalid_values,
                 invalid_value_warnings=options.invalid_value_warnings)

    # write bias adjustment parameters into attributes of sim_fut_ba
    uf.add_basd_attributes(sim_fut_ba, options, 'ba_')

    # save output data
    iris.save(sim_fut_ba if sim_fut_ba_time_constraint is None else 
              sim_fut_ba.extract(sim_fut_ba_time_constraint),
              options.sim_fut_ba, 
              saver=iris.fileformats.netcdf.save,
              unlimited_dimensions=None
              if options.limit_time_dimension else ['time'],
              fill_value=1.e20, zlib=True, complevel=1)



if __name__ == '__main__':
    main()
