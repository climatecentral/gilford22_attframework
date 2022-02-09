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
Utility functions
=================

Provides auxiliary functions used by the modules bias_adjustment and
statistical_downscaling.

"""



import collections
import warnings
# import iris
import numpy as np
import datetime as dt
import scipy.stats as sps
import scipy.linalg as spl
import scipy.interpolate as spi



def assert_validity_of_months(months):
    """
    Raises an assertion error if any of the numbers in months is not in
    {1,...,12}.

    Parameters
    ----------
    months : array_like
        Sequence of ints representing calendar months.

    """
    msg = 'months have to be integers from {1,...,12}'
    months_allowed = np.arange(1, 13)
    for month in months:
        assert month in months_allowed, msg



def assert_coord_axis(cube, coord='time', axis=0):
    """
    Raises an assertion error if the axis of the specified coordinate in the
    specified iris cube differs from the specified one.

    Parameters
    ----------
    cube : iris cube
        Cube for which the assertion shall be done.
    coord : str, optional
        Coordinate identifier.
    axis : int, optional
        Dimension number.

    """
    msg = coord+' must be the coordinate of axis %i'%axis
    assert cube.coord_dims(coord) == (axis,), msg



def assert_consistency_of_bounds_and_thresholds(
        lower_bound=None, lower_threshold=None,
        upper_bound=None, upper_threshold=None):
    """
    Raises an assertion error if the pattern of specified and
    unspecified bounds and thresholds is not valid or if
    lower_bound < lower_threshold < upper_threshold < upper_bound
    does not hold.

    Parameters
    ----------
    lower_bound : float, optional
        Lower bound of values in time series.
    lower_threshold : float, optional
        Lower threshold of values in time series. All values below this
        threshold will be replaced by random numbers between lower_bound and
        lower_threshold before bias adjustment.
    upper_bound : float, optional
        Upper bound of values in time series.
    upper_threshold : float, optional
        Upper threshold of values in time series. All values above this
        threshold will be replaced by random numbers between upper_threshold and
        upper_bound before bias adjustment.

    """
    lower = lower_bound is not None and lower_threshold is not None
    upper = upper_bound is not None and upper_threshold is not None

    if not lower:
       msg = 'lower_bound is not None and lower_threshold is None'
       assert lower_bound is None, msg
       msg = 'lower_bound is None and lower_threshold is not None'
       assert lower_threshold is None, msg
    if not upper:
       msg = 'upper_bound is not None and upper_threshold is None'
       assert upper_bound is None, msg
       msg = 'upper_bound is None and upper_threshold is not None'
       assert upper_threshold is None, msg

    if lower:
        assert lower_bound < lower_threshold, 'lower_bound >= lower_threshold'
    if upper:
        assert upper_bound > upper_threshold, 'upper_bound <= upper_threshold'
    if lower and upper:
        msg = 'lower_threshold >= upper_threshold'
        assert lower_threshold < upper_threshold, msg



def assert_consistency_of_distribution_and_bounds(
        distribution,
        lower_bound=None, lower_threshold=None,
        upper_bound=None, upper_threshold=None):
    """
    Raises an assertion error if the the distribution is not consistent with the
    pattern of specified and unspecified bounds and thresholds.

    Parameters
    ----------
    distribution : str
        Kind of distribution used for parametric quantile mapping:
        ['normal', 'weibull', 'gamma', 'beta', 'rice'].
    lower_bound : float, optional
        Lower bound of values in time series.
    lower_threshold : float, optional
        Lower threshold of values in time series. All values below this
        threshold will be replaced by random numbers between lower_bound and
        lower_threshold before bias adjustment.
    upper_bound : float, optional
        Upper bound of values in time series.
    upper_threshold : float, optional
        Upper threshold of values in time series. All values above this
        threshold will be replaced by random numbers between upper_threshold and
        upper_bound before bias adjustment.

    """
    lower = lower_bound is not None and lower_threshold is not None
    upper = upper_bound is not None and upper_threshold is not None

    msg = distribution+' distribution '
    if distribution == 'normal':
        assert not lower and not upper, msg+'can not have bounds'
    elif distribution in ['weibull', 'gamma', 'rice']:
        assert lower and not upper, msg+'must only have lower bound'
    elif distribution == 'beta':
        assert lower and upper, msg+'must have lower and upper bound'
    else:
        raise AssertionError(msg+'not supported')



def assert_calendar(cube, calendar='proleptic_gregorian'):
    """
    Raises an assertion error if the calendar of the given iris cube differs
    from the one given.

    Parameters
    ----------
    cube : iris cube
        Cube for which the assertion shall be done.
    calendar : string, optional
        Calendar identifier.

    """
    msg = 'cube calendar != '+calendar
    assert cube.coord('time').units.calendar == calendar, msg



def assert_no_infs_or_nans(x_before, x_after):
    """
    Raises a value error if there are infs or nans in x_after. Prints the
    corresponding values in x_before.

    Parameters
    ----------
    x_before : ndarray
        Array before bias adjustement or statistical downscaling.
    x_after : ndarray
        Array after bias adjustement or statistical downscaling.

    """
    is_invalid = np.logical_or(np.isinf(x_after), np.isnan(x_after))
    if np.any(is_invalid):
        print(x_before[is_invalid])
        print(x_after[is_invalid], flush=True)
        msg = 'found infs or nans in x_after'
        raise ValueError(msg)



def add_basd_attributes(cube, options, prefix=''):
    """
    Adds keys and values of all command line options as attributes to iris cube.

    Parameters
    ----------
    cube : iris cube
        Cube to which attributes shall be added. Is changed in-place.
    options : optparse.Values
        Command line options parsed by optparse.OptionParser.
    prefix : str, optional
        Prefix attached to all attributes before they are put into the cube.

    """
    a = cube.attributes
    for key, value in options.__dict__.items():
        a[prefix+key] = str(value)
    cube.attributes = a



def extend0axis_periodic(a, n):
    """
    Extends an array periodically along axis 0.

    Parameters
    ----------
    a : ndarray
        Array to be extended.
    n : int
        Length of extension.

    Returns
    -------
    b : ndarray
        Result of extension. If a has shape (N,...) then b will have shape (N+2*n,...).

    """
    assert a.shape[0] >= n, 'length of a along axis 0 less than n'
    return np.concatenate((a[-n:], a, a[:n]), axis=0)



def aggregate0axis_periodic(a, halfwin, aggregator=np.mean):
    """
    Aggregates a along axis 0 using the given aggregator and a running window of
    length 2 * halfwin + 1, assuming that a is circular along axis 0.

    Parameters
    ----------
    a : ndarray
        Array to be aggregated.
    halfwin : int
        Determines length of running window used for aggregation.
    aggregator : function, optional
        Function used to aggregate a along axis 0 for every running window.

    Returns
    -------
    rm : ndarray
        Result of aggregation. Same shape as a.

    """
    window = 2 * halfwin + 1
    b = extend0axis_periodic(a, halfwin)
    rm = np.empty_like(a)
    for i in range(a.shape[0]):
        rm[i] = aggregator(b[i:i+window], axis=0)
    return rm



def get_upper_bound_climatology(c, halfwin):
    """
    Estimates an annual cycle of upper bounds for every grid cell of c as
    running mean values of running maximum values of multi-year daily maximum
    values.

    Parameters
    ----------
    c : iris cube
        Cube for which annual cycles of upper bounds shall be estimated.
    halfwin : int
        Determines length of running windows used for estimation.

    Returns
    -------
    c_multi_year_max : iris cube
        Cube containing annual cycles of upper bounds.

    """
    # compute multi-year daily maxima
    c_multi_year_max = c.aggregated_by('day_of_year', iris.analysis.MAX)

    # check length of time axis of resulting cube
    n = c_multi_year_max.shape[0]
    if n != 366:
        msg = 'upper bound climatology only defined for %i days of the year'%n \
            + ': this may imply an invalid computation of the climatology'
        warnings.warn(msg)

    # smooth climatology
    a0 = c_multi_year_max.data
    a1 = aggregate0axis_periodic(a0, halfwin, aggregator=np.max)
    a2 = aggregate0axis_periodic(a1, halfwin, aggregator=np.mean)
    c_multi_year_max.data = a2

    return c_multi_year_max



def ccs_transfer_sim2obs_upper_bound_climatology(obs_hist, sim_hist, sim_fut):
    """
    Multiplicatively transfers simulated climate change signal from sim_hist,
    sim_fut to obs_hist.

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

    Returns
    -------
    sim_fut_ba : iris cube
        Result of climate change signal transfer.

    """
    assert obs_hist.shape == sim_hist.shape == sim_fut.shape, \
           'obs_hist, sim_hist, sim_fut differ in shape'
    x_obs_hist = obs_hist.data
    x_sim_hist = sim_hist.data
    x_sim_fut = sim_fut.data
    with np.errstate(divide='ignore', invalid='ignore'):
        x_sim_fut_ba = np.where(x_sim_hist == 0, 0,
                       x_obs_hist * x_sim_fut / x_sim_hist)
    sim_fut_ba = sim_fut.copy()
    sim_fut_ba.data = x_sim_fut_ba
    return sim_fut_ba



def scale_by_upper_bound_climatology(
        cube, upper_bound_climatology, divide=True):
    """
    Scales all values in cube using the annual cycle of upper bounds.

    Parameters
    ----------
    cube : iris cube
        Cube to be scaled. Is changed in-place.
    upper_bound_climatology : iris cube
        Cube of annual cycle of upper bounds to be used for scaling.
    divide : boolean, optional
        If True the cube is divided by upper_bound_climatology, otherwise they
        are multiplied.

    """
    doys_cube = cube.coord('day_of_year').points
    doys = upper_bound_climatology.coord('day_of_year').points
    cube_data = cube.data
    for doy in doys:
        upper_bounds = upper_bound_climatology[doys == doy].data
        if divide:
            with np.errstate(divide='ignore', invalid='ignore'):
                scaling_factors = np.where(upper_bounds == 0, 1.,
                                  1. / upper_bounds)
        else:
            scaling_factors = upper_bounds
        cube_data[doys_cube == doy] = cube_data[doys_cube == doy] * \
                                      scaling_factors



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



def map_quantiles_non_parametric_trend_preserving(
        x_obs_hist, x_sim_hist, x_sim_fut, 
        trend_preservation='additive', n_quantiles=50,
        max_change_factor=100., max_adjustment_factor=9.,
        adjust_obs=False, lower_bound=None, upper_bound=None):
    """
    Adjusts biases with a modified version of the quantile delta mapping by
    Cannon (2015) <https://doi.org/10.1175/JCLI-D-14-00754.1> or uses this
    method to transfer a simulated climate change signal to observations.

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
    trend_preservation : str, optional
        Kind of trend preservation:
        'additive'       # Preserve additive trend.
        'multiplicative' # Preserve multiplicative trend, ensuring
                         # 1/max_change_factor <= change factor
                         #                     <= max_change_factor.
        'mixed'          # Preserve multiplicative or additive trend or mix of
                         # both depending on sign and magnitude of bias. Purely
                         # additive trends are preserved if adjustment factors
                         # of a multiplicative adjustment would be greater then
                         # max_adjustment_factor.
        'bounded'        # Preserve trend of bounded variable. Requires
                         # specification of lower_bound and upper_bound. It is
                         # ensured that the resulting values stay within these
                         # bounds.
    n_quantiles : int, optional
        Number of quantile-quantile pairs used for non-parametric quantile
        mapping.
    max_change_factor : float, optional
        Maximum change factor applied in non-parametric quantile mapping with
        multiplicative or mixed trend preservation.
    max_adjustment_factor : float, optional
        Maximum adjustment factor applied in non-parametric quantile mapping
        with mixed trend preservation.
    adjust_obs : boolean, optional
        If True then transfer simulated climate change signal to x_obs_hist,
        otherwise apply non-parametric quantile mapping to x_sim_fut.
    lower_bound : float, optional
        Lower bound of values in x_obs_hist, x_sim_hist, and x_sim_fut. Used
        for bounded trend preservation.
    upper_bound : float, optional
        Upper bound of values in x_obs_hist, x_sim_hist, and x_sim_fut. Used
        for bounded trend preservation.

    Returns
    -------
    y : array
        Result of quantile mapping or climate change signal transfer.

    """
    n = n_quantiles + 1
    p_zeroone = np.linspace(0., 1., n)
    p_percent = np.linspace(0., 100., n)

    # compute quantiles of input data
    q_obs_hist = np.percentile(x_obs_hist, p_percent)
    q_sim_hist = np.percentile(x_sim_hist, p_percent)
    q_sim_fut = np.percentile(x_sim_fut, p_percent)

    # compute quantiles needed for quantile delta mapping
    if adjust_obs: p = np.interp(x_obs_hist, q_obs_hist, p_zeroone)
    else: p = np.interp(x_sim_fut, q_sim_fut, p_zeroone)
    F_sim_fut_inv  = np.interp(p, p_zeroone, q_sim_fut)
    F_sim_hist_inv = np.interp(p, p_zeroone, q_sim_hist)
    F_obs_hist_inv = np.interp(p, p_zeroone, q_obs_hist)

    # do augmented quantile delta mapping
    if trend_preservation == 'bounded':
        msg = 'lower_bound or upper_bound not specified'
        assert lower_bound is not None and upper_bound is not None, msg
        assert lower_bound < upper_bound, 'lower_bound >= upper_bound'
        y = ccs_transfer_sim2obs(
            F_obs_hist_inv, F_sim_hist_inv, F_sim_fut_inv,
            lower_bound, upper_bound)
    elif trend_preservation in ['mixed', 'multiplicative']:
        assert max_change_factor > 1, 'max_change_factor <= 1'
        with np.errstate(divide='ignore', invalid='ignore'):
            y = np.where(F_sim_hist_inv == 0, 1., F_sim_fut_inv/F_sim_hist_inv)
            y[y > max_change_factor] = max_change_factor
            y[y < 1. / max_change_factor] = 1. / max_change_factor
        y *= F_obs_hist_inv
        if trend_preservation == 'mixed':  # if not then we are done here
            assert max_adjustment_factor > 1, 'max_adjustment_factor <= 1'
            y_additive = F_obs_hist_inv + F_sim_fut_inv - F_sim_hist_inv
            fraction_multiplicative = np.zeros_like(y)
            fraction_multiplicative[F_sim_hist_inv >= F_obs_hist_inv] = 1.
            i_transition = np.logical_and(F_sim_hist_inv < F_obs_hist_inv,
                F_obs_hist_inv < max_adjustment_factor * F_sim_hist_inv)
            fraction_multiplicative[i_transition] = .5 * (1. + 
                np.cos((F_obs_hist_inv[i_transition] /
                F_sim_hist_inv[i_transition] - 1.) * 
                np.pi / (max_adjustment_factor - 1.)))
            y = fraction_multiplicative * y + (1. -
                fraction_multiplicative) * y_additive
    elif trend_preservation == 'additive':
        y = F_obs_hist_inv + F_sim_fut_inv - F_sim_hist_inv
    else:
        msg = 'trend_preservation = '+trend_preservation+' not supported'
        raise AssertionError(msg)

    return y



def map_quantiles_non_parametric_with_constant_extrapolation(x, q_sim, q_obs):
    """
    Uses quantile-quantile pairs represented by values in q_sim and q_obs
    for quantile mapping of x.

    Values in x beyond the range of q_sim are mapped following the constant
    extrapolation approach, see Boe et al. (2007)
    <https://doi.org/10.1002/joc.1602>.

    Parameters
    ----------
    x : array
        Simulated time series.
    q_sim : array
        Simulated quantiles.
    q_obs : array
        Observed quantiles.
    
    Returns
    -------
    y : array
        Result of quantile mapping.

    """
    # make sure that q_sim and q_obs represent quantile-quantile pairs
    assert q_sim.size == q_obs.size, 'q_sim and q_obs are not of equal size'
    msg = ' is not monotonically increasing'
    assert np.all(np.diff(q_sim) > -1e-10), 'q_sim' + msg
    assert np.all(np.diff(q_obs) > -1e-10), 'q_obs' + msg

    # do the quantile mapping
    transfer_function = spi.interp1d(q_sim, q_obs)
    lunder = x < q_sim[0]
    lover = x > q_sim[-1]
    y = transfer_function(x)
    y[lunder] = x[lunder] + (q_obs[0] - q_sim[0])
    y[lover] = x[lover] + (q_obs[-1] - q_sim[-1])

    return y



def ccs_transfer_sim2obs(
        x_obs_hist, x_sim_hist, x_sim_fut,
        lower_bound=0., upper_bound=1.):
    """
    Generates pseudo future observation(s) by transfering a simulated climate
    change signal to historical observation(s) respecting the given bounds.

    Parameters
    ----------
    x_obs_hist : float or array
        Historical observation(s).
    x_sim_hist : float or array
        Historical simulation(s).
    x_sim_fut : float or array
        Future simulation(s).
    lower_bound : float, optional
        Lower bound of values in input and output data.
    upper_bound : float, optional
        Upper bound of values in input and output data.
    
    Returns
    -------
    x_obs_fut : float or array
        Pseudo future observation(s).

    """
    # change scalar inputs to arrays
    if np.isscalar(x_obs_hist): x_obs_hist = np.array([x_obs_hist])
    if np.isscalar(x_sim_hist): x_sim_hist = np.array([x_sim_hist])
    if np.isscalar(x_sim_fut): x_sim_fut = np.array([x_sim_fut])

    # check input
    assert lower_bound < upper_bound, 'lower_bound >= upper_bound'
    for x_name, x in zip(['x_obs_hist', 'x_sim_hist', 'x_sim_fut'],
                         [x_obs_hist, x_sim_hist, x_sim_fut]):
        assert np.all(x >= lower_bound), 'found '+x_name+' < lower_bound'
        assert np.all(x <= upper_bound), 'found '+x_name+' > upper_bound'

    # compute x_obs_fut
    i_decrease = x_sim_fut < x_sim_hist
    i_increase = x_sim_fut > x_sim_hist
    x_obs_fut = x_obs_hist.copy()  # for where x_sim_fut == x_sim_hist
    x_obs_fut[i_decrease] = lower_bound + \
                            (x_obs_hist[i_decrease] - lower_bound) * \
                            (x_sim_fut[i_decrease] - lower_bound) / \
                            (x_sim_hist[i_decrease] - lower_bound)
    x_obs_fut[i_increase] = upper_bound - \
                            (upper_bound - x_obs_hist[i_increase]) * \
                            (upper_bound - x_sim_fut[i_increase]) / \
                            (upper_bound - x_sim_hist[i_increase])

    # make sure x_obs_fut is within bounds
    x_obs_fut = np.maximum(lower_bound, np.minimum(upper_bound, x_obs_fut))

    return x_obs_fut[0] if x_obs_fut.size == 1 else x_obs_fut



def transfer_odds_ratio(p_obs_hist, p_sim_hist, p_sim_fut):
    """
    Transfers simulated changes in event likelihood to historical observations
    by multiplying the historical odds by simulated future-over-historical odds
    ratio. The method is inspired by the return interval scaling proposed by
    Switanek et al. (2017) <https://doi.org/10.5194/hess-21-2649-2017>.

    Parameters
    ----------
    p_obs_hist : array
        Culmulative probabbilities of historical observations.
    p_sim_hist : array
        Culmulative probabbilities of historical simulations.
    p_sim_fut : array
        Culmulative probabbilities of future simulations.

    Returns
    -------
    p_obs_fut : array
        Culmulative probabbilities of pseudo future observations.

    """
    x = np.sort(p_obs_hist)
    y = np.sort(p_sim_hist)
    z = np.sort(p_sim_fut)

    # interpolate x and y if necessary
    if x.size != z.size or y.size != z.size:
        p_x = np.linspace(0, 1, x.size)
        p_y = np.linspace(0, 1, y.size)
        p_z = np.linspace(0, 1, z.size)
        ppf_x = spi.interp1d(p_x, x)
        ppf_y = spi.interp1d(p_y, y)
        x = ppf_x(p_z)
        y = ppf_y(p_z)

    # transfer
    A = x * (1. - y) * z
    B = (1. - x) * y * (1. - z)
    z_scaled = 1. / (1. + B / A)

    # avoid the generation of unrealistically extreme p-values
    z_min = 1. / (1. + np.power(10.,  1. - np.log10(x / (1. - x))))
    z_max = 1. / (1. + np.power(10., -1. - np.log10(x / (1. - x))))
    z_scaled = np.maximum(z_min, np.minimum(z_max, z_scaled))

    return z_scaled[np.argsort(np.argsort(p_sim_fut))]



def randomize_censored_values(x,
        lower_bound=None, lower_threshold=None,
        upper_bound=None, upper_threshold=None,
        inplace=False, inverse=False,
        seed=None, lower_power=1., upper_power=1.):
    """
    Randomizes values beyond threshold in x or de-randomizes such formerly
    randomized values.

    Parameters
    ----------
    x : array
        Time series to be (de-)randomized.
    lower_bound : float, optional
        Lower bound of values in time series.
    lower_threshold : float, optional
        Lower threshold of values in time series.
    upper_bound : float, optional
        Upper bound of values in time series.
    upper_threshold : float, optional
        Upper threshold of values in time series.
    inplace : boolean, optional
        If True, change x in-place. If False, change a copy of x.
    inverse : boolean, optional
        If True, values beyond thresholds in x are set to the respective bound.
        If False, values beyond thresholds in x are randomized, i.e. values that
        exceed upper_threshold by are replaced by random numbers from the
        interval [lower_bound, lower_threshold), and values that fall short
        of lower_threshold are replaced by random numbers from the interval
        (upper_threshold, upper_bound].
    seed : int, optional
        Used to seed the random number generator before replacing values beyond
        threshold.
    lower_power : float, optional
        Numbers for randomizing values that fall short of lower_threshold are
        drawn from a uniform distribution and then taken to this power.
    upper_power : float, optional
        Numbers for randomizing values that exceed upper_threshold are drawn
        from a uniform distribution and then taken to this power.

    Returns
    -------
    x : array
        Randomized or de-randomized time series.

    """
    y = x if inplace else x.copy()
    if seed is not None:
        np.random.seed(seed)

    # randomize lower values
    if lower_bound is not None and lower_threshold is not None:
        if inverse:
            y[y < lower_threshold] = lower_bound
        else:
            i_lower = y <= lower_bound
            n_lower = np.sum(i_lower)
            if n_lower:
                p = np.power(np.random.uniform(0, 1, n_lower), lower_power)
                y[i_lower] = lower_bound + p * (lower_threshold - lower_bound)

    # randomize upper values
    if upper_bound is not None and upper_threshold is not None:
        if inverse:
            y[y > upper_threshold] = upper_bound
        else:
            i_upper = y >= upper_bound
            n_upper = np.sum(i_upper)
            if n_upper:
                p = np.power(np.random.uniform(0, 1, n_upper), upper_power)
                y[i_upper] = upper_bound - p * (upper_bound - upper_threshold)

    return y



def check_shape_loc_scale(spsdotwhat, shape_loc_scale):
    """
    Analyzes how distribution fitting has worked.

    Parameters
    ----------
    spsdotwhat : sps distribution class
        Known classes are [sps.norm, sps.weibull_min, sps.gamma, sps.rice,
        sps.beta].
    shape_loc_scale : tuple
        Fitted shape, location, and scale parameter values.

    Returns
    -------
    i : int
        0 if everything is fine,
        1 if there are infs or nans in shape_loc_scale,
        2 if at least one value in shape_loc_scale is out of bounds,
        3 if spsdotwhat is unknown.

    """
    if np.any(np.isnan(shape_loc_scale)) or np.any(np.isinf(shape_loc_scale)):
        return 1
    elif spsdotwhat == sps.norm:
        return 2 if shape_loc_scale[1] <= 0 else 0
    elif spsdotwhat in [sps.weibull_min, sps.gamma, sps.rice]:
        return 2 if shape_loc_scale[0] <= 0 or shape_loc_scale[2] <= 0 else 0
    elif spsdotwhat == sps.beta:
        return 2 if shape_loc_scale[0] <= 0 or shape_loc_scale[1] <= 0 \
            or shape_loc_scale[0] > 1e10 or shape_loc_scale[1] > 1e10 else 0
    else:
        return 3



def fit(spsdotwhat, x, fwords):
    """
    Attempts to fit a distribution from the family defined through spsdotwhat
    to the data represented by x, holding parameters fixed according to fwords.

    A maximum likelihood estimation of distribution parameter values is tried
    first. If that fails the method of moments is tried for some distributions.

    Parameters
    ----------
    spsdotwhat : sps distribution class
        Known classes are [sps.norm, sps.weibull_min, sps.gamma, sps.rice,
        sps.beta].
    x : array
        Data to be fitted.
    fwords : dict
        Dictionary with keys 'floc' and (optinally) 'fscale' specifying location
        and scale parmeter values, respectively, that are to be held fixed when
        fitting.

    Returns
    -------
    shape_loc_scale : tuple
        Fitted shape, location, and scale parameter values if fitting worked,
        otherwise None.

    """
    # make sure that there are at least two distinct data points because
    # otherwise it is impossible to fit more than 1 parameter
    if np.unique(x).size < 2:
        msg = 'found fewer then 2 different values in x: returning None'
        warnings.warn(msg)
        return None

    # try maximum likelihood estimation
    try:
        shape_loc_scale = spsdotwhat.fit(x, **fwords)
    except:
        shape_loc_scale = (np.nan,)

    # try method of moment estimation
    if check_shape_loc_scale(spsdotwhat, shape_loc_scale):
        msg = 'maximum likelihood estimation'
        if spsdotwhat == sps.gamma:
            msg += ' failed: method of moments'
            x_mean = np.mean(x) - fwords['floc']
            x_var = np.var(x)
            scale = x_var / x_mean
            shape = x_mean / scale
            shape_loc_scale = (shape, fwords['floc'], scale)
        elif spsdotwhat == sps.beta:
            msg += ' failed: method of moments'
            y = (x - fwords['floc']) / fwords['fscale']
            y_mean = np.mean(y)
            y_var = np.var(y)
            p = np.square(y_mean) * (1. - y_mean) / y_var - y_mean
            q = p * (1. - y_mean) / y_mean
            shape_loc_scale = (p, q, fwords['floc'], fwords['fscale'])
    else:
        msg = ''

    # return result and utter warning if necessary
    if check_shape_loc_scale(spsdotwhat, shape_loc_scale):
        msg += ' failed: returning None'
        warnings.warn(msg)
        return None
    else:
        if len(msg):
            msg += ' succeeded'
            warnings.warn(msg)
        return shape_loc_scale



def sample_invalid_values(a, seed=None, if_all_invalid_use=None, warn=False):
    """
    Replaces missing/inf/nan values in a by if_all_invalid_use or by sampling
    from all other values.

    Parameters
    ----------
    a : array or masked array
        If this is an array then infs and nans in a are replaced.
        If this is a masked array then infs, nans, and missing values in a.data
        are replaced using a.mask to indicate missing values.
    seed : int, optional
        Used to seed the random number generator before replacing invalid
        values.
    if_all_invalid_use : float, optional
        Used as replacement of invalid values if no valid values can be found.
    warn : boolean, optional
        Warn user about replacements being made.

    Returns
    -------
    d_replaced : array
        Result of invalid data replacement.
    l_invalid : array
        Boolean array indicating indices of replacement.

    """
    # assert that a is a 1d masked array
    if isinstance(a, np.ma.MaskedArray):
        d = a.data
        m = a.mask
    else:
        d = a
        m = np.zeros(a.shape, dtype=bool)
    
    # handle simple cases of no or only missing values
    if not isinstance(m, np.ndarray):
        if m:
            msg = 'found only missing value(s) in a'
            if if_all_invalid_use is None:
                raise ValueError(msg)
            else:
                msg += ': setting them all to %f'%if_all_invalid_use
                if warn: warnings.warn(msg)
                l_invalid = np.ones(m.shape, dtype=bool)
                d_replaced = np.empty_like(d)
                d_replaced[:] = if_all_invalid_use
                return d_replaced, l_invalid
        else:
            return d, None
    
    # look for missing values
    l_invalid = m
    n_missing = np.sum(l_invalid)
    if n_missing:
        msg = 'found %i missing value(s)'%n_missing
        if warn: warnings.warn(msg)
    
    # look for infs
    l_inf = np.isinf(d)
    n_inf = np.sum(l_inf)
    if n_inf:
        msg = 'found %i inf(s)'%n_inf
        if warn: warnings.warn(msg)
        l_invalid = np.logical_or(l_inf, l_invalid)
    
    # look for nans
    l_nan = np.isnan(d)
    n_nan = np.sum(l_nan)
    if n_nan:
        msg = 'found %i nan(s)'%n_nan
        if warn: warnings.warn(msg)
        l_invalid = np.logical_or(l_nan, l_invalid)
    
    # return d if all values are valid
    n_invalid = np.sum(l_invalid)
    if not n_invalid:
        return d, None
    
    # are there any valid values in a?
    if np.all(l_invalid):
        msg = 'found no valid value(s) in a'
        if if_all_invalid_use is None:
            raise ValueError(msg)
        else:
            msg += ': setting them all to %f'%if_all_invalid_use
            if warn: warnings.warn(msg)
            d_replaced = np.empty_like(d)
            d_replaced[:] = if_all_invalid_use
            return d_replaced, l_invalid

    # replace invalid values
    msg = 'replacing %i invalid value(s)'%n_invalid + \
    ' by sampling from %i valid value(s)'%(a.size - n_invalid)
    if warn: warnings.warn(msg)
    d_valid = d[np.logical_not(l_invalid)]
    if seed is not None: np.random.seed(seed)
    p_sampled = 100. * np.random.random_sample(n_invalid)
    d_sampled = np.percentile(d_valid, p_sampled)
    d_replaced = d.copy()
    d_replaced[l_invalid] = d_sampled

    return d_replaced, l_invalid



def time_range_to_iris_constraints(time_range):
    """
    Transforms a time range to an iris time constraint.

    Parameters
    ----------
    time_range : str
        Time range specified in format start+'-'+end, where start and end are
        each of format %Y%m%dT%H%M%S or parts thereof.

    Returns
    -------
    c : iris.Constraint
        Time constraint for constraining iris cubes.

    """
    if time_range is None:
      return None
    time_tuples = []
    for t in time_range.split('-'):
      try:
        tt = dt.datetime.strptime(t, '%Y').timetuple()[:1]
      except:
        try:
          tt = dt.datetime.strptime(t, '%Y%m').timetuple()[:2]
        except:
          try:
            tt = dt.datetime.strptime(t, '%Y%m%d').timetuple()[:3]
          except:
            try:
              tt = dt.datetime.strptime(t, '%Y%m%dT%H').timetuple()[:4]
            except:
              try:
                tt = dt.datetime.strptime(t, '%Y%m%dT%H%M').timetuple()[:5]
              except:
                try:
                  tt = dt.datetime.strptime(t, '%Y%m%dT%H%M%S').timetuple()[:6]
                except:
                  tt = None
      if tt is None:
        raise ValueError('unable to turn '+t+' into datetime object')
      time_tuples.append(tt)
    pdts = [iris.time.PartialDateTime(*tt) for tt in time_tuples]
    return iris.Constraint(time=lambda cell: pdts[0] <= cell <= pdts[1])



def get_anonymous_dimension_indices(c):
    """
    Returns a list of indices of anonymous dimensions of a given iris cube.

    Parameters
    ----------
    c : iris cube
        Cube whose anonymous dimensions are to be found.

    Returns
    -------
    l : list
        List of anonymous dimension indices.

    """
    all_dims = set(range(c.ndim))
    covered_dims = set(c.coord_dims(coord)[0] for coord in c.dim_coords)
    anonymous_dims = all_dims - covered_dims
    return list(anonymous_dims)



def name_first_anonymous_dimension(c, standard_name):
    """
    Adds a new dimension coordinate to an iris cube. The new coordinate has
    integer values from 0 to the length of the first anonymous dimension of 
    the iris cube minus 1.

    Parameters
    ----------
    c : iris cube
        Cube whose first anonymous dimension is to be named. Is changed
        in-place.
    standard_name : str
        Standard name of the new dimension coordinate.

    """
    if standard_name is not None:
        indices = get_anonymous_dimension_indices(c)
        if len(indices):
            i = indices[0]
            c.add_dim_coord(iris.coords.DimCoord(np.arange(c.shape[i]),
                standard_name=standard_name), i)



def generateCREmatrix(n):
    """
    Returns a random orthogonal n x n matrix from the circular real ensemble
    (CRE), see Mezzadri (2007) <http://arxiv.org/abs/math-ph/0609050v2>

    Parameters
    ----------
    n : int
        Number of rows and columns of the CRE matrix.

    Returns
    -------
    m : (n,n) ndarray
        CRE matrix.

    """
    z = np.random.randn(n, n)
    q, r = spl.qr(z)  # QR decomposition
    d = np.diagonal(r)
    return q * (d / np.abs(d))



def generate_rotation_matrix_fixed_first_axis(v, transpose=False):
    """
    Generates an n x n orthogonal matrix whose first row or column is equal to
     v/|v|, and whose other rows or columns are found by Gram-Schmidt
    orthogonalisation of v and the standard unit vectors except the first.

    Parameters
    ----------
    v : (n,) array
        Array of n non-zero numbers.
    transpose : boolean, optional
        If True/False generate an n x n orthogonal matrix whose first row/column
        is equal to v/|v|.

    Returns
    -------
    m : (n,n) ndarray
        Rotation matrix.

    """
    assert np.all(v > 0), 'all elements of v have to be positive'

    # generate matrix of vectors that span the R^n with v being the first vector
    a = np.diag(np.ones_like(v))
    a[:,0] = v

    # use QR decomposition for Gram-Schmidt orthogonalisation of these vectors
    q, r = spl.qr(a)

    return -q.T if transpose else -q



def get_downscaling_factor(shape_fine, shape_coarse):
    """
    Derives the downscaling factor from fine and coarse grid shapes.

    Parameters
    ----------
    shape_fine : tuple
        Shape of fine resolution grid.
    shape_coarse : tuple
        Shape of coarse resolution grid.

    Returns
    -------
    downscaling_factor : int
        The downscaling factor.

    """
    msg = 'number of spatial dimensions differs between fine and coarse grid'
    assert len(shape_fine) == len(shape_coarse), msg

    msg = 'downscaling factors are not all integers'
    assert np.all(np.array(shape_fine) % np.array(shape_coarse) == 0), msg

    downscaling_factors = set(np.array(shape_fine) // np.array(shape_coarse))
    msg = 'downscaling factors are not identical for all dimensions'
    assert len(downscaling_factors) == 1, msg

    return list(downscaling_factors)[0]



def flatten_all_dimensions_but_first(a):
    """
    Flattens all dimensions but the first of a multidimensional array.

    Parameters
    ----------
    a : ndarray
        Array to be flattened.

    Returns
    -------
    b : ndarray
        Result of flattening, two-dimensional.

    """
    s = a.shape
    s_flattened = (s[0], np.prod(s[1:]))
    return a.reshape(*s_flattened)



class StepSliceIterator(collections.Iterator):
    """
    Creates a step slice iterator to iterate over slices of an iris cubes,
    with slices of adjustable step size. This class is very similar to the
    iris.cube._SliceIterator class.

    """
    def __init__(self, cube, requested_dims, step):
        self._cube = cube
        dims_index = list(np.array(cube.shape) // step)
        for d in requested_dims:
            dims_index[d] = 1
        self._ndindex = np.ndindex(*dims_index)
        self._requested_dims = requested_dims
        self._step = step

    def __next__(self):
        index_tuple = next(self._ndindex)
        index_list = list(index_tuple)
        for d, i in enumerate(index_list):
            if d in self._requested_dims:
                index_list[d] = slice(None, None)
            else:
                index_list[d] = slice(self._step * i, self._step * (i + 1))
        cube = self._cube[tuple(index_list)]
        return cube

    next = __next__
