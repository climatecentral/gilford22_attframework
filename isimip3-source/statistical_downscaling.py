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
Statistical downscaling
=======================

Provides functions for statistical downscaling of climate simulation data using
climate observation data with the same temporal and higher spatial resolution.

The following variable-specific parameter values (variable units in brackets)
were used to produce the results presented in Stefan Lange: Trend-preserving
bias adjustment and statistical downscaling with ISIMIP3BASD (v1.0),
Geoscientific Model Development Discussions, 2019.

hurs (%)
    --lower-bound 0
    --lower-threshold .01
    --upper-bound 100
    --upper-threshold 99.99

pr (mm day-1)
    --lower-bound 0
    --lower-threshold .1

prsnratio (1)
    --lower-bound 0
    --lower-threshold .0001
    --upper-bound 1
    --upper-threshold .9999
    --if-all-invalid-use 0.

psl (Pa)

rlds (W m-2)

rsds (W m-2)
    --lower-bound 0
    --lower-threshold .01

sfcWind (m s-1)
    --lower-bound 0
    --lower-threshold .01

tas (K)

tasrange (K)
    --lower-bound 0
    --lower-threshold .01

tasskew (1)
    --lower-bound 0
    --lower-threshold .0001
    --upper-bound 1
    --upper-threshold .9999

"""



import dask
import iris
import warnings
import numpy as np
import utility_functions as uf
import iris.coord_categorisation as icc
import multiprocessing as mp
from optparse import OptionParser
from functools import partial



def weighted_sum_preserving_mbcn(
        x_obs, x_sim_coarse, x_sim,
        sum_weights, rotation_matrices=[], n_quantiles=50):
    """
    Applies the core of the modified MBCn algorithm for statistical downscaling
    as described in Lange (2019) <https://doi.org/>.

    Parameters
    ----------
    x_obs : (M,N) ndarray
        Array of N observed time series of M time steps each at fine spatial
        resolution.
    x_sim_coarse : (M,) array
        Array of simulated time series of M time steps at coarse spatial
        resolution.
    x_sim : (M,N) ndarray
        Array of N simulated time series of M time steps each at fine spatial
        resolution, derived from x_sim_coarse by bilinear interpolation.
    sum_weights : (N,) array
        Array of N grid cell-area weights.
    rotation_matrices : list of (N,N) ndarrays, optional
        List of orthogonal matrices defining a sequence of rotations in the  
        second dimension of x_obs and x_sim.
    n_quantiles : int, optional
        Number of quantile-quantile pairs used for non-parametric quantile
        mapping.

    Returns
    -------
    x_sim : (M,N) ndarray
        Result of application of the modified MBCn algorithm.

    """
    # initialize total rotation matrix
    n_variables = sum_weights.size
    o_total = np.diag(np.ones(n_variables))

    # p-values in percent for non-parametric quantile mapping
    p = np.linspace(0., 100., n_quantiles+1)

    # normalise the sum weights vector to length 1
    sum_weights = sum_weights / np.sqrt(np.sum(np.square(sum_weights)))

    # rescale x_sim_coarse for initial step of algorithm
    x_sim_coarse = x_sim_coarse * np.sum(sum_weights)

    # iterate
    n_loops = len(rotation_matrices) + 2
    for i in range(n_loops):
        if not i:  # rotate to the sum axis
            o = uf.generate_rotation_matrix_fixed_first_axis(sum_weights)
        elif i == n_loops - 1:  # rotate back to original axes for last qm
            o = o_total.T
        else:  # do random rotation
            o = rotation_matrices[i-1]

        # compute total rotation
        o_total = np.dot(o_total, o)

        # rotate data
        x_sim = np.dot(x_sim, o)
        x_obs = np.dot(x_obs, o)
        sum_weights = np.dot(sum_weights, o)

        if not i:
            # restore simulated values at coarse grid scale
            x_sim[:,0] = x_sim_coarse

            # quantile map observations to values at coarse grid scale
            q_sim = np.percentile(x_sim_coarse, p)
            q_obs = np.percentile(x_obs[:,0], p)
            x_obs[:,0] = \
                uf.map_quantiles_non_parametric_with_constant_extrapolation(
                x_obs[:,0], q_obs, q_sim)
        else:
            # do univariate non-parametric quantile mapping for every variable
            x_sim_previous = x_sim.copy()
            q_sim = np.percentile(x_sim, p, axis=0)
            q_obs = np.percentile(x_obs, p, axis=0)
            for j in range(n_variables):
                x_sim[:,j] = \
                    uf.map_quantiles_non_parametric_with_constant_extrapolation(
                    x_sim[:,j], q_sim[:,j], q_obs[:,j])

            # preserve weighted sum of original variables
            if i < n_loops - 1:
                x_sim -= np.outer(np.dot(x_sim - x_sim_previous,
                                         sum_weights), sum_weights)

    return x_sim



def downscale_one_time_series(
        x_obs_fine, x_sim_coarse, x_sim_coarse_remapbil,
        lower_bound=None, lower_threshold=None,
        upper_bound=None, upper_threshold=None,
        randomization_seed=None,
        **kwargs):
    """
    First, replaces values beyond thresholds by random numbers.
    Secondly, applies the modified MBCn algorithm for statistical downscaling.
    Thirdly, replaces values beyond thresholds by the respective bound.

    Parameters
    ----------
    x_obs_fine : (M,N) ndarray
        Array of N observed time series of M time steps each at fine spatial
        resolution.
    x_sim_coarse : (M,) array
        Array of simulated time series of M time steps at coarse spatial
        resolution.
    x_sim_coarse_remapbil : (M,N) ndarray
        Array of N simulated time series of M time steps each at fine spatial
        resolution, derived from x_sim_coarse by bilinear interpolation.
    lower_bound : float, optional
        Lower bound of values in x_obs_fine, x_sim_coarse, and
        x_sim_coarse_remapbil.
    lower_threshold : float, optional
        Lower threshold of values in x_obs_fine, x_sim_coarse, and
        x_sim_coarse_remapbil. All values below this threshold are replaced by
        random numbers between lower_bound and lower_threshold before
        application of the modified MBCn algorithm.
    upper_bound : float, optional
        Upper bound of values in x_obs_fine, x_sim_coarse, and
        x_sim_coarse_remapbil.
    upper_threshold : float, optional
        Upper threshold of values in x_obs_fine, x_sim_coarse, and
        x_sim_coarse_remapbil. All values above this threshold are replaced by
        random numbers between upper_threshold and upper_bound before
        application of the modified MBCn algorithm.
    randomization_seed : int, optional
        Used to seed the random number generator before replacing values beyond
        the specified thresholds.

    Returns
    -------
    x_sim_fine : (M,N) ndarray
        Result of application of the modified MBCn algorithm.

    Other Parameters
    ----------------
    **kwargs : Passed on to weighted_sum_preserving_mbcn.
    
    """
    # randomize censored values
    # use high powers to create many values close to the bounds as this
    # keeps weighted sums similar to original values
    x_obs_fine = uf.randomize_censored_values(x_obs_fine, 
        lower_bound, lower_threshold, upper_bound, upper_threshold,
        False, False, randomization_seed, 10., 10.)
    x_sim_coarse = uf.randomize_censored_values(x_sim_coarse,
        lower_bound, lower_threshold, upper_bound, upper_threshold,
        False, False, randomization_seed, 10., 10.)
    x_sim_coarse_remapbil = uf.randomize_censored_values(x_sim_coarse_remapbil,
        lower_bound, lower_threshold, upper_bound, upper_threshold,
        False, False, randomization_seed, 10., 10.)

    # downscale
    x_sim_fine = weighted_sum_preserving_mbcn(
        uf.flatten_all_dimensions_but_first(x_obs_fine),
        x_sim_coarse,
        uf.flatten_all_dimensions_but_first(x_sim_coarse_remapbil),
        **kwargs).reshape(*x_sim_coarse_remapbil.shape)

    # de-randomize censored values
    uf.randomize_censored_values(x_sim_fine, 
        lower_bound, lower_threshold, upper_bound, upper_threshold, True, True)

    # make sure there are no invalid values
    uf.assert_no_infs_or_nans(x_sim_coarse_remapbil, x_sim_fine)

    return x_sim_fine



def downscale_one_month(
        obs_fine, sim_coarse, sim_coarse_remapbil,
        restore_invalid_values=False,
        randomization_seed=None,
        if_all_invalid_use=None,
        **kwargs):
    """
    Extracts data arrays from iris cubes, replaces invalid values in these, 
    passes resulting arrays to downscale_one_time_series, restores invalid
    values in resulting array if desired, returns result as an iris cube.

    Parameters
    ----------
    obs_fine : (M,N) iris cube
        Cube of N observed time series of M time steps each at fine spatial
        resolution.
    sim_coarse : (M,) iris cube
        Cube of simulated time series of M time steps at coarse spatial
        resolution.
    sim_coarse_remapbil : (M,N) iris cube
        Cube of N simulated time series of M time steps each at fine spatial
        resolution, derived from x_sim_coarse by bilinear interpolation.
    restore_invalid_values : boolean, optional
        Restore invalid values in input data after statistical downscaling.
    randomization_seed : int, optional
        Used to seed the random number generator before replacing invalid
        values.
    if_all_invalid_use : float, optional
        Used to replace invalid values if there are no valid values. An error
        is raised if there are no valid values and this parameter is None.

    Returns
    -------
    sim_fine : (M,N) iris cube
        Result of application of the modified MBCn algorithm.

    Other Parameters
    ----------------
    **kwargs : Passed on to downscale_one_time_series.

    """
    x_sim_coarse_remapbil_masked = sim_coarse_remapbil.data

    # load data from iris cubes and remove invalid values
    x_obs_fine, m_obs_fine = uf.sample_invalid_values(
        obs_fine.data, randomization_seed, if_all_invalid_use)
    x_sim_coarse, m_sim_coarse = uf.sample_invalid_values(
        sim_coarse.data, randomization_seed, if_all_invalid_use)
    x_sim_coarse_remapbil, m_sim_coarse_remapbil = uf.sample_invalid_values(
        x_sim_coarse_remapbil_masked, randomization_seed, if_all_invalid_use)

    # downscale and restore formerly invalid values
    x_sim_fine = downscale_one_time_series(
        x_obs_fine, x_sim_coarse, x_sim_coarse_remapbil,
        randomization_seed=randomization_seed, **kwargs)
    if restore_invalid_values:
        if m_sim_coarse_remapbil is not None:
            x_sim_fine[m_sim_coarse_remapbil] = \
                x_sim_coarse_remapbil_masked.data[m_sim_coarse_remapbil]
        m_sim_fine = x_sim_coarse_remapbil_masked.mask
    else:
        m_sim_fine = np.zeros_like(x_sim_coarse_remapbil_masked.mask)

    # create iris cube
    sim_fine = sim_coarse_remapbil.copy()
    sim_fine.data = np.ma.array(x_sim_fine, mask=m_sim_fine,
        fill_value=x_sim_coarse_remapbil_masked.fill_value)

    return sim_fine



def downscale_one_location(
        cubes_tuple, months=[], years=[], **kwargs):
    """
    Applies the modified MBCn algorithm for statistical downscaling calendar
    month by calendar month to climate data within one coarse grid cell.

    Parameters
    ----------
    cubes_tuple : (3,) tuple
        Tuple of iris cubes obs_fine, sim_coarse, and sim_coarse_remapbil 
        containing climate data.
    months : list, optional
        List of ints from {1,...,12} representing calendar months for which 
        results of statistical downscaling are to be returned.
    years : list, optional
        List of years for which results of statistical downscaling are to be
        returned.

    Returns
    -------
    sim_fine.data : (M,N) ndarray
        Result of application of the modified MBCn algorithm.

    Other Parameters
    ----------------
    **kwargs : Passed on to downscale_one_month.

    """
    # prevent dask from opening new threads every time lazy data are realized
    # as this results in RuntimeError: can't start new thread
    # see <http://docs.dask.org/en/latest/scheduler-overview.html>
    dask.config.set(scheduler='single-threaded')

    # put local iris cubes into dictionary
    cubes = {
    'obs_fine': cubes_tuple[0],
    'sim_coarse': cubes_tuple[1],
    'sim_coarse_remapbil': cubes_tuple[2]
    }

    # load iris cube data into memory
    for key, cube in cubes.items():
        d = cube.data

    # get grid cell area weights
    sum_weights = iris.analysis.cartography.cosine_latitude_weights(
                  cubes['sim_coarse_remapbil'][0]).flatten()

    # do statistical downscaling calendar month by calendar month
    cubes_this_month = {}
    cubes_downscaled = []
    for month in months:
        for key, cube in cubes.items():
            cubes_this_month[key] = \
                cube.extract(iris.Constraint(month_number=month))
        sim_fine_this_month = downscale_one_month(
            cubes_this_month['obs_fine'],
            cubes_this_month['sim_coarse'],
            cubes_this_month['sim_coarse_remapbil'],
            sum_weights = sum_weights,
            **kwargs)
        cubes_downscaled.extend([sim_fine_this_month.extract(
                                 iris.Constraint(year=year))
                                 for year in years])
    
    # merge results across calendar months
    sim_fine = iris.cube.CubeList(cubes_downscaled).concatenate_cube()

    return sim_fine.data



def downscale(
        obs_fine, sim_coarse,
        realize_cubes=False, anonymous_dimension_name=None,
        n_processes=1, n_iterations=20, randomization_seed=None,
        **kwargs):
    """
    Applies the modified MBCn algorithm for statistical downscaling calendar
    month by calendar month and coarse grid cell by coarse grid cell.

    Parameters
    ----------
    obs_fine : iris cube
        Cube of observed climate data at fine spatial resolution.
    sim_coarse : iris cube
        Cube of simulated climate data coarse spatial resolution.
    realize_cubes : boolean, optional
        Realize data of obs_fine and sim_coarse before beginning the statistical
        downscaling coarse grid cell by coarse grid cell.
    anonymous_dimension_name : str, optional
        Used to name the first anonymous dimension of obs_fine and sim_coarse.
    n_processes : int, optional
        Number of processes used for parallel processing.
    n_iterations : int, optional
        Number of iterations used in the modified MBCn algorithm.
    randomization_seed : int, optional
        Used to seed the random number generator before generating random 
        rotation matrices for the modified MBCn algorithm.

    Returns
    -------
    sim_fine : iris cube
        Result of application of the modified MBCn algorithm.

    Other Parameters
    ----------------
    **kwargs : Passed on to downscale_one_location.

    """
    # put iris cubes into dictionary
    cubes = {
    'obs_fine': obs_fine,
    'sim_coarse': sim_coarse,
    }

    space_shapes = {}
    for key, cube in cubes.items():
        # get cube shape beyond time axis
        space_shapes[key] = cube.shape[1:]
        # load iris cube data into memory
        if realize_cubes: d = cube.data
        # make sure the proleptic gregorian calendar is used in all input files
        uf.assert_calendar(cube, 'proleptic_gregorian')
        # make sure that time is the leading coordinate
        uf.assert_coord_axis(cube, 'time', 0)
        # name the first anonymous dimension
        uf.name_first_anonymous_dimension(cube, anonymous_dimension_name)
        # prepare statistical downscaling calendar month by calendar month
        icc.add_month_number(cube, 'time')

    # derive downscaling factor from cube shapes beyond time axis
    downscaling_factor = uf.get_downscaling_factor(
        space_shapes['obs_fine'], space_shapes['sim_coarse'])

    # get list of rotation matrices to be used for all locations and months
    n_fine_per_coarse = downscaling_factor ** len(space_shapes['obs_fine'])
    if randomization_seed is not None: np.random.seed(randomization_seed)
    rotation_matrices = [uf.generateCREmatrix(n_fine_per_coarse)
                         for i in range(n_iterations)]

    # bilinearly interpolate sim_coarse to grid of obs_fine
    print('interpolating to fine grid ...')
    sim_coarse_remapbil = sim_coarse.regrid(obs_fine, iris.analysis.Linear())
    icc.add_year(sim_coarse_remapbil, 'time')
    years = list(np.unique(sim_coarse_remapbil.coord('year').points))
    
    # downscale every location individually using multiprocessing
    print('downscaling at coarse location ...')
    sdol = partial(downscale_one_location, 
        years=years, rotation_matrices=rotation_matrices,
        randomization_seed=randomization_seed, **kwargs)
    pool = mp.Pool(n_processes, maxtasksperchild=1000)
    time_series_downscaled = pool.imap(sdol, zip(
        uf.StepSliceIterator(obs_fine, [0], downscaling_factor),
        sim_coarse.slices('time'),
        uf.StepSliceIterator(sim_coarse_remapbil, [0], downscaling_factor),))
    pool.close()

    # replace time series in sim_coarse_remapbil by the downscaled time series
    sim_fine = sim_coarse_remapbil
    d = sim_fine.data
    i_locations_coarse = np.ndindex(space_shapes['sim_coarse'])
    for ilc, tsd in zip(i_locations_coarse, time_series_downscaled):
        ilf = tuple([slice(downscaling_factor * i,
                           downscaling_factor * (i + 1)) for i in ilc])
        d[(slice(None, None),) + ilf] = tsd
        print(ilc)

    # remove auxiliary coordinates
    sim_fine.remove_coord('year')
    sim_fine.remove_coord('month_number')

    return sim_fine



def main():
    """
    Prepares and concludes the application of the modified MBCn algorithm for
    statistical downscaling.

    """
    # parse command line options and arguments
    parser = OptionParser()
    parser.add_option('-o', '--obs-fine', action='store',
        type='string', dest='obs_fine', default=None,
        help='path to input netcdf file with observation at fine resolution')
    parser.add_option('-s', '--sim-coarse', action='store',
        type='string', dest='sim_coarse', default=None,
        help='path to input netcdf file with simulation at coarse resolution')
    parser.add_option('-f', '--sim-fine', action='store',
        type='string', dest='sim_fine', default=None,
        help=('path to output netcdf file with simulation statistically '
              'downscaled to fine resolution'))
    parser.add_option('-v', '--variable', action='store',
        type='string', dest='variable', default=None,
        help=('standard name of variable to be downscaled in netcdf files '
              '(has to be the same in all files)'))
    parser.add_option('-m', '--months', action='store',
        type='string', dest='months', default=None,
        help=('comma-separated list of integers from {1,...,12} representing '
              'calendar months that shall be statistically downscaled'))
    parser.add_option('--n-processes', action='store',
        type='int', dest='n_processes', default=1,
        help='number of processes used for multiprocessing (default: 1)')
    parser.add_option('--n-iterations', action='store',
        type='int', dest='n_iterations', default=20,
        help=('number of iterations used for statistical downscaling '
              '(default: 20)'))
    parser.add_option('-a', '--anonymous-dimension-name', action='store',
        type='string', dest='anonymous_dimension_name', default=None,
        help=('if loading into iris cubes results in the creation of one or '
              'multiple anonymous dimensions, then the first of those will be '
              'given this name if specified'))
    parser.add_option('--o-time-range', action='store',
        type='string', dest='obs_fine_time_range', default=None,
        help=('time constraint for data extraction from input netcdf file with '
              'observation of format %Y%m%dT%H%M%S-%Y%m%dT%H%M%S '
              '(if not specified then no time constraint is applied)'))
    parser.add_option('--s-time-range', action='store',
        type='string', dest='sim_coarse_time_range', default=None,
        help=('time constraint for data extraction from input netcdf file with '
              'simulation of format %Y%m%dT%H%M%S-%Y%m%dT%H%M%S '
              '(if not specified then no time constraint is applied)'))
    parser.add_option('--f-time-range', action='store',
        type='string', dest='sim_fine_time_range', default=None,
        help=('time constraint for data extraction from output netcdf file '
              'with simulation of format %Y%m%dT%H%M%S-%Y%m%dT%H%M%S '
              '(if not specified then no time constraint is applied)'))
    parser.add_option('--lower-bound', action='store',
        type='float', dest='lower_bound', default=None,
        help=('lower bound of variable that has to be respected during '
              'statistical downscaling (default: not specified)'))
    parser.add_option('--lower-threshold', action='store',
        type='float', dest='lower_threshold', default=None,
        help=('lower threshold of variable that has to be respected during '
              'statistical downscaling (default: not specified)'))
    parser.add_option('--upper-bound', action='store',
        type='float', dest='upper_bound', default=None,
        help=('upper bound of variable that has to be respected during '
              'statistical downscaling (default: not specified)'))
    parser.add_option('--upper-threshold', action='store',
        type='float', dest='upper_threshold', default=None,
        help=('upper threshold of variable that has to be respected during '
              'statistical downscaling (default: not specified)'))
    parser.add_option('--randomization-seed', action='store',
        type='int', dest='randomization_seed', default=None,
        help=('seed used during randomization to generate reproducible results '
              '(default: not specified)'))
    parser.add_option('-q', '--n-quantiles', action='store',
        type='int', dest='n_quantiles', default=50,
        help=('number of quantiles used for non-parametric quantile mapping '
              '(default: 50)'))
    parser.add_option('--if-all-invalid-use', action='store',
        type='float', dest='if_all_invalid_use', default=None,
        help=('replace missing values, infs and nans by this value before '
              'biases adjustment if there are no other values in a time series '
              '(default: None)'))
    parser.add_option('--realize-cubes', action='store_true',
        dest='realize_cubes', default=False,
        help=('realize iris cube data right after loading '
              '(this can reduce run time, default: do not)'))
    parser.add_option('--repeat-warnings', action='store_true',
        dest='repeat_warnings', default=False,
        help='repeat warnings for the same source location (default: do not)')
    parser.add_option('--restore-invalid-values', action='store_true',
        dest='restore_invalid_values', default=False,
        help=('restore missing values, infs and nans after statistical '
              'downscaling (note that missing values, infs and nans are always '
              'replaced by sampling from all other values before statistical '
              'downscaling; default: do not)'))
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

    # process time constraints
    obs_fine_time_constraint = uf.time_range_to_iris_constraints(
                               options.obs_fine_time_range)
    sim_coarse_time_constraint = uf.time_range_to_iris_constraints(
                                 options.sim_coarse_time_range)
    sim_fine_time_constraint = uf.time_range_to_iris_constraints(
                               options.sim_fine_time_range)

    # load input data
    obs_fine = iris.load_cube(options.obs_fine, options.variable
               if obs_fine_time_constraint is None else
               options.variable & obs_fine_time_constraint)
    sim_coarse = iris.load_cube(options.sim_coarse, options.variable
                 if sim_coarse_time_constraint is None else
                 options.variable & sim_coarse_time_constraint)

    # do statistical downscaling
    sim_fine = downscale(
               obs_fine, sim_coarse,
               options.realize_cubes,
               options.anonymous_dimension_name,
               options.n_processes,
               options.n_iterations,
               options.randomization_seed,
               months=months,
               lower_bound=options.lower_bound,
               lower_threshold=options.lower_threshold,
               upper_bound=options.upper_bound,
               upper_threshold=options.upper_threshold,
               n_quantiles=options.n_quantiles,
               if_all_invalid_use=options.if_all_invalid_use,
               restore_invalid_values=options.restore_invalid_values)

    # write statistical downscaling parameters into attributes of sim_fine
    uf.add_basd_attributes(sim_fine, options, 'sd_')

    # save output data
    iris.save(sim_fine if sim_fine_time_constraint is None else 
              sim_fine.extract(sim_fine_time_constraint),
              options.sim_fine, 
              saver=iris.fileformats.netcdf.save,
              unlimited_dimensions=None
              if options.limit_time_dimension else ['time'], 
              fill_value=1.e20, zlib=True, complevel=1)



if __name__ == '__main__':
    main()
