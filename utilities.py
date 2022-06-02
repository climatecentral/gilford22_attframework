"""
This module code contains utility functions that we use throughout the attribution framework 
codebase, by first appending the directory that contains this set of utilities, and then
calling the particular function we want to use.

Author: [Daniel Gilford](https://github.com/dgilford)
Last updated: 5/31/2022 by Daniel Gilford
"""

# import modules to use in functions
import numpy as np
import xarray as xr
from datetime import datetime
import os
import random

### ------------------- DATETIME UTILITIES ------------------- ###

# Function to organize the data's time grids with datetime formats
def get_dt64(ds):
    # ds is an input xarray data structure with dimension/variable "time"
    # grab and reformat the time index from the netcdf files
    inttime=[str(int(i)) for i in ds.time.values]
    # convert to dt64
    dt64=[np.datetime64(datetime.strptime(i, '%Y%m%d')) for i in inttime]
    dt64=np.asarray(dt64)
    return(dt64)

# Function to take a slice between two years with dt64 type
def dt64_yrslice(lower_yr=1985,upper_yr=2015):
    slice_range=[str(int(lower_yr))+'-01-01',str(int(upper_yr))+'-12-31']
    sliceout=slice(slice_range[0],slice_range[1])
    return(sliceout)

### ------------------- VISUALIZATION UTILITIES ------------------- ###

# define a function to wrap around the prime meridian 
# (via https://github.com/pydata/xarray/issues/1005 solution)
def xr_add_cyclic_point(ds, dim='lon', period=None):
  if period is None:
    period = ds.sizes[dim] * ds.coords[dim][:2].diff(dim).item()
  first_point = ds.isel({dim: slice(1)})
  first_point.coords[dim] = first_point.coords[dim]+period
  return xr.concat([ds, first_point], dim=dim)

# Function to fix xarray data structures with N96 grid so that SOUTH POLE points are set to missing
def set_N96_SouthPole_missing(dsN96,METRIC):
    # ds is an input xarray data structure with dimensions/variables "lat" and "lon"
    # METRIC is the variable (a string) that we are fixing
    
    # subselect the locations that aren't the southern pole
    midworld=dsN96[METRIC].sel(lon=slice(0,360),lat=slice(-89.5,90.5)).load()
    # subselect the southern pole and fill it with missing values (np.nan)
    sh=dsN96[METRIC].sel(lon=slice(0,360),lat=slice(-90.5,-89.5)).load()
    sh.values[:]=np.nan
    # merge the datasets and return to the above program level
    together=xr.merge([midworld,sh])
    return(together)

### ------------------- DATA MANAGEMENT ------------------- ###

# Function using 1d interpolation to fill missing values
def fill_nan(A):
    '''
    interpolate to fill nan values
    '''
    from scipy import interpolate
    
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good],bounds_error=False)
    B = np.where(np.isfinite(A),A,f(inds))
    return B

# Function to load local zarr files/directories into an xarray dataset
def load_zarr_local(loadpath,zarrname,chunks='auto'):
    # define the path where the *.zarr data directory is located 
    path=loadpath+zarrname
    # check to make sure if the the directory exists
    if(not(os.path.exists(path))):
            raise Exception('local file {} does not exist'.format(zarrname))
    # if the directory exists, load it with xarray, and return the data structure
    zout=xr.open_zarr(path, chunks=chunks)
    return(zout)

# Function to save xarray datasets as zarr files/directories locally
def save_zarr_local(ds,savepath,zarrname):
    import zarr
    # define the path where the *.zarr data directory should be saved
    path=savepath+zarrname
    # define the compression and encoding, then save to a zarr file directory
    compressor = zarr.Blosc(cname='zstd', clevel=3)
    encoding = {vname: {'compressor': compressor} for vname in ds.data_vars}
    ds.to_zarr(path, encoding=encoding, consolidated=True, mode='w')

# Function to find the middle element of a list
def middle_element(lst):
    middlei=(len(lst) - 1)//2
    return(lst[middlei])

### -------------------STATISTICAL UTILITIES ------------------- ###

# Function to randomly and uniformly resample a histogram from its counts
def randomly_sample_histogram(hist,N=10_000,seed=616,bins=None):
    
    # create a cdf from the underlying histogram
    cdf=np.cumsum(hist)

    # if this is a cdf that has values in it, proceed to resample
    if cdf[-1]>0:
        cdf = cdf / cdf[-1]
        
        # set the random seed
        random.seed(seed)
        # get N random uniform values
        values = np.random.rand(N)

        # find their location/ranl
        value_bins = np.searchsorted(cdf, values)

        # resample
        random_from_cdf = bins[value_bins]

        # go back to the above program level, returning the resampled pdf
        return(random_from_cdf)
    
    # otherwise, return missing values as the samples
    else:
        return(np.repeat(np.nan,N))
    
# Function to subset the natcount data structure along a single specific percentile
def get_1percentile_natcounts(pi,natcountdat):
    
    # slice the data along the chosen percentile
    ppi=natcountdat.sel(pp=pi)
        
    # go back to the above program level
    return(ppi)

### ------------------- METEOROLOGICAL/PHYSICAL UTILITIES ------------------- ###

# Function to convert Celsius to kelvin
def CtoK(data_in_Celsius):
    return data_in_Celsius+273.15

# Function to convert Kelvin to Celsius
def KtoC(data_in_kelvin):
    return data_in_kelvin-273.15

# define a function to convert longitudes from 180W-180E to 0-360E
def lon180to360(lon180):
    return(lon180 % 360)