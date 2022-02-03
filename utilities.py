"""
This module code contains functions that we use throughout the attribution framework 
codebase, by first appending the directory that contains this set of utilities, and then
calling the particular function we want to use.

Author: [Daniel Gilford](https://github.com/dgilford)
Last updated: 1/31/2022 by Daniel Gilford
"""

# import modules to use in functions
import numpy as np
import xarray as xr
from datetime import datetime

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

# Functions to find the middle element of a list (to find the middle time from a datetime64 array)
# def middle_element_dt64(lst):
#     return lst[len(lst) // 2]

# def middle_element(lst):
#   if len(lst) % 2 != 0:
#     return lst[len(lst) // 2]
#   else:
#     return lst[len(lst) // 2 + len(lst) // 2 - 1]

# Function to find the locations of the leap year in the index
def is_leap_and_29Feb(s):
    return (s.index.year % 4 == 0) & \
           ((s.index.year % 100 != 0) | (s.index.year % 400 == 0)) & \
           (s.index.month == 2) & (s.index.day == 29)

# Function to take a slice between two years with dt64 type
def dt64_yrslice(lower_yr=1985,upper_yr=2015):
    slice_range=[str(int(lower_yr))+'-01-01',str(int(upper_yr))+'-12-31']
    sliceout=slice(slice_range[0],slice_range[1])
    return(sliceout)

### ------------------- DATA MANAGEMENT ------------------- ###

# Function to 1d interpolation to fill missing values
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