ISIMIP3BASD
-----------
This is the code base used for bias adjustment and statistical downscaling
in phase 3 of the Inter-Sectoral Impact Model Intercomparison Project
(ISIMIP, <https://www.isimip.org/>).



DOCUMENTATION
-------------
Stefan Lange: Trend-preserving bias adjustment and statistical downscaling with
ISIMIP3BASD (v1.0), Geoscientific Model Development Discussions, 2019.



COPYRIGHT
---------
(C) 2019 Potsdam Institute for Climate Impact Research (PIK)



LICENSE
-------
ISIMIP3BASD is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the
Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ISIMIP3BASD is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with ISIMIP3BASD. If not, see <http://www.gnu.org/licenses/>.



REQUIREMENTS
------------
ISIMIP3BASD is writtin in Python 3. It has been tested to run well with the
following Python release and package versions.
- python 3.6.6
- numpy 1.14.3
- scipy 1.1.0
- iris 2.2.0
- dask 0.19.3



HOW TO USE
----------
The bias_adjustment module provides functions for bias adjustment of climate
simulation data using climate observation data with the same spatial and
temporal resolution.

The statistical_downscaling module provides functions for statistical
downscaling of climate simulation data using climate observation data with the
same temporal and higher spatial resolution.

The utility_functions module provides auxiliary functions used by the modules
bias_adjustment and statistical_downscaling.

It is assumed that prior to applying the statistical_downscaling module,
climate simulation data are bias-adjusted at their spatial resolution using the
bias_adjustment module and spatially aggregated climate observation data.

The modules bias_adjustment and statistical_downscaling are written to work
with input and output climate data stored in the NetCDF file format. 

Thanks to their many parameters, the bias adjustment and statistical downscaling
methods implemented here are applicable to many climate variables. Parameter
values can be specified via command line options to the main functions of the
modules bias_adjustment and statistical_downscaling. The variable-specific
parameter values used to produce the results presented in Lange (2019)
<https://doi.org/> are listed in the respective module description.



CONTACT
-------
<slange@pik-potsdam.de>



AUTHOR
------
Stefan Lange
Potsdam Institute for Climate Impact Research (PIK)
Member of the Leibniz Association
P.O. Box 60 12 03
14412 Potsdam
Germany
