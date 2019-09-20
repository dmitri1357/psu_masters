"""
The following is a tool I wrote to interpolate and extract geospatial data
using a unit circle of specified radius and degree resolution.

This method returns a vector of values referenced to a polar coordinate grid,
centered around the latitude/longitude point of interest.

-Dmitri Kalashnikov
"""

import numpy as np
import geopy
import geopy.distance as gd
import scipy.interpolate as si


def radial_interp(array, array_latmin, array_latmax, array_latstep,
                  array_lonmin, array_lonmax, array_lonstep,
                  interp_centerlat, interp_centerlon, start_dist, dist_step,
                  end_dist, degree_resolution):
    """
    array = 2D input array of values from which to interpolate (data should be spatial
            and continuous on a regular grid, e.g. raster, reanalysis, climate model output, etc)

    array_latmin = southern latitude extent of input data array
    array_latmax = northern latitude extent of input data array
    array_latstep = latitudinal resolution (in degrees) of input data array
    array_lonmin = western longitude extent of input data array
    array_lonmax = eastern longitude extent of input data array
    array_lonstep = longitudinal resolution (in degrees) of input data array

    interp_centerlat = central latitude point around which the unit circle interpolator will be built
    interp_centerlon = central longitude point around which the unit circle interpolator will be built

    start_dist = radius distance in km (from central point) of first (innermost) interpolation ring
    dist_step = radius distance in km between each interpolation ring
    end_dist = radius distance in km (from central point) of last (outermost) interpolation ring;
               this distance defines the radius of the circle and the outer edge of
               interpolated values relative to the origin latitude/longitude point (e.g. 2500 km)

    degree_resolution = spacing, in degrees, between each interpolation point on interpolation rings
    """

    array_lats = np.arange(array_latmin,array_latmax+1,array_latstep)
    array_lons = np.arange(array_lonmin,array_lonmax+1,array_lonstep)
    radial_increments = np.arange(start_dist,end_dist+1,dist_step)
    azimuth_increments = np.arange(0+degree_resolution,360+1,degree_resolution)
    latvec = []
    lonvec = []

    for km in radial_increments:
        for deg in azimuth_increments:
            start = geopy.Point(interp_centerlat,interp_centerlon)
            circ = gd.distance(kilometers = km)
            dest = circ.destination(point = start, bearing = deg)
            latvec.append(dest[0])
            lonvec.append(dest[1])

    interp_vals = si.interpn((array_lons,array_lats),array,(lonvec,latvec))
    return interp_vals


