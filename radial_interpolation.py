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


def radial_interp(array, array_lats, array_lons, interp_centerlat, interp_centerlon,
                  start_radius, radius_step, end_radius, degree_resolution):
    """
    array = 2D input array of values from which to interpolate (data should be spatial
            and continuous on a regular grid, e.g. raster, reanalysis, climate model output, etc)
    array_lats = vector of latitude values defining input data array
    array_lons = vector of longitude values defining input data array
    interp_centerlat = latitude defining origin around which the unit circle interpolator
                       will be built
    interp_centerlon = longitude defining origin around which the unit circle interpolator
                       will be built
    start_radius = radius distance in km (from origin) of first (innermost) interpolation ring
    radius_step = radius distance in km between each interpolation ring
    end_radius = radius distance in km (from origin) of last (outermost) interpolation ring;
               this distance defines the radius of the full circle and the outer edge of
               interpolated values relative to the origin latitude/longitude point (e.g. 2500 km)
    degree_resolution = spacing, in degrees, between each interpolation point on unit circle.
                        For example, specifying a value of 1 will yield 360 interpolation points
                        on each interpolation ring.


    *** Returns vector of interpolated values drawn from input array at every lat,lon point on
        radial grid
    """

    radius_increments = np.arange(start_radius,end_radius+1,radius_step)
    degree_increments = np.arange(0+degree_resolution,360+1,degree_resolution)
    interp_lats = []
    interp_lons = []

    for km in radius_increments:
        for deg in degree_increments:
            start = geopy.Point(interp_centerlat,interp_centerlon)
            transect = gd.distance(kilometers = km)
            dest = transect.destination(point = start, bearing = deg)
            interp_lats.append(dest[0])
            interp_lons.append(dest[1])

    interp_vals = si.interpn((array_lons,array_lats),array,(interp_lons,interp_lats))
    return interp_vals




