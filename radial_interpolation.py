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


def define_radial_grid(start_radius, radius_step, end_radius, degree_resolution):
    """
    start_radius = radius distance in km (from origin) of first (innermost) interpolation ring.
                   If start_radius = 0, interpolation at origin and at 0 degree azimuth
                   will be duplicated in order to produce a continous contourf plot for visualization.
                   If start_radius is set to any value higher than zero, both the origin and
                   0 degree azimuth will be interpolated only once. Use this version if computing
                   spatial statistics as all interpolation points will be unique. Use version
                   with start_radius = 0 for plotting, as otherwise the plot will show no values
                   around center and no values in a vertical slice at 0 degrees from center.
    radius_step = radius distance in km between each interpolation ring. When start_radius != 0,
                  radius_step will be equal to start_radius.
    end_radius = radius distance in km (from origin) of last (outermost) interpolation ring;
               this distance defines the radius of the full circle and the outer edge of
               interpolated values relative to the origin latitude/longitude point (e.g. 2500 km)
    degree_resolution = spacing, in degrees, between each interpolation point on unit circle.
                        For example, specifying a value of 1 will yield 360 interpolation points
                        on each interpolation ring.


    *** Returns vectors of polar coordinates (kilometers, degrees) defining the radial
        interpolation grid in geographic space
    """

    assert start_radius >= 0, "radius must be positive value"

    if start_radius == 0:
        radius_inc = np.arange(start_radius,end_radius+1,radius_step)
        degree_inc = np.arange(0,360+1,degree_resolution)

    else:
        radius_inc = np.arange(start_radius,end_radius+1,radius_step)
        degree_inc = np.arange(0+degree_resolution,360+1,degree_resolution)

    return radius_inc, degree_inc


def radial_interp(array, array_lats, array_lons, interp_centerlat, interp_centerlon,
                  radius_inc, degree_inc):
    """
    array = 2D input array of values from which to interpolate (data should be spatial
            and continuous on a regular grid, e.g. raster, reanalysis, climate model output, etc)
    array_lats = vector of latitude values defining input data array
    array_lons = vector of longitude values defining input data array
    interp_centerlat = latitude defining origin around which the unit circle interpolator
                       will be built
    interp_centerlon = longitude defining origin around which the unit circle interpolator
                       will be built
    radius_inc = distance (km) between interpolation rings
    degree_inc = azimuth resolution (degrees) between interpolation points


    *** Returns vector of interpolated values drawn from input array at every lat,lon point on
        radial grid
    """

    assert radius_inc[0] >= 0, "starting radius must not be negative"

    interp_lats = []
    interp_lons = []

    if radius_inc[0] == 0:

        for km in radius_inc:
            for deg in degree_inc:
                start = geopy.Point(interp_centerlat,interp_centerlon)
                transect = gd.distance(kilometers = km)
                dest = transect.destination(point = start, bearing = deg)
                interp_lats.append(dest[0])
                interp_lons.append(dest[1])

    else:

        for km in radius_inc:
            for deg in degree_inc:
                start = geopy.Point(interp_centerlat,interp_centerlon)
                transect = gd.distance(kilometers = km)
                dest = transect.destination(point = start, bearing = deg)
                interp_lats.append(dest[0])
                interp_lons.append(dest[1])
        interp_lats = np.hstack([interp_centerlat, interp_lats])
        interp_lons = np.hstack([interp_centerlon, interp_lons])

    interp_vals = si.interpn((array_lons,array_lats),array,(interp_lons,interp_lats))
    del interp_lats, interp_lons

    return interp_vals





