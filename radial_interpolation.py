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
    A function to define a radial interpolation grid based on a unit circle, and centered
    on a geographic (lat, lon) point of interest. This methodology was developed by Loikith
    & Broccoli 2012* and was originally written in MATLAB. This Python implementation was 
    developed as part of my M.S. thesis at Portland State University.
    
    -Dmitri Kalashnikov (dmitrik1357@gmail.com)
    
    * Loikith, P. C., and A. J. Broccoli, 2012: Characteristics of Observed Atmospheric 
     Circulation Patterns Associated with Temperature Extremes over North America. J. Climate, 25, 
     7266–7281, https://doi.org/10.1175/JCLI-D-11-00709.1
    
    Parameters
    ----------
    start_radius : radius distance in km (from origin) of first (innermost) interpolation ring.
                   If start_radius = 0, interpolation at origin and at 0 degree azimuth
                   will be duplicated in order to produce a continous contourf plot for visualization.
                   If start_radius is set to any value higher than zero, both the origin and
                   0 degree azimuth will be interpolated only once. Use this version if computing
                   spatial statistics as all interpolation points will be unique. Use version
                   with start_radius = 0 for plotting, as otherwise the plot will show no values
                   around center and no values in a vertical slice at 0 degrees from center.
    radius_step : radius distance in km between each interpolation ring. When start_radius != 0,
                  radius_step will be equal to start_radius.
    end_radius : radius distance in km (from origin) of last (outermost) interpolation ring;
                 this distance defines the radius of the full circle and the outer edge of
                 interpolated values relative to the origin latitude/longitude point (e.g. 2500 km)
    degree_resolution : spacing, in degrees, between each interpolation point on unit circle.
                        For example, specifying a value of 1 will yield 360 interpolation points
                        on each interpolation ring.

    Returns
    -------
    radius_steps, degree_steps : vectors of polar coordinates (kilometers, degrees) defining the radial
                                 interpolation grid in geographic space
    """

    assert start_radius >= 0, "radius values must not be negative"

    if start_radius == 0:
        radius_steps = np.arange(start_radius,end_radius+1,radius_step)
        degree_steps = np.arange(0,360+1,degree_resolution)

    else:
        radius_steps = np.arange(start_radius,end_radius+1,radius_step)
        degree_steps = np.arange(0+degree_resolution,360+1,degree_resolution)

    return radius_steps, degree_steps


def radial_interp(a, a_lats, a_lons, interp_centerlat, interp_centerlon,
                  radius_steps, degree_steps):
    """
    A function to interpolate continuous, geographic data using a unit circle centered
    on a geographic (lat, lon) point of interest. This methodology was developed by Loikith
    & Broccoli 2012* and was originally written in MATLAB. This Python implementation was 
    developed as part of my M.S. thesis at Portland State University.
    
    -Dmitri Kalashnikov (dmitrik1357@gmail.com)
    
    * Loikith, P. C., and A. J. Broccoli, 2012: Characteristics of Observed Atmospheric 
     Circulation Patterns Associated with Temperature Extremes over North America. J. Climate, 25, 
     7266–7281, https://doi.org/10.1175/JCLI-D-11-00709.1
    
    Parameters
    ----------
    a : 2D or 3D input array of values from which to interpolate (data should be spatial
            and continuous on a regular grid, e.g. raster, reanalysis, climate model output, etc);
            with 3rd dimension representing time
    a_lats : vector of latitude values defining input data array
    a_lons : vector of longitude values defining input data array
    interp_centerlat : latitude defining origin around which the unit circle interpolator
                       will be built
    interp_centerlon : longitude defining origin around which the unit circle interpolator
                       will be built
    radius_steps : distance (km) between interpolation rings
    degree_steps : azimuth resolution (degrees) between interpolation points

    Returns
    -------
    interp_vals :  For 2D input array: returns vector of interpolated values drawn from input 
                   array at every (lat,lon) point on radial grid. For 3D input arrary, returns 2D array 
                   of (interpolated values, timesteps)
    """

    assert radius_steps[0] >= 0, "starting radius must not be negative"

    interp_lats = []
    interp_lons = []

    if radius_steps[0] == 0:

        for km in radius_steps:
            for deg in degree_steps:
                start = geopy.Point(interp_centerlat,interp_centerlon)
                transect = gd.distance(kilometers = km)
                dest = transect.destination(point = start, bearing = deg)
                interp_lats.append(dest[0])
                interp_lons.append(dest[1])

    else:

        for km in radius_steps:
            for deg in degree_steps:
                start = geopy.Point(interp_centerlat,interp_centerlon)
                transect = gd.distance(kilometers = km)
                dest = transect.destination(point = start, bearing = deg)
                interp_lats.append(dest[0])
                interp_lons.append(dest[1])
        interp_lats = np.hstack([interp_centerlat, interp_lats])
        interp_lons = np.hstack([interp_centerlon, interp_lons])

    interp_vals = si.interpn((array_lons,array_lats),array,(interp_lons,interp_lats))
    return interp_vals




