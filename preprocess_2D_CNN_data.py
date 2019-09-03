# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:36:36 2019

@author: dmitri4
"""

import numpy as np
import gzip
from scipy.interpolate import griddata
from twilio.rest import Client

'''
This code is exploratory work from my Master's thesis at PSU, attempting
to predict lightning days in interior Pacific Northwest using meteorological
variables.

This program converts interpolated meteorological data to an equal-area
meshgrid for input into a 2D Convolutional Neural Network (CNN) model. The data
had originally been interpolated from geographic (lat-lon) coords onto a unit
circle using polar coordinates in order to normalize distances around each
data point.

-Dmitri Kalashnikov
'''

# variables are 14880 x 756
with gzip.open('z500_deps_2deg_dups.npy.gz', 'rb') as f: # Geopotential heights
    z500_deps_2deg = np.load(f)

with gzip.open('slp_deps_2deg_dups.npy.gz', 'rb') as f: # Sea-level pressure
    slp_deps_2deg = np.load(f)

with gzip.open('tqv_deps_2deg_dups.npy.gz', 'rb') as f: # Atmospheric moisture
    tqv_deps_2deg = np.load(f)

with gzip.open('lapse700500_deps_2deg_dups.npy.gz', 'rb') as f: # Vertical instability
    lapse700500_deps_2deg = np.load(f)

with gzip.open('qv500_deps_2deg_dups.npy.gz', 'rb') as f: # Moisture in mid-troposphere
    qv500_deps_2deg = np.load(f)

with gzip.open('qv700_deps_2deg_dups.npy.gz', 'rb') as f: # Moisture at approx. 10,000 feet
    qv700_deps_2deg = np.load(f)

with gzip.open('qv2M_deps_2deg_dups.npy.gz', 'rb') as f: # Moisture at ground level
    qv2M_deps_2deg = np.load(f)

# scaling values to between 0 and 1, following neural network tutorial
a = z500_deps_2deg
z500_scaled = np.interp(a, (a.min(), a.max()), (0, +1))
a = slp_deps_2deg
slp_scaled = np.interp(a, (a.min(), a.max()), (0, +1))
a = tqv_deps_2deg
tqv_scaled = np.interp(a, (a.min(), a.max()), (0, +1))
a = lapse700500_deps_2deg
lapse700500_scaled = np.interp(a, (a.min(), a.max()), (0, +1))
a = qv500_deps_2deg
qv500_scaled = np.interp(a, (a.min(), a.max()), (0, +1))
a = qv700_deps_2deg
qv700_scaled = np.interp(a, (a.min(), a.max()), (0, +1))
a = qv2M_deps_2deg
qv2M_scaled = np.interp(a, (a.min(), a.max()), (0, +1))

# concatenating scaled values into single input field
all_vars = np.concatenate((z500_scaled, slp_scaled, tqv_scaled, # 104160,756
                           lapse700500_scaled, qv500_scaled, qv700_scaled,
                           qv2M_scaled), axis = 0)

# reshaping to 2D in order to convert polar (theta, rho) circle coords back to cartesian (x,y)
all_vals = np.reshape(all_vars, (104160,21,36)).astype('float32')

# defining function to convert polar coords to cartesian
def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return(x, y)

# preprocessing for converstion of data on polar coordinates onto square meshgrid
rho = np.arange(0,1050,50)/1000 # radial steps (50 km) used for unit circle interpolation of original data
rho_vec = np.repeat(rho,36) # rho coord values for each theta-rho pair
theta = np.deg2rad(np.arange(10,370,10)) # azimuth steps (10 deg.) used for radial interpolation
theta_vec = np.tile(theta,21) # theta coord values for each theta-rho pair

xs = np.sort(np.unique(theta_vec)) # spatially ordered list of all x values
ys = np.sort(np.unique(rho_vec)) # spatially ordered list of all y values
(TH, RH) = np.meshgrid(xs,ys) # explicitly assign polar coords to data grid
(X,Y) = pol2cart(TH, RH) # convert polar coords to cartesian (x,y) using function

# defining dimensions of square meshgrid for input into CNN model
grid_x, grid_y = np.mgrid[0:1:40j, 0:1:40j] # dimension of 40x40 is about every 50 km

# defining coordinates of the meshgrid
grid_x = np.interp(grid_x, (grid_x.min(), grid_x.max()), (-1, +1)) # data is on unit circle (r = 1)
grid_y = np.interp(grid_y, (grid_y.min(), grid_y.max()), (-1, +1)) # data is on unit circle (r = 1)
x = np.reshape(X, (756)) # vectorizing x coords of data values
y = np.reshape(Y, (756)) # vectorizing y coords
points = (x,y) # these are the points at which underlying grid will be interpolated

# since runtime of following interpolation varies greatly based on input dims, I wanted my program
# to notify me when it has finished running. Text message via Twilio was my solution.
account_SID = 'removed for security reasons'
auth_token = 'removed for security reasons'
twilio_cli = Client(account_SID, auth_token)
twilio_num = '+19712564892' # my twilio number
my_cell = '+15039296502' # my cell number

# interpolating values onto square meshgrid (the 'image') for input to 2D CNN model
regrid_vals_scaled = np.empty([104160,40,40])
for k in range(104160):
    values = all_vals[k,:,:]
    values = np.reshape(values, (756))
    regrid_vals_scaled[k,:,:] = griddata(points, values, (grid_x, grid_y), method='linear')
message = twilio_cli.messages.create(body = 'Code has finished', # text me when finished
                                    from_ = twilio_num, to = my_cell)


