# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:30:58 2019

@author: dmitri4
"""

# Random Forest Model for lightning prediction
import numpy as np
import geopy
import geopy.distance
import scipy.interpolate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd

'''
This code is exploratory work from my Master's thesis at PSU, attempting
to predict lightning days in interior Pacific Northwest using meteorological
variables. Binary classification (1 = lightning, 0 = no lightning) using Random
Forest Classifier.

-Dmitri Kalashnikov
'''

# import meteorological variables
z500_ja = np.load('z500_ja.npy')
slp_ja = np.load('slp_ja.npy')
tqv_ja = np.load('tqv_ja.npy')
lapse700500_ja = np.load('lapse700500_ja.npy')
omega500_ja = np.load('omega500_ja.npy')
qv500_ja = np.load('qv500_ja.npy')
qv700_ja = np.load('qv700_ja.npy')
qv2M_ja = np.load('qv2M_ja.npy')
tqi_ja = np.load('tqi_ja.npy')

# import lightning info, subset to study area (interior Pacific Northwest)
cg_8cells = np.load('cg_8cells.npy')
cg_8cells_vec = np.reshape(cg_8cells, 14880)
idx_cg = np.where(cg_8cells_vec > 0) # 5,830 lightning days across all 8 cells
idx_no_cg = np.repeat(0,14880)
idx_no_cg[idx_cg] = 1
idx_all = idx_no_cg # index of all days (1 = lightning, 0 = no lightning)

lats_8 = [47,45,45,45,43,43,43,43] # center lats
lons_8 = [-116,-120,-118,-116,-122,-120,-118,-116] # center lons
merra_lats = np.arange(8.5,72.5,0.5) # full extent encompassing all circles
merra_lons = np.arange(-161.25,-67.5,0.625) # full extent encompassing all circles

# iteration steps --> 1,860 days at each grid cell (x8)
step1 = np.arange(0,1860,1)
step2 = np.arange(1860,3720,1)
step3 = np.arange(3720,5580,1)
step4 = np.arange(5580,7440,1)
step5 = np.arange(7440,9300,1)
step6 = np.arange(9300,11160,1)
step7 = np.arange(11160,13020,1)
step8 = np.arange(13020,14880,1)
steps = [step1,step2,step3,step4,step5,step6,step7,step8]

# radial interpolation of atmpospheric variables
# repeated for other variables, and different radius distances (not all shown here)
# this example is for Geopotential Heights (z500) within 1,500 km of location
z500_1500km = np.empty([14880,1081])
for k in range(8):
    lon = lons_8[k]
    lat = lats_8[k]
    latvec = []
    lonvec = []
    step = steps[k]
    # calculate interpolation grid
    for a in np.arange(50,1550,50): # 50 km radial increments (rho's)
        for b in np.arange(10,370,10): # 10 degree azimuth increments (theta's)
            start = geopy.Point(lat,lon) # origin
            circ = geopy.distance.distance(kilometers = a) # great circle distance
            dest = circ.destination(point = start, bearing = b) # transect along great circle
            latvec.append(dest[0])
            lonvec.append(dest[1])
    latvec = np.hstack([lat, latvec]) # add origin lat
    lonvec = np.hstack([lon, lonvec]) # add origin lon
    # interpolate underlying values using radial grid
    interp_vals = scipy.interpolate.interpn((merra_lons,merra_lats),z500_ja,(lonvec,latvec))
    del latvec, lonvec
    interp_vals = np.squeeze(interp_vals)
    z500_1500km[step,:] = interp_vals.T # interpolated values from r = 1,500 km grid

# some summary statistics
z500_500km_mean = np.mean(z500_500km, axis = 1)
z500_1000km_mean = np.mean(z500_1000km, axis = 1)
z500_1500km_mean = np.mean(z500_1500km, axis = 1)
z500_500km_ptp = np.ptp(z500_500km, axis = 1)
z500_1000km_ptp = np.ptp(z500_1000km, axis = 1)
z500_1500km_ptp = np.ptp(z500_1500km, axis = 1)

# differences between points in NE and SW quadrant,
# as anomaly gradient between these locations matters for
# moisture advection and other favorable features for lightning development
z500_300km_diff1 = z500_1500km[:,222] - z500_1500km[:,240]
z500_300km_diff2 = z500_1500km[:,223] - z500_1500km[:,241]
z500_300km_diff3 = z500_1500km[:,224] - z500_1500km[:,242]
z500_300km_diff4 = z500_1500km[:,225] - z500_1500km[:,243]
z500_400km_diff1 = z500_1500km[:,296] - z500_1500km[:,314]
z500_400km_diff2 = z500_1500km[:,297] - z500_1500km[:,315]
z500_400km_diff3 = z500_1500km[:,298] - z500_1500km[:,316]
z500_400km_diff4 = z500_1500km[:,299] - z500_1500km[:,317]
z500_500km_diff1 = z500_1500km[:,370] - z500_1500km[:,388]
z500_500km_diff2 = z500_1500km[:,371] - z500_1500km[:,389]
z500_500km_diff3 = z500_1500km[:,372] - z500_1500km[:,390]
z500_500km_diff4 = z500_1500km[:,373] - z500_1500km[:,391]
z500_600km_diff1 = z500_1500km[:,444] - z500_1500km[:,462]
z500_600km_diff2 = z500_1500km[:,445] - z500_1500km[:,463]
z500_600km_diff3 = z500_1500km[:,446] - z500_1500km[:,464]
z500_600km_diff4 = z500_1500km[:,447] - z500_1500km[:,465]
z500_700km_diff1 = z500_1500km[:,518] - z500_1500km[:,536]
z500_700km_diff2 = z500_1500km[:,519] - z500_1500km[:,537]
z500_700km_diff3 = z500_1500km[:,520] - z500_1500km[:,538]
z500_700km_diff4 = z500_1500km[:,521] - z500_1500km[:,539]
z500_800km_diff1 = z500_1500km[:,592] - z500_1500km[:,610]
z500_800km_diff2 = z500_1500km[:,593] - z500_1500km[:,611]
z500_800km_diff3 = z500_1500km[:,594] - z500_1500km[:,612]
z500_800km_diff4 = z500_1500km[:,595] - z500_1500km[:,613]
z500_900km_diff1 = z500_1500km[:,666] - z500_1500km[:,684]
z500_900km_diff2 = z500_1500km[:,667] - z500_1500km[:,685]
z500_900km_diff3 = z500_1500km[:,668] - z500_1500km[:,686]
z500_900km_diff4 = z500_1500km[:,669] - z500_1500km[:,687]
z500_1000km_diff1 = z500_1500km[:,740] - z500_1500km[:,758]
z500_1000km_diff2 = z500_1500km[:,741] - z500_1500km[:,759]
z500_1000km_diff3 = z500_1500km[:,742] - z500_1500km[:,760]
z500_1000km_diff4 = z500_1500km[:,743] - z500_1500km[:,761]

# interpolation for atmospheric moisture content (TQV)
tqv_500km = np.empty([14880,361])
for k in range(8):
    lon = lons_8[k]
    lat = lats_8[k]
    latvec = []
    lonvec = []
    step = steps[k]
    for a in np.arange(50,550,50):
        for b in np.arange(10,370,10):
            start = geopy.Point(lat,lon)
            circ = geopy.distance.distance(kilometers = a)
            dest = circ.destination(point = start, bearing = b)
            latvec.append(dest[0])
            lonvec.append(dest[1])
    latvec = np.hstack([lat, latvec])
    lonvec = np.hstack([lon, lonvec])
    interp_vals = scipy.interpolate.interpn((merra_lons,merra_lats),tqv_ja,(lonvec,latvec))
    del latvec, lonvec
    interp_vals = np.squeeze(interp_vals)
    tqv_500km[step,:] = interp_vals.T

# means within certain radius distances
tqv_origin = tqv_500km[:,0]
tqv_50km_mean = np.mean(tqv_500km[:,0:37], axis = 1)
tqv_100km_mean = np.mean(tqv_500km[:,0:73], axis = 1)
tqv_150km_mean = np.mean(tqv_500km[:,0:109], axis = 1)
tqv_200km_mean = np.mean(tqv_500km[:,0:145], axis = 1)
tqv_250km_mean = np.mean(tqv_500km[:,0:181], axis = 1)
tqv_300km_mean = np.mean(tqv_500km[:,0:217], axis = 1)
tqv_350km_mean = np.mean(tqv_500km[:,0:253], axis = 1)
tqv_400km_mean = np.mean(tqv_500km[:,0:289], axis = 1)
tqv_450km_mean = np.mean(tqv_500km[:,0:325], axis = 1)
tqv_500km_mean = np.mean(tqv_500km[:,0:361], axis = 1)

# atmospheric moisture at ~10,000 feet (700 hPa pressure level)
qv700_500km = np.empty([14880,361])
for k in range(8):
    lon = lons_8[k]
    lat = lats_8[k]
    latvec = []
    lonvec = []
    step = steps[k]
    for a in np.arange(50,550,50):
        for b in np.arange(10,370,10):
            start = geopy.Point(lat,lon)
            circ = geopy.distance.distance(kilometers = a)
            dest = circ.destination(point = start, bearing = b)
            latvec.append(dest[0])
            lonvec.append(dest[1])
    latvec = np.hstack([lat, latvec])
    lonvec = np.hstack([lon, lonvec])
    interp_vals = scipy.interpolate.interpn((merra_lons,merra_lats),qv700_ja,(lonvec,latvec))
    del latvec, lonvec
    interp_vals = np.squeeze(interp_vals)
    qv700_500km[step,:] = interp_vals.T

# means within certain radius distances
qv700_origin = qv700_500km[:,0]
qv700_50km_mean = np.mean(qv700_500km[:,0:37], axis = 1)
qv700_100km_mean = np.mean(qv700_500km[:,0:73], axis = 1)
qv700_150km_mean = np.mean(qv700_500km[:,0:109], axis = 1)
qv700_200km_mean = np.mean(qv700_500km[:,0:145], axis = 1)
qv700_250km_mean = np.mean(qv700_500km[:,0:181], axis = 1)
qv700_300km_mean = np.mean(qv700_500km[:,0:217], axis = 1)
qv700_350km_mean = np.mean(qv700_500km[:,0:253], axis = 1)
qv700_400km_mean = np.mean(qv700_500km[:,0:289], axis = 1)
qv700_450km_mean = np.mean(qv700_500km[:,0:325], axis = 1)
qv700_500km_mean = np.mean(qv700_500km[:,0:361], axis = 1)

# variables for Random Forest model

# one set of engineered features
features = np.vstack([z500_500km_mean,z500_1000km_mean,z500_1500km_mean,
                      z500_500km_ptp,z500_1000km_ptp,z500_1500km_ptp,z500_300km_diff1,
                      z500_300km_diff2,z500_300km_diff3,z500_300km_diff4,z500_400km_diff1,
                      z500_400km_diff2,z500_400km_diff3,z500_400km_diff4,z500_500km_diff1,
                      z500_500km_diff2,z500_500km_diff3,z500_500km_diff4,z500_600km_diff1,
                      z500_600km_diff2,z500_600km_diff3,z500_600km_diff4,z500_700km_diff1,
                      z500_700km_diff2,z500_700km_diff3,z500_700km_diff4,z500_800km_diff1,
                      z500_800km_diff2,z500_800km_diff3,z500_800km_diff4,z500_900km_diff1,
                      z500_900km_diff2,z500_900km_diff3,z500_900km_diff4,z500_1000km_diff1,
                      z500_1000km_diff2,z500_1000km_diff3,z500_1000km_diff4,tqv_origin,
                      tqv_100km_mean,tqv_200km_mean,tqv_300km_mean,qv700_origin,
                      qv700_100km_mean,qv700_200km_mean,qv700_300km_mean])

# a different set of features
features = np.vstack([tqv_50km_mean,tqv_100km_mean,tqv_150km_mean,tqv_200km_mean,
                      tqv_250km_mean,tqv_300km_mean,tqv_350km_mean,tqv_400km_mean,
                      tqv_450km_mean,tqv_500km_mean,qv700_50km_mean,qv700_100km_mean,
                      qv700_150km_mean,qv700_200km_mean,qv700_250km_mean,qv700_300km_mean,
                      qv700_350km_mean,qv700_400km_mean,qv700_450km_mean,qv700_500km_mean,
                      z500_700km_diff1,z500_700km_diff2,tqv_origin,qv700_origin,
                      z500_700km_diff3,z500_700km_diff4,z500_800km_diff1,z500_800km_diff2,
                      z500_800km_diff3,z500_800km_diff4,z500_900km_diff1,z500_900km_diff2,
                      z500_900km_diff3,z500_900km_diff4,z500_1000km_diff1,z500_1000km_diff2,
                      z500_1000km_diff3,z500_1000km_diff4])
features = features.T

labels = idx_all

# split data into train/test
train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                             test_size = 0.25, random_state = 31)

# Instantiate model with 500 decision trees
rf = RandomForestClassifier(n_estimators = 500, random_state = 31)

# Train the model
rf.fit(train_features, train_labels)

# predict test data
predictions = rf.predict(test_features)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(test_labels, predictions))

# finding important features
feature_imp = pd.Series(rf.feature_importances_).sort_values(ascending=False)
feature_imp
