# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:33:11 2017

@author: Taran Driver
"""
import numpy as np
import pC2DMS
import pC2DMSUtils
import scipy.io #for saving as a mat file
import time

textFile=r"D:\Vitali\Data\20160905_1151_UN14_2+_1nM_CID_NCE35_maxInjectTime1000ms_Turbo.txt"
path=r"D:\Vitali\Data\20160905_1151_UN14_2+_1nM_CID_NCE35_maxInjectTime1000ms_Turbo_test01292020"

saveName=path+'/3000feats_jackknifeResampled_defaultParams_100rays'
numScans='all'

#%%
print 'path:', path
print numScans, 'scans'
print 'save in', saveName

start = time.time()
pC2DMSUtils.readTextFile(textFile, path)
readTime = time.time()
scan1=pC2DMS.Scan(path)
scanTime = time.time()
map1=pC2DMS.PCovMap(scan1, scan1.tic(), numScans=numScans)
mapTime = time.time()
np.save(path+'/PCV_TIC_map.npy', map1.array) #this saves the pcv map itself
features=map1.analyse(3000) #this can be 
analyseTime = time.time()
np.save(saveName+'.npy', features)

#%% Added by Ruth for 3D plotting
# - - save for matlab if needed - -
ap = np.load(path+r"\array_parameters.npy")
sliceSize = ap[3]
minMZ = ap[5]
maxMZ = ap[6]
x = y = np.arange(minMZ, maxMZ, sliceSize)
X, Y = np.meshgrid(x, y)
Z = pC2DMSUtils.cutAC(map1.array)
OneD=scan1.oneD()
#**maybe plot the enhanced scan using readTextFile(textFile, path)
scipy.io.savemat(path+'\pCovMapPrepped.mat', dict(X=X, Y=Y, Z=Z, OneD=OneD))