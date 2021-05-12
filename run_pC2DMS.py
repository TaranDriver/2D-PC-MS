# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:33:11 2017

@author: Taran Driver
"""
import numpy as np
import pC2DMS
import pC2DMSUtils
import time

textFile=None # savePath to text file containing MS data

savePath=None # savePath to save converted processed data

saveName=savePath+'/3000feats_jackknifeResampled'
numScans='all'

#%%
print 'path to save:', savePath
print numScans, 'scans'
print 'save in', saveName

start = time.time()
pC2DMSUtils.readTextFile(textFile, savePath)
readTime = time.time()
scan1=pC2DMS.Scan(savePath)
scanTime = time.time()
map1=pC2DMS.PCovMap(scan1, scan1.tic(), numScans=numScans)
mapTime = time.time()
np.save(savePath+'/PCV_TIC_map.npy', map1.array) 
#this saves the pcv map itself
features=map1.analyse(3000) #this can be 
analyseTime = time.time()
np.save(saveName+'.npy', features)