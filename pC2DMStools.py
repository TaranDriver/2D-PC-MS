# -*- coding: utf-8 -*-
"""
@author: Taran Driver

This module holds useful functions for pC-2DMS analysis
"""

import numpy as np
import scipy.io

def circList(r):
    'Returns all indices within circle of radius r'
    return [[x, y] for x in range(r+1) for y in range(r+1) if x**2+y**2<=r**2]
    
def clearCirc(array, r, c, circList, setVal=0):
    'Sets all elements within shape specified by circList to setVal'
    for x in circList:
        try:
            array[r+x[0], c+x[1]]=setVal
        except:
            pass
        try:
            array[r+x[0], c-x[1]]=setVal        
        except:
            pass
        try:
            array[r-x[0], c+x[1]]=setVal
        except:
            pass
        try:
            array[r-x[0], c-x[1]]=setVal
        except:
            pass
    
def covXI(scanList, varList):
    'Compute the covariance between a scan list and a single variable'   
    assert len(varList) == scanList.shape[0]
    
    numScans = len(varList)
    numSlices = scanList.shape[1]
    
    SiSx = (varList.sum(axis=0)) * (scanList.sum(axis=0)) #can be kept in
    #np.array format because this is multiplication of array by scalar
    
    Six=np.zeros(numSlices)

    for scanIndex in range(numScans): #scanIndex corresponds to scan
    #number (scanIndex+1) 

        Six += varList[scanIndex] * scanList[scanIndex] #Six for the 
        #single scan indexed by 'scanindex'. Can be kept in np.array format
        #because this is multiplication of array by scalar.
        
    return (Six/(numScans - 1))-(SiSx/(numScans * (numScans - 1)))

def cutAC(arrayToCut, pixAcross=8, threshFrac=1e-6):
    "Cuts along diagonal (x=y) line that is pixAcross pixels wide of a \
    SQUARE array (array being covariance/partial covariance map,\
    the x=y line is the autocorrelation (AC) line). All values above the\
    threshold value (= threshFrac *  max value of array) along this line \
    are set to the threshold value. \
    Runs quicker with these 3 clumsy loops than with 'try' clause.\
    Returns the cut array."
    
    array=np.copy(arrayToCut) #copy otherwise it changes the array in place
    
    thresh = np.nanmax(array) * threshFrac
    arrayWidth = len(array)
    
    for row in range(pixAcross, (arrayWidth - pixAcross)):
        for pixel in range((row - pixAcross), (row + (pixAcross + 1))):
            if array[row, pixel] > thresh:
                array[row, pixel] = thresh 
    
    for row2 in range((arrayWidth - pixAcross), arrayWidth):
        for col in range((row2 - pixAcross), arrayWidth):
            if array[row2, col] > thresh:
                array[row2, col] = thresh 
    
    for row3 in range(0, pixAcross):
        for col3 in range(0, (row3 + (pixAcross + 1))):
            if array[row3, col3] > thresh:
                array[row3, col3] = thresh          
                
    return array
        
def maxIndices(array):
    "Returns row and column index of the maximum value in a 2D array"
    posMax = np.unravel_index(array.argmax(), array.shape)
    return posMax[0], posMax[1]
    
def minIndices(array):
    "Returns row and column index of the minimum value in a 2D array"
    posMin = np.unravel_index(array.argmin(), array.shape)
    return posMin[0], posMin[1]

def removeACFeats(featList, acThresh=5.5):
    'Removes all features closer than or equal to acThresh Da in m/z.\
    Returns numpy array. Takes list or numpy array as featList'
    featList=np.asarray(featList) #ensures input is numpy array to allow the 
    #fancy indexing below
    return featList[np.array([(abs(feat[0]-feat[1])>acThresh) for feat in \
    featList])]

def saveMatFile(array, saveTo, fieldName='array'):
    'No need to specify .mat extension'
    scipy.io.savemat(saveTo, {fieldName: array})
    
def saveSyxEinSum(scanFolder, numScans=13):
    "Calculate Syx and save in scan folder, this is a computationally \
    heavy operation and it is useful to have the result saved for further \
    analyses. Uses numpy einsum (Einstein summation) routine" 
    
    print 'Performing saveSyxEinSum for '+scanFolder+' for '+str(numScans)+\
    ' scans'        
    
    if numScans=='all':
        params = np.load(scanFolder + '/array_parameters.npy')
        numScans=int(params[1])
        print '(all scans = '+str(numScans)+')'

    array = np.load(scanFolder + '/array.npy')
    
    Syx=np.einsum('ij,ik->jk', array[1:numScans+1], array[1:numScans+1])  
    #row 0 of array holds m/z values so start from from 1
    np.save(scanFolder + '/Syx_' + str(numScans) + '_scans.npy', Syx)
    
    print 'Completed saveSyxEinSum for '+scanFolder+' for '+str(numScans)+\
    ' scans'
    if numScans=='all':
        print '(all scans = '+str(int(params[1]))+')'
    
    return

def sortList(listToSort, sortCol=3):
    'Default sortCol is 3 (significance). Returns numpy array.'
    """Takes list or numpy array as listToSort"""
    scores=[entry[sortCol] for entry in listToSort]
    return np.array([listToSort[index] for index in \
    reversed(np.argsort(scores))])
    
def varII(varList):
    'Compute the variance of a single variable, i'
    numScans = len(varList)
    Si2 = (varList**2).sum(axis=0)    
    S2i = varList.sum(axis=0)**2

    return (Si2 - S2i/numScans)/(numScans-1)
    
#%%
"""This is old and very poorly written but works"""
    
def readTextFile(textFile, fullScanFolder):
    """Function reads in mass spectral scan from .txt file of Thermo LTQ XL MS 
    after conversion from .raw file by Xcalibur software (File Converter tool).
    Reads text file line-by-line, it's typically too big to load into RAM.
    
    Void function, performs:
    1) Save in full scan folder 'array' with rows 1: being intensity readings 
    sampled at the m/z specified in row 0.
    2) Save in scan folder 'array_parameters.npy', which holds details of the 
    scan itself and how it was read in. It is now an artefact of how the
    pC2DMS software used to run, and many fields are now irrelevant (many other 
    fields are information that is available elsewhere, e.g. through direct 
    inspection of 'array.npy', but are accessed faster using 
    'array_parameters.npy').
    
    This software refers to each m/z sampling point as a 'slice', Thermo 
    software refers to it as a 'packet'.
    
    This function assumes uniform spacing in m/z of points at which the
    ion intensity is sampled, and consistent sampling points across all scans. 
    Non-uniform sampling density that is consistent across scans may require 
    interpolation and integration, any sampling density that is
    inconsistent across scans definitely requires interpolation and 
    integration. 
    Code to implement this (which does so as a stand-alone .py program) is 
    found in:
    "D:/Taran/Computing/Software/fullScript/usefulArchive/ReadTextFile.py"
    """ 
    
    print 'Reading file '+textFile+' to '+fullScanFolder
    
    startLine = 24 #line tells you number of readings
    lineWithPacket = 36 #line of first packet
    scan = 0 #set to 0, to be increased over iterations
    mzSlice = 0 #set to 0, to be increaed over iterations
    
    with open(textFile) as f:
        for i, line in enumerate(f):
      
            if i == 4: #5th line, tells you how many scans in text file. This
            #has historically occasionally varied from the number of scans 
            #actually available to be read in.
                firstScan = int(line.split(',')[0].split(' ')[2]) 
                lastScan = int(line.split(',')[1].split(' ')[3]) 
                
                numScans = (lastScan - firstScan) + 1  
                
            elif i == 5: #6th line, tells you min m/z and max m/z of the scan
                minMZ = float(line.split(',')[0].split(' ')[2])
                maxMZ = float(line.split(',')[1].split(' ')[3])

            if i == startLine:
         
                numSlices = int(line.split(',')[0].split(' ')[2]) #gives the 
                #number of intensity sampling points in this first scan,
                #should be the same number as for the other scans in the file
                
                array = np.zeros((numScans + 1, numSlices))  #declare final 
                #array of all spectra now we have number of scans and 
                #number of slices
                
            if i == lineWithPacket:
                
                if mzSlice < numSlices: #indexing for mzSlice starts at 0
    
                    array[0, mzSlice] = \
                    float("{0:.3f}".format(float(line.split(',')[2].\
                    split(' ')[3])))
        
                    mzSlice += 1
                        
                    lineWithPacket += 3 #spacing between
                    #lines with info on is 3 lines
                
                else:
                    break #break this loop to avoid needing to test both
                    #of the above conditions for each line.

    #Now reset these two
    scan = 0
    mzSlice = 0
    
    with open(textFile) as f:
        for i, line in enumerate(f):
                
            if i == startLine:
                
                scan += 1
                if scan % 100 == 0:
                    print 'Reading scan number '+str(int(scan))
                mzSlice = 0
                lineWithPacket = startLine + 12
                startLine = startLine + 16 + (3 * numSlices)  #next scan will 
                #have a different value for startLine
                
            elif i == lineWithPacket:
                if mzSlice < numSlices: #This conditional shouldn't be needed
                #provided the number of slices is the same in each scan as (is
                #declared at the start of) the first scan. I have kept it here
                #because at one point it wasn't the case.
                    splitLine = line.split(',')
                    
#                    #This is a historic check. Can comment back in if
#                    #required. The exception has never been raised as of 
#                    #18/10/2016. It checks the m/z value for this slice 
#                    #number in the current scan corresponds to (to 3 d.p.) 
#                    #the m/z for this slice number in the first scan 
#                    #[the slice-to-mz mapping in row 0 of the array is done 
#                    #from the first scan].
#                    
#                    mzValue = float("{0:.3f}".format(float(splitLine[2].\
#                    split(' ')[3])))
#                    
#                    if mzValue != array[0, mzSlice]: 
#                        raise ValueError('m/z values misaligned: scan', scan,\
#                        ', slice', mzSlice, 'should be', array[0, mzSlice], \
#                        'but is', mzValue)
                                            
                    array[scan, mzSlice] = float(splitLine[1].split(' ')[3])

                    mzSlice += 1 #increase mz Slice we're filling
                    lineWithPacket += 3 #spacing between
                    #lines with info on is 3 lines
    
    f.close()
    
    #Parameters for 'array_parameters.npy'
    params = np.zeros(8)

    params[0] = numSlices
    params[1] = scan #Total number of scans read in to the array. 
    #Historically sometimes not all scans declared at the top of the 
    #text file have been read in. When this happens, 
    #it does not affect the other scans that are read in to the array.
    params[2] = numScans #Total number of scans specified at top of text file.
    #See above, historically it has sometimes been the case that this number
    #is larger than the number of scans actually read in.
    sliceSize = np.average(np.diff(array[0, :])) #Computational limits 
    #means that some specified slice sizes
    #(e.g. 1/3 Da for Turbo mode in LTQ XL) are represented as e.g. two times
    #0.33 and one time 0.34 (and repeat), so this step provides uniformly
    #spaced m/z's. CAREFUL here though - if you start needing super precise
    #sub-Da resolution on maps, you may have to omit this step and deal with 
    #non-uniform spacings in m/z as they come.
    params[3] = sliceSize
    params[4] = 0 #this was 'interpInt', not needed when no interpolation
    #takes place.
    params[5] = minMZ #as specified in the header of the full text file
    params[6] = maxMZ #as specified in the header of the full text file
    params[7] = sliceSize #this was oneDDataInt which is not needed when no
    #interpolation takes place (and automatically has the same value as 
    #sliceSize when there is no interpolation and integration).
    
    array = array[:scan + 1,:] #truncates the array on the occasion that the
    #number of scans read in is lower than the number of scans declared at the
    #text file, see above.
    
    np.save(fullScanFolder + '/array.npy', array)
    np.save(fullScanFolder + '/array_parameters.npy', params)
    
    print 'readTextFile complete for '+fullScanFolder    
    
    return