# -*- coding: utf-8 -*-
"""
@author: Taran Driver

This module defines the classes required to produce a pC-2DMS map,
identify features and calculate their pC-2DMS correlation score,
given input of text file of a converted Thermo .raw file.
"""

import os
import numpy as np
from scipy import interpolate
from scipy import ndimage
from pC2DMStools import maxIndices, varII, covXI, cutAC, saveSyxEinSum, circList, clearCirc, readTextFile
from matplotlib.path import Path
import time
import argparse

class Scan:
    "All the data for a scan is saved in path designated by scanFolder"
    def __init__(self, scanFolder, AGCtarget=100):
        'initialise class by loading all data'
        self.scanFolder = scanFolder
        self.AGCtarget = AGCtarget
        self.normFactor = AGCtarget/10000.
        self.scanList = self.normFactor*np.load(scanFolder + '/array.npy')[1:]
        params = np.load(scanFolder + '/array_parameters.npy')
        self.sliceSize = params[3]
        self.minMZ = params[5] #using this currently means index2mz (and 
        # equally, mz2index) gives an
        # m/z value which is 1*self.sliceSize lower than the mz value that is
        # provided for the reading in the text file, but for the charge states
        # and scan mode that we have been working with this provides more
        # accurate m/z read off from the map
        self.maxMZ = params[6]
        #The text file that gets read in reports a min mz and max mz and then
        #gives intensity readings at each m/z value. The min m/z value at 
        #which is gives an intensity reading for is one sliceSize higher than
        #the reported min m/z, the max m/z value that it gives an intensity 
        #reading for is the same as the reported max m/z.
        self.fullNumScans = int(params[1])

    def tic(self):
        'return total ion count for scan'
        return self.scanList.sum(axis=1)
    
    def index2mz(self, index):
        "Provides the m/z value relating to the specified index for the \
        relevant scan dataset. index need not be an integer value."
        return self.minMZ + index*self.sliceSize
        
    def mz2index(self, mz):
        "Provides the closest (rounded) integer m/z slice index relating to \
        the m/z value for the relevant scan dataset."
        if mz < self.minMZ or mz > self.maxMZ:
            raise ValueError, 'm/z value outside range of m/z values for this scan'
        if mz % self.sliceSize < self.sliceSize/2:
            return int((mz - self.minMZ)/self.sliceSize)
        else:
            return int((mz - self.minMZ)/self.sliceSize + 1)
        #this doesn't work with arrays, yet. (index2mz does)
    
    def oneD(self, numScans='all'):
        'provides summed 1D spectra over specified scans in scan object'
        x=self.fullNumScans if numScans=='all' else numScans
        return self.scanList[:x].sum(0)    

class Map:
    'Simple or partial covariance map'
    def __init__(self, scan, numScans='all'):
        self.scan = scan
        if numScans=='all':
            self.numScans = self.scan.fullNumScans
        else:
            self.numScans = numScans
        self.build()
    
    def syx(self):
        'Syx is attribute, syx is method. syx is <XY> matrix'
        try: 
            return self.Syx
        except:
            syxPath = self.scan.scanFolder + \
            '/Syx_'+str(self.numScans)+'_scans.npy'
            
            if os.path.isfile(syxPath):
                return self.scan.normFactor**2 * np.load(syxPath)
            else:
                print 'Syx not saved for this map, beginning calculation with saveSyxEinSum...'
                saveSyxEinSum(self.scan.scanFolder, numScans=self.numScans)
                return self.scan.normFactor**2 * np.load(syxPath)
            
    def loadSyx(self):
        'Syx is attribute, syx is method'
        self.Syx = self.syx()
        
    def analyse(self, numFeats, clearRad=25, chemFilt=[], \
        chemFiltTol=2.0, shapeFilt=False, shFiltThr=-0.2, shFiltRef='map',\
        shFiltRad=15, breakTime=3600,\
        pixOut=15, comPixOut=3, cutPeaks=True, integThresh=0, numRays=100,\
        perimOffset=-0.5, pixWidth=1, sampling='jackknife', bsRS=None,\
        bsRS_diffs=None, saveAt=False, basePath=None, useT4R=False, \
        printAt=50): 
        #last 13 kwargs (after break) are parameters for sampleFeats
        'Picks numFeats feats from map and then samples them using jackknife'

        indexList=self.topNfeats(numFeats, clearRad=clearRad,\
        chemFilt=chemFilt, chemFiltTol=chemFiltTol, shapeFilt=shapeFilt,\
        shFiltThr=shFiltThr, shFiltRef=shFiltRef, shFiltRad=shFiltRad,\
        breakTime=breakTime, boundFilt=True, boundLimit=pixOut,\
        returnDiscards=False) 
        #boundFilt must be true with boundLimit as pixOut for sampleFeats 
        #to not raise an exception
        
        return self.sampleFeats(indexList, pixOut=pixOut, \
        comPixOut=comPixOut, cutPeaks=cutPeaks, integThresh=integThresh, \
        numRays=numRays, perimOffset=perimOffset, pixWidth=pixWidth, \
        sampling=sampling, bsRS=bsRS, bsRS_diffs=bsRS_diffs, saveAt=saveAt,\
        basePath=basePath, useT4R=useT4R, printAt=printAt)
    
    def sampleFeats(self, indexList, pixOut=15, comPixOut=3,\
        cutPeaks=True, integThresh=0, numRays=100, perimOffset=-0.5,\
        pixWidth=1, sampling='jackknife', bsRS=None, bsRS_diffs=None,\
        useT4R=False, saveAt=False, basePath=None, printAt=50):
        'Returns m/z\'s, volume and sig for feats with r,c in indexList'    
        """pixOut is for peak dimensions, comPixOut is for centre of mass
        routine, pixWidth is for peak integration (how many Da each pixel 
        corresponds to, if we care). bsRS and bsRS_diffs only relevant to 
        bootstrap resampling"""
        
        self.Syx=self.syx() #load this into RAM so it runs quicker
        
        featList=np.zeros((len(indexList), 4)) 
        featNo=0
        
        if type(saveAt) is int and basePath is None:
            basePath=raw_input('base file name for peaks file:')        
        
        for indices in indexList:
            peak=self.getPeak(indices[0], indices[1], pixOut=pixOut)
            com_r, com_c=peak.com(pixEachSide=comPixOut)
            
            if cutPeaks:
                if useT4R:
                    template=peak.template4ray(integThresh=integThresh)
                else:
                    template=peak.templateNray(integThresh=integThresh,\
                    numRays=numRays, perimOffset=perimOffset)
                peak.cutPeak(template)
            else:
                template=None #because template is passed as kwarg to the 
                #***ResampleVar method below and needs to be defined
                
            peakVol=peak.bivSplIntegrate(pixWidth=pixWidth)
            if sampling=='jackknife':
                peakVar=peak.jkResampleVar(cutPeak=cutPeaks, template=template)
            #there are other resampling methods that can be called (e.g. have
            #tested bootstrap), choose
            #jackknife for computational efficiency and repeatability
            
            featList[featNo]=round(self.scan.index2mz(indices[0]-\
            comPixOut+com_r),2), round(self.scan.index2mz(indices[1]-\
            comPixOut+com_c),2), peakVol, peakVol/np.sqrt(peakVar)
            
            featNo+=1
            if featNo%printAt==0:            
                print 'sig calculated for feature '+str(featNo)
            if type(saveAt) is int and featNo%saveAt==0:
                np.save(basePath+'_feats'+str(int(featNo-saveAt+1))+'to'+\
                str(int(featNo))+'.npy', featList[featNo-saveAt:featNo])
            
        return featList
        
    def topNfeats(self, numFeats, clearRad=25, chemFilt=[], chemFiltTol=2.0,
                  shapeFilt=False, shFiltThr=-0.2, shFiltRef='map',\
                  shFiltRad=15, boundFilt=True, boundLimit=15, breakTime=3600,\
                  returnDiscards=False):
        #chemFiltTol in Da, chemFilt condition is less than or equal to.
        #shFiltThr is fraction of highest peak on a/c cut pCov map.
        "Return highest numFeats features on map"
        """This looks like it is written by a moron - partially because it \
        was, and partially because I chose not to optimise it much because I \
        currently have more pressing things to do. \
        'returnDiscards' allows to return features discarded by any applied\
        filters as well (this array is the second element in the returned\
        tuple)."""
        
        array=np.triu(cutAC(self.array))
        if shFiltRef=='map': #'shape filter reference' taken globally from map
            shFiltThrAbs=shFiltThr*np.nanmax(array)
        
        #'Picks the top N highest legitimate features'
        feats=np.zeros((numFeats, 2))
        featCount=0
        
        if returnDiscards:
            discardFeats=[] #list of all feats discarded by any of the filters
        
        circListClear=circList(clearRad)
        if shapeFilt:
            circListFilt=circList(shFiltRad)
        
        startTime=time.time()
        
        while featCount<numFeats:
            
            featPass=True
            r,c = maxIndices(array)
            if shFiltRef=='peak': #shape filter reference taken as height of
                shFiltThrAbs=shFiltThr*array[r,c] #each individual peak
            
            #Apply chemical filter if requested                    
            for chemFiltmz in chemFilt:
                if abs(self.scan.index2mz(r) - chemFiltmz) <= chemFiltTol or \
                abs(self.scan.index2mz(c) - chemFiltmz)  <= chemFiltTol: #this 
                #takes m/z of highest pixel, not m/z of CoM of feature
                    featPass=False
                    break           
                
            #Apply shape filter if requested  
            if shapeFilt and featPass:
                for x in circListFilt:
                    if array[r+x[0],c+x[1]]<=shFiltThrAbs or \
                    array[r+x[0],c-x[1]]<=shFiltThrAbs or \
                    array[r-x[0],c+x[1]]<=shFiltThrAbs or \
                    array[r-x[0],c-x[1]]<=shFiltThrAbs:
                        featPass=False
                        break
            
            #Apply boundary filter so that features too close to the edge of 
            #the map to be sampled are not counted
            if boundFilt and featPass:            
                if r < boundLimit or c < boundLimit or r  > len(array) - \
                (boundLimit+1) or c > len(array) - (boundLimit+1):
                    featPass=False
                    
            #Because of the way Python was compiling (everything 
            #indented relative to a for clause, including the continue 
            #statement, is not executed as soon as the break statement is hit, 
            #even if it is unindented relative to the break statement),
            #there was no possible configuration of indentations for the 
            #boundFilt and shapeFilt to break out of the for loops doing the
            #testing and then continue to the top of the main while loop, so I
            #used the flag variable featPass instead
                
            if featPass:
                feats[featCount,0], feats[featCount,1] = r, c
                featCount+=1

                if featCount%100==0:
                    print 'found '+str(featCount)+' good features'
                    
            elif returnDiscards:
                discardFeats.append([r, c])
                
            clearCirc(array, r, c, circListClear)
            
            if time.time()-startTime>breakTime:
                print 'topNfeats breaking out at '+str(featCount)+' features'\
                +' - running time exceeded '+str(breakTime)+' secs'
                if not returnDiscards:
                    return feats[:featCount] #cut to the last appended feature
                else:
                    return feats[:featCount], np.array(discardFeats) 
                    #discardFeats was a list so is converted to array for 
                    #consistency of output
        
        if not returnDiscards:
            return feats
        else:    
            return feats, np.array(discardFeats) #discardFeats was a list
            #so is converted to array for consistency of output
            
class CovMap(Map):
    'Simple covariance map from scan object' 
       
    def build(self):
        'Constructs covariance map according to covariance equation'
        syx = self.syx()
        
        sx = np.matrix(self.scan.scanList[:self.numScans].sum(axis=0))
        sysx = np.matrix.transpose(sx) * sx
        
        self.array = np.array((syx - (sysx/self.numScans))/(self.numScans-1))
        
    def getPeak(self, r, c, pixOut=15):
        'method returns a peak from the map, given row, column (r,c) index'
        return CovPeak(self, r, c, pixOut=pixOut)
        
class PCovMap(Map):
    'Partial covariance map from scan object'
    def __init__(self, scan, pCovParams, numScans='all'):
        if numScans=='all':
            self.pCovParams = pCovParams
        else:
            self.pCovParams = pCovParams[:numScans]      
        Map.__init__(self, scan, numScans)
        
    def build(self):
        "Calculate full partial covariance map with single partial \
        covariance parameter pCovParams"            
        
        mapScanList = self.scan.scanList[:self.numScans, :]
        avYX = self.syx()/(self.numScans - 1)
        
        SxVec = np.matrix(mapScanList.sum(axis = 0)) #also made matrix type  
        #(row vector) for subsequent matrix multiplication

        avYavX = np.array(np.matrix.transpose(SxVec) * SxVec)/\
        (self.numScans * (self.numScans - 1))
        
        var = varII(self.pCovParams)
        covXIvec = np.matrix(covXI(mapScanList, self.pCovParams)) #made matrix 
        #type (row vector) for subsequent matrix multiplication
        
        self.array = (self.numScans-1)/(self.numScans-2) * (avYX - avYavX - \
        np.array(np.matrix.transpose(covXIvec) * covXIvec)/var)
    
    def getPeak(self, r, c, pixOut=15):
        'method returns a peak from the map, given row, column (r,c) index'
        return PCovPeak(self, r, c, pixOut=pixOut)
        
    def Si(self):
        'returns sum of partial covariance parameter, I, across scan'
        return self.pCovParams.sum(axis=0)
        
    def Si2(self):
        'returns sum of square of partial covariance parameter, I, across scan'
        return (self.pCovParams**2).sum(axis=0)
        
class Peak:
    'Any peak from a 2D map'
    def __init__(self, array):
        self.array=array
        
    def com(self, pixEachSide=3):
        "Returns the row and column index of the centre of mass of a square \
        on the 2D array 'array', centred on the pixel indexed by r, c and of \
        width (2 * pixEachSide + 1)"

        square = np.zeros((2*pixEachSide+1, 2*pixEachSide+1))
        
        for i in range(2*pixEachSide+1):
            for j in range(2*pixEachSide+1):
                square[i,j] = self.array[self.pixOut-pixEachSide+i, \
                self.pixOut-pixEachSide+j]    
                #It is clearer to cast the indexing of the array like this 
                #because it is consistent with how the index of the COM is 
                #returned
                
        #Having a negative value in 'square' can cause the centre of mass routine
        #to return a value outside of the boundaries of 'square'. This is
        #undesirable for the purposes of this function (a result of the way in
        #which negative values are interpreted on a CV/pCV map).
        #So we check if there are negative values in 'square', and if so we get 
        #rid of them by uniformly raising the values of 'square' so that the 
        #minimum value is not negative but zero (and all other values are 
        #positive). This allows the centre-of-mass formula to provide the required 
        #indices for our purpose here.
        squareMin = np.nanmin(square)
        if squareMin < 0:    
            squareTwo = abs(squareMin) * \
            np.ones((2*pixEachSide+1, 2*pixEachSide+1))
            square += squareTwo
            
        COMi, COMj = ndimage.measurements.center_of_mass(square)
        
        return COMi, COMj #self.r-pixEachSide+COMi, self.c-\
        #pixEachSide+COMj

    def templateNray(self, numRays,perimOffset=-0.5,integThresh=0,\
    maxRayLength='square_width',r_i='square_centre',c_i='square_centre'):
        "Return boolean template of peak, created by joining end of N rays"
        """integThresh condition is <=. r_i and c_i are row and column indices
        to cast rays from. perimOffset is perimeter offset - offset between
        vertices and the perimeter of the template outline (passed as radius 
        to Path.contains_points() method)."""
    
        array=self.array
        dim=len(array) #if dim odd, maxRayLength falls one pixel short of 
        #'bottom' and 'right' edges of array
        cent=int(np.floor((dim-1)/2)) #either central pixel if dim is odd or 
        #'top left' of central 4 pixels if dim is even
        
        if maxRayLength=='square_width':
            maxRayLength=cent
        if r_i=='square_centre':
            r_i=cent
        if c_i=='square_centre':
            c_i=cent
        
        vertices=[] #first point at end of each ray where value<=integThresh
        for theta in np.linspace(0, 2*np.pi, numRays, endpoint=False):
            #endpoint=False because ray at 2*pi is same direction as ray at 0
            r=r_i
            c=c_i
            
            incR=np.cos(theta) #increment in Row index - so first ray cast 
            #directly 'down' (when theta==0, cos(theta)=1, sin(theta)=0)
            incC=np.sin(theta) #increment in Column index
            
            for x in range(maxRayLength):
                r+=incR
                c+=incC
                if array[int(np.round(r)), int(np.round(c))]<=integThresh:
                    if (np.round(r), np.round(c)) not in vertices:
                        vertices.append((np.round(r), np.round(c)))
                    break
            else:#this is equivalent to saying the pixel the next step out 
            #would have been below the integThresh
                r+=incR
                c+=incC
                vertices.append((np.round(r), np.round(c)))
                       
        vertices=Path(vertices) #instance of matplotlib.path.Path class,
        #efficiently finds points within arbitrary polygon defined by
        #vertices
        
        points=np.zeros((dim**2, 2), dtype=int)
        points[:,0]=np.repeat(np.arange(0, dim, 1), dim)
        points[:,1]=np.tile(np.arange(0, dim, 1), dim)
        
        points=points[vertices.contains_points(points, radius=perimOffset)] 
        #only choose those points which are inside the polygon traced by the 
        #cast rays. 'radius' kwarg poorly documented but setting to -0.5 draws 
        #polygon inside vertices as required (because vertices are elements 
        #with value <= thresh).
        template=np.zeros((dim, dim), dtype=bool)
        for x in points:
            template[x[0], x[1]]=True
            
        return template
    
    def cutPeak(self, peakTemplate):
        'set any values indexed by False in peakTemplate to 0'
        self.array[peakTemplate == False] = 0
    
    def bivSplIntegrate(self, pixWidth=1):
        'integrate array using bivariate spline of default degree 3'
        mesh_array = np.arange(len(self.array)) * pixWidth #bivariate spline 
        #requires a meshgrid. Because the MS gives out arbitrary units in the 
        #text files used to make the CV maps (and this function has been the 
        #only one so far used to extract anything quantitative from the maps), 
        #the spacing of the elements (=pixWidth) in the two vectors to define 
        #this meshgrid has been unimportant (provided it is uniform) and for 
        #convenience has by default been unity. To maintain consistency for 
        #testing, pixWidth is therefore currently set to 1 when this function 
        #is called from the sampleFeats. If required, pixWidth can 
        # of course be set to sliceSize.
        spline = interpolate.RectBivariateSpline(mesh_array, mesh_array, \
        self.array)
        #make the bivariate spline, default degree is 3
        return spline.integral(np.nanmin(mesh_array), np.nanmax(mesh_array),\
        np.nanmin(mesh_array), np.nanmax(mesh_array))
        #returns the integral across this spline
        
class tempPeak(Peak):
    pass #this is so that when you resample you can e.g. integrate and find
    #the CoM if for some reason you would like to.
        
class CovPeak(Peak):
    'peak on a simple covariance map'
    def __init__(self, fromMap, r, c, pixOut=15):
        self.fromMap = fromMap
        self.r = int(r)
        self.c = int(c)
        self.pixOut = int(pixOut)
        self.build()

    def build(self):
        numScans = self.fromMap.numScans
        self.array = (self.Syx() - self.SySx()/(numScans))/(numScans - 1)
        
    def reCentre(self, maxChange=3):
        "For when you are not entirely sure where the maximum is. Max change\
        is the number of pixels you are willing to change x and y by."
        rf, cf = maxIndices(self.array\
        [self.pixOut-maxChange:self.pixOut+maxChange+1, \
        self.pixOut-maxChange:self.pixOut+maxChange+1])

        if rf == maxChange and cf == maxChange:
            print 'no shift in peak apex'
        elif cf == maxChange:
            print 'r shifted down by '+str(rf-maxChange)+' pixels'
            self.r += rf-maxChange
            self.build()
        elif rf == maxChange:
            print 'c shifted right by '+str(cf-maxChange)+' pixels'
            self.c += cf-maxChange
            self.build()
        else:
            print 'r shifted down by '+str(rf-maxChange)+\
            ' pixels and c shifted right by '+str(cf-maxChange)+' pixels'
            self.r += rf-maxChange
            self.c += cf-maxChange
            self.build()
            
    def Sy(self):
        'returns sum of intensities across all yScans'
        return self.yScans().sum(axis=0)
    
    def Sx(self):
        'returns sum of intensities across all xScans'
        return self.xScans().sum(axis=0)
    
    def Syx(self):
        'returns <YX> matrix, second term in covariance formula'
        fromIndexRow = self.r - self.pixOut
        toIndexRow = self.r + self.pixOut
        fromIndexCol = self.c - self.pixOut
        toIndexCol = self.c + self.pixOut
        return self.fromMap.syx()[fromIndexRow:toIndexRow + 1,\
        fromIndexCol:toIndexCol + 1]
        
    def SySx(self):
        'returns <Y><X> matrix, first term in covariance formula'
        return np.array(np.matrix.transpose(np.matrix(self.Sy())) * \
        np.matrix(self.Sx()))
        
    def yScans(self):
        'returns the subset of m/z bins within m/z range of y-axis of peak'
        fromIndexRow = self.r - self.pixOut
        toIndexRow = self.r + self.pixOut
        return self.fromMap.scan.scanList[:self.fromMap.numScans, \
        fromIndexRow:toIndexRow + 1]
        
    def xScans(self):
        'returns the subset of m/z bins within m/z range of x-axis of peak'
        fromIndexCol = self.c - self.pixOut
        toIndexCol = self.c + self.pixOut
        return self.fromMap.scan.scanList[:self.fromMap.numScans, \
        fromIndexCol:toIndexCol + 1]
        
class PCovPeak(CovPeak):
    
    def build(self):
        numScans = self.fromMap.numScans
        
        Si2 = self.fromMap.Si2()
        Si = self.fromMap.Si()
        S2i = Si**2

        varII = (Si2 - S2i/numScans)/\
        (numScans-1)
        
        Syx = self.Syx()      
        SySx = self.SySx()
        
        SiSx = self.SiSx()
        SySi = self.SySi()
        
        Six = self.Six()
        Syi = self.Syi()
        
        covYX = (Syx - SySx/(numScans))/(numScans - 1)
        covYI = (Syi - SySi/(numScans))/(numScans - 1)
        covIX = (Six - SiSx/(numScans))/(numScans - 1)
        
        self.array = ((numScans-1)/(numScans-2)) * (covYX - \
        np.array(np.matrix.transpose(np.matrix(covYI)) \
        * np.matrix(covIX))/varII)
        
    
    def jkResampleVar(self, cutPeak=True, template=None):
        'calculate std dev of volume of peak upon jackknife resampling'
        numScans=self.fromMap.numScans
        pCovParams=self.fromMap.pCovParams
        yScans=self.yScans()
        xScans=self.xScans()
        SyxFull=self.Syx()
        SyFull= self.Sy() 
        SxFull=self.Sx()
        SiFull=self.fromMap.Si()
        Si2Full=self.fromMap.Si2()
        SixFull=self.Six()
        SyiFull=self.Syi()

        pCovSum=0
        pCovSumSqd=0
        
        for missingScan in range(numScans):
            
            Syx = SyxFull - \
            np.array(np.matrix.transpose(np.matrix(yScans[missingScan,:])) \
            * np.matrix(xScans[missingScan,:]))
            
            Sy = SyFull - yScans[missingScan,:]
            Sx = SxFull - xScans[missingScan,:]
            
            SySx = np.matrix.transpose(np.matrix(Sy)) * \
            np.matrix(Sx)
            
            Si = SiFull - pCovParams[missingScan]
            Si2 = Si2Full - (pCovParams[missingScan])**2
        
            SiSx = Si * Sx
            SySi = Sy * Si
            
            Six = SixFull - pCovParams[missingScan] * xScans[missingScan,:]
            Syi = SyiFull - yScans[missingScan,:] * pCovParams[missingScan]
            
            #Number of scans for each partial covariance square on the resample
            # = numScans - 1
            
            covYX = (Syx - SySx/(numScans - 1))/(numScans - 2)
            covYI = (Syi - SySi/(numScans - 1))/(numScans - 2)
            covIX = (Six - SiSx/(numScans - 1))/(numScans - 2)
            varII = (Si2 - Si**2/(numScans - 1))/(numScans - 2)
            
            pCovSquare = Peak(((numScans-2)/(numScans-3)) * \
            (covYX - np.array(np.matrix.transpose(np.matrix(covYI)) * \
            np.matrix(covIX))/varII)) #factors of numScans-2 could cancel out
            
            if cutPeak:
                pCovSquare.cutPeak(template)
            
            vol=pCovSquare.bivSplIntegrate()
                         
            pCovSum += vol
            pCovSumSqd += vol**2
            
        return (pCovSumSqd - (pCovSum**2)/(numScans))/numScans #this should 
        #include Bessel's correction, but has not historically. Because we are 
        #currently unconcerned with absolute value of stdDev (and can always 
        #retrospectively adjust) but do want to compare with previous results, 
        #omit for now.
        
    def SiSx(self):
        'return <I><X>'
        return self.fromMap.Si() * self.Sx()
        
    def SySi(self):
        'return <Y><I>'
        return self.Sy() * self.fromMap.Si()
        
    def Syi(self):   
        'return <YI>'
        SyiPeak = np.zeros(self.pixOut*2+1)
        yScans = self.yScans()
        for scanIndex in range(self.fromMap.numScans):
            SyiPeak += yScans[scanIndex,:] * self.fromMap.pCovParams[scanIndex]
        return SyiPeak
        
    def Six(self):   
        'return <IX>'
        SixPeak = np.zeros(self.pixOut*2+1)
        xScans = self.xScans()
        for scanIndex in range(self.fromMap.numScans):
            SixPeak += xScans[scanIndex,:] * self.fromMap.pCovParams[scanIndex]      
        return SixPeak
    
#%% 
"""this calls the above classes to:
   - read in a text file for a scan as a numpy array
   - calculate the pC-2DMS map for the scan
   - identify features on the pC-2DMS map
   - calcualte the pC-2DMS correlation score for these features
   - save the results in a numpy file"""
   
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--textfile","-t", type=str, required=True)
    parser.add_argument("--readto","-r", type=str, required=True)
    
    args=parser.parse_args()
    
    #%%    
    args.textfile.replace("\\", "/") #in case there are backslashes in 
    args.readto.replace("\\", "/")   #either of these
    
    #%% Now run the analysis
    assert os.path.isdir(args.readto) #check that the folder we are saving to
    #exists
    readTextFile(args.textfile, args.readto)
    scan1=Scan(args.readto)
    map1=PCovMap(scan1, scan1.tic())
    np.save(args.readto+'/PCV_TIC_map.npy', map1.array)
    
    feats=map1.analyse(3000) #3000 is the number of features to return
    
    np.save(args.readto+'/3000feats_jackknifeResampled.npy', feats)