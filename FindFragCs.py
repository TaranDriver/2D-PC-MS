# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 17:30:43 2018

@author: Taran Driver

Extract charge states from pC-2DMS map.

Mass conservation lines are fundamentally different to neutral loss lines
and this is written in although Hough transform should be able to find all
NL lines as well
"""

import numpy as np
import sys
sys.path.append(['D:/Vitali/TaranCode'])
import HoughTransform as ht
import pC2DMStools as p2
from matplotlib import pyplot as plt


pointsIn=r"D:\Vitali\Data\Cytochrome_c_12+\3000feats_jackknifeResampled_defaultParams_100rays.npy" #points
parMz=1031.09 #from Sigma calculation for average mass
parCs=12
ionName='Cytochrome_c_12+'

#%% set parameters
savePath=r'D:\xxxxTaran\Data\TopDown\Cytochrome_c\12+\AGC300\20180417_sampled_4'
save=False

houghOnly=True
topN=100 #150  
houghThresh=7 #how many points does m/c line have to pass through to be called
#a line?
clustertol=3.0 #for clustering of lines found by Hough, distance separating 
#two distinct lines

distThresh=1.5 #for the Hough transform, distance between point and line to
#decide if point is on a line
parMzThresh=1.5
acThresh=7. #threshold for determining whether features are autocorrelation
#features to be removed
onLineThresh=distThresh #threshold for determining whether a correlation falls
#on a particular line or not

protectNLs=False #protect lines corresponding to an NL of up to protectNLsUpTo
protectNLsUpTo=28 #from the parent ion in the same way that lines correponding
#to the intact parent ion are protected - if they have a duplicate with
#different charge states they are still kept and not removed

#%% initial warning
if (not houghOnly) and protectNLs:
    raise ValueError, 'no implementation for ~houghOnly and protectNLs!'

#%% Preliminary computations
points=np.load(pointsIn)
points=p2.sortList(p2.removeACFeats(points, acThresh=acThresh))[:topN, :2]
points=np.concatenate((points, np.flip(points, axis=1))) #to get full map,
#one point on each side of the a/c line
parMass=parMz*float(parCs)

#%% if choosing to set the mass conservation lines with parent mass
# and parent charge state (houghOnly=False), do that now
if not houghOnly:
    linesPrimary=[]
    for csx0, csy0 in ht.chargeStates(parCs, onlySteep=True):
        theta=ht.getTheta(csx0, csy0)
        rho=ht.getRho(theta, parMass, csx0, csy0)
        linesPrimary.append([theta, rho, 0, csx0, csy0, parMass]) #initialise
        #number of points on line at zero, this will be changed later
    linesPrimary=np.array(linesPrimary)

#%% Now use Hough transform to find m/c lines at all possible gradients
# given the parent charge state
allPossCs=np.concatenate([ht.chargeStates(cs) for cs in \
                          np.arange(2, parCs+1)],axis=0) #ht.ChargeStates
#with onlySteep=True (this is default, as here) returns array of (csx, csy)'s
#with csx>=csy
allPossThetas=np.array([ht.getTheta(csx, csy) for csx,csy in allPossCs])
allPossCombs=np.concatenate((allPossCs, allPossThetas[:,np.newaxis]), axis=1)

linesFound=[]
for csx, csy, theta in allPossCombs:
    #csx still biggest
    if theta==np.pi/4.:
    #if charge state separation symmetric, same points will lie on the m/c 
    #line each side of the a/c line
        houghThresh_i=houghThresh*2
    else:
        houghThresh_i=houghThresh

    linesFound_i=ht.HoughLines(points, accThresh=houghThresh_i, \
                             thetaMin=theta, thetaMax=theta,\
                             distThresh=distThresh, plotTransform=False, 
                             rhoRes=0.1)
    
    if len(linesFound_i)!=0:
        linesFound_i=np.concatenate((linesFound_i, np.array([[csx, csy,\
        ht.getParMass(theta, x[1], csx, csy)] for x in linesFound_i])), axis=1)
    
        linesFound_i=ht.clusterLines(linesFound_i, clustertol=clustertol)#each 
        #cluster returned by clusterLines is ordered by the clusteron parameter
        #ht.clusterLines clusters on parent mass by default, so clustertol
        #is parent mass as well
#        if (csx, csy)==(4., 2.):
#            print 'hit it'
#            toSee0=linesFound_i[:]
        #Now for each cluster only take the lines that go through the max
        #number of points        
        linesFound_i=[np.array(x) for x in linesFound_i]
        linesFound_i=[x[x[:,2]==np.nanmax(x[:,2])] for x in linesFound_i]

        linesFound_i=[x[len(x)/2] for x in linesFound_i] #take middle of 
        #cluster or highest of two middle parent masses (from above, these
        #clusters are ordered on parent mass) if there are even number in 
        #cluster 
        linesFound.append(linesFound_i) 

linesFound=np.concatenate(linesFound)
toSee0a=np.copy(linesFound)
#column indices are (theta, rho, no. points on line, cs1, cs2, parent mass)

#%% if houghOnly, now take all the ones that are mass conservation lines of
# the full parent mass, making them immune to getting removed as
# duplicates. Leave them in linesFound for the minute, so that if they
# do have duplicates these are removed in the duplicate removal step
if houghOnly:
    if protectNLs:
        intactMcMask=np.logical_and(\
        (linesFound[:,5]-parMass)<=(parMzThresh*parCs),\
        ((linesFound[:,5]-parMass)>=-(parMzThresh*parCs+protectNLsUpTo)))
        linesPrimary=linesFound[intactMcMask]
    else:
        intactMcMask=abs(linesFound[:,5]-parMass)<=(parMzThresh*parCs)
        linesPrimary=linesFound[intactMcMask]

toSee1=np.copy(linesFound)
#%%
#Now remove any mass conservation violating lines
mcViolMask=linesFound[:,5]-parMass>(parMzThresh*parCs) #mass conservation
#violation
linesFound=linesFound[~mcViolMask]
toSee2=np.copy(linesFound)

#%% Now take out any lines which have made it past the m/c violation filter
# but aren't unique in rho and theta, i.e. could correspond to 2 or more 
# different charge state combinations
if len(linesFound)!=0:
    _,indices,counts=np.unique(linesFound[:,:2], return_index=True, \
                               return_counts=True, axis=0)
    indicesToTake=indices[counts==1]
    linesFound=linesFound[indicesToTake]
toSee3=np.copy(linesFound)

#%% Now of the surviving lines, take out those which correspond to one of the
# already-set mass conservation lines. If there was one of these in the found
# lines which had two possible charge states (and so two possible parent 
# masses) related to it, it would have already been taken out in the above
# step
# now recalculate the intact masks incase some have been removed as duplicates
# or mass conservation violating
if protectNLs:
    intactMcMask=np.logical_and(\
    (linesFound[:,5]-parMass)<=(parMzThresh*parCs),\
    ((linesFound[:,5]-parMass)>=-(parMzThresh*parCs+protectNLsUpTo)))
else:
    intactMcMask=abs(linesFound[:,5]-parMass)<=(parMzThresh*parCs)

linesFound=linesFound[~intactMcMask]
toSee4=np.copy(linesFound)

#%% Now fill the mcLines list with each line as a dictionary
mcLines=[]
# fill with the m/c lines corresponding to full parent ion mass (from list 
# called linesPrimary but that holds lines corresponding to full parent mass even
# if onlyHough=True
for line in linesPrimary:
    pointsOnLine=ht.onLine(points, (line[1],line[0]), \
    distThresh=onLineThresh)
    
    mcLines.append({'charges':(line[3], line[4]), \
                             #(csx, csy)
    'mass':line[5], 'rho':line[1], 'theta':line[0], \
    'points': pointsOnLine[:len(pointsOnLine)/2] if line[0]==np.pi/4 else \
    pointsOnLine, 'from':'predefined' if not houghOnly else 'found by Hough, intact'})
    
## fill with the other m/c lines
#for line in linesFound:
#    pointsOnLine=ht.onLine(points, (line[1],line[0]), \
#    distThresh=onLineThresh)
#    
#    mcLines.append({'charges':(line[3], line[4]), \
#                             #(csx, csy)
#    'mass':line[5], 'rho':line[1], 'theta':line[0], \
#    'points': pointsOnLine[:len(pointsOnLine)/2] if line[0]==np.pi/4 else \
#    pointsOnLine, 'from':'found by Hough'})

#%% Now add charge state to experimental data
# if no unique charge state identified, charge state is given as zero
expCorrs=np.concatenate((p2.sortList(p2.removeACFeats(np.load(pointsIn),\
                        acThresh=acThresh))[:topN], np.zeros((topN, 2))), \
                        axis=1)

ions=np.zeros((1, 4))
for x in mcLines:
    csx=np.ones(len(x['points']))*x['charges'][0]
    csy=np.ones(len(x['points']))*x['charges'][1]
    # one way round
    ions=np.append(ions, \
    np.stack((x['points'][:,0], x['points'][:,1], csx, csy), axis=1), \
    axis=0)
    # the other way round
    ions=np.append(ions, \
    np.stack((x['points'][:,1], x['points'][:,0], csy, csx), axis=1), \
    axis=0)
ions=ions[1:]

for expCorr in expCorrs:
    where=np.where(expCorr[:2]==ions[:,:2])
    if np.all(where[1]==np.array([0,1])) and where[0][0]==where[0][1]:
        #so if one and only one charge state has been found for the pair
        expCorr[4:6]=ions[where[0][0]][2:4] 

#%% this cell only used now for determining the number of repeats. What is
# saved is now 'expCorrs' above
def ionsWCs(mcLines):
    ions=np.zeros((1, 4))
    for x in mcLines:
        csx=np.ones(len(x['points']))*x['charges'][0]
        csy=np.ones(len(x['points']))*x['charges'][1]
        ions=np.append(ions, \
        np.stack((x['points'][:,0], csx, x['points'][:,1], csy), axis=1), \
        axis=0)
    return ions[1:]

returned=ionsWCs(mcLines)
numRepeats=len(returned)-len(np.unique(np.stack((returned[:,0], \
              returned[:,2]), axis=1), axis=0))
print str(numRepeats)+' repeats'
    
#%% now plotting
cInd=0
xPoints=np.linspace(np.nanmin(points), np.nanmax(points), 100)
colours=['b', 'g', 'r', 'm', 'c', 'y']*30
plot0=plt.figure(figsize=(10,10))
ax0=plot0.add_subplot(111)
ax0.scatter(points[:,0], points[:,1], color='k')
ax0.set_xlim(np.nanmin(points)-10, np.nanmax(points)+10)
ax0.set_ylim(np.nanmin(points)-10, np.nanmax(points)+10)

for x in mcLines:
    ax0.scatter(x['points'][:,0], x['points'][:,1], s=60, \
                c=colours[cInd], marker='*')
    ax0.plot(xPoints, xPoints*-(np.cos(x['theta'])/\
             np.sin(x['theta']))+\
             x['rho']/np.sin(x['theta']), 
             label='x='+str(int(x['charges'][0]))+'+/y='+\
                   str(int(x['charges'][1]))+\
                   '+, par mass='+str(np.round(x['mass'], 2))+',\n'+\
                   x['from']+' ['+str(len(x['points']))+' points] ', \
                   c=colours[cInd])
    thetaForFlip=np.pi/2.-x['theta']
    ax0.plot(xPoints, xPoints*-(np.cos(thetaForFlip)/\
         np.sin(thetaForFlip))+\
         x['rho']/np.sin(thetaForFlip), c=colours[cInd])
    cInd+=1

plt.legend()
plt.title(' '.join(ionName.split('_')))
plot0.show()

#%% Now save corrs and scatter with lines
if save:
    plt.savefig(savePath+'/'+ionName+('_HoughOnly' if houghOnly else '')+\
                '_'+str(numRepeats)+'repeats.png', dpi=300)
    np.save(savePath+'/'+ionName+('_HoughOnly' if houghOnly else '')+\
            '_expCorrswithCs', expCorrs)
