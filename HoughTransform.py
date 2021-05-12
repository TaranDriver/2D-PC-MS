# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:05:59 2018

@author: Taran Driver

Code for development of Hough transform for identifying mass conservation
lines on the pC-2DMS map
"""
import numpy as np
import sys
sys.path.append('D:/Taran/Code')
import pC2DMSUtils as p2
from matplotlib import pyplot as plt
from scipy import optimize

#%% clustering for Hough lines
def clusterLines(lines, clusteron=lambda lines: lines[:,5], clustertol=2.*3):
    #condition for clustertol is less than. In default case (clusteron is
    #parent mass)
    clusteron=clusteron(lines) #redefined from lambda function to list of
    #variables
    order=np.argsort(clusteron)

    cluster=0 #initialisation
    x_i=clusteron[order[0]] #initialisation
    out=[[lines[order[0]]]] #initialisation, list so you can have flexible 
    #number of 'columns' in each 'row'
    for i in range(1, len(clusteron)):
        x_f=clusteron[order[i]]
        if abs(x_f-x_i)<clustertol: #lines are already sorted above but use of
        #abs() makes more robust
            out[cluster].append(lines[order[i]])
        else:
            out.append([lines[order[i]]])
            cluster+=1     
        x_i=x_f
    return out

def clusterLines2(lines, clustertolmz=1.5):
    #condition for clustertol is less than. In default case (clusteron is
    #parent mass). 
    #this has charge state dependent tolerance and is specific to Hough lines
    #defined in this module. clustertolmz to show fact it is explicitly m/z
    #(so changes with charge state)
    clusteron=lines[:,5]
    order=np.argsort(clusteron)

    cluster=0 #initialisation
    x_i=clusteron[order[0]] #initialisation
    fullcs_i=lines[order[0]][3]+lines[order[0]][4] #initialisation
    out=[[lines[order[0]]]] #initialisation, list so you can have flexible 
    #number of 'columns' in each 'row'
    for i in range(1, len(clusteron)):
        x_f=clusteron[order[i]]
        fullcs_f=lines[order[0]][3]+lines[order[0]][4] #initialisation
        if abs(x_f-x_i)<clustertolmz*(fullcs_i+fullcs_f)/2.: #lines are 
        #already sorted above but use of abs() makes more robust
            out[cluster].append(lines[order[i]])
        else:
            out.append([lines[order[i]]])
            cluster+=1     
        x_i=x_f
    return out

def HoughLines(points, accThresh, rhoRes=0.5, thetaMin=0., thetaMax=np.pi/2.,\
               thetaRes=0.1*np.pi/180., distThresh=3., plotTransform=False):

    # define rhos    
    rhoMax=np.sqrt(2*((np.nanmax(points))**2))
    rhoMin=np.floor(np.nanmin(points))
    rhos=np.arange(rhoMin, rhoMax+rhoRes, rhoRes)
    
    # now define thetas
    if thetaMin==thetaMax:
        thetas=np.array([thetaMin])
    else:
        thetas=np.arange(thetaMin, thetaMax+thetaRes, thetaRes)
    
    acc=np.zeros((len(thetas), len(rhos))) #initialisation of accumulator
    
    for x, y in points:
        dists=abs(x*np.cos(thetas)[:,np.newaxis] + \
                  y*np.sin(thetas)[:,np.newaxis]-\
                  rhos[np.newaxis,:]) #smallest distance between point and 
                  #Hessian normal line
        #The above assumes the correlation islands should be treated as
        #circles, in reality they should probably be squares
        wheres=np.where(dists<distThresh)
        acc[wheres]+=1
        
    if plotTransform:
        fig0=plt.figure(figsize=(8, 8))
        ax0=fig0.add_subplot(111)
        if len(thetas)==1:
            plot0=plt.plot(rhos, acc.T, color='k')
        else:
            plot0=ax0.pcolorfast(rhos, (thetas/np.pi)*180.0, acc)
#            plot0=ax0.pcolormesh(thetas, rhos, acc)
            plt.colorbar(plot0)
        plt.show()
    
    overThresh=np.where(acc>=accThresh)
    lines=np.vstack(overThresh+(acc[overThresh],)).T #cols are 
    #(row in parameter space, column in parameter space, value of accumulator)
    # this stacks a length-3 tuple of arrays
    lines=lines[np.argsort(lines[:,2])[::-1]] #sort lines on value of
    #accumulator
    
    # now change cols 0 and 1 in lines from indices to the values indexed by
    # those indices
    lines[:,0]=thetas[lines[:,0].astype(int)]
    lines[:,1]=rhos[lines[:,1].astype(int)]
    
    return lines

def lstSqs(funcTofit, xs, ys, p_i):

    errFunc = lambda p, x, y: abs(funcTofit(p, x) - y) #define error function
    p_f = optimize.leastsq(errFunc, p_i[:], args=(xs, ys)) #either a result 
    #or the last value attempted (if call was unsucessful)
    return p_f

def splitLines(lines, diffRho, diffTheta):

    if len(lines)>1:
        uniqLines=lines[0][np.newaxis,:]
        for line in lines[1:]:
            line=line[np.newaxis,:]
            test=abs(line[:,:2]-uniqLines[:,:2])
            if np.any((test[:,0]<diffTheta)*(test[:,1]<diffRho)):
                pass
            else:
                uniqLines=np.append(uniqLines, line, axis=0)
        return uniqLines
    else:
        return lines

def onLine(points, linePars, distThresh=3.):
    "linePars=(rho, theta). Does line pass through circles with centre at \
    points and radius distThres"
    rho,theta=linePars
    mask=abs(points[:,0]*np.cos(theta)+points[:,1]*np.sin(theta)-rho)\
         <distThresh
    return points[mask]

def getTheta(csx, csy):
    grad=-(csx/float(csy))
    return (np.pi/2.)-np.arctan(-grad)

def getRho(theta, parMass, csx, csy):
    return parMass/(csx*np.cos(theta)+csy*np.sin(theta))

def getParMass(theta, rho, csx, csy):
    return rho*(np.cos(theta)*csx+np.sin(theta)*csy)

def radToDeg(rad):
    return 180.*rad/np.pi

def chargeStates(parCs, onlySteep=True):
    "when onlySteep gives only csxs higher or equal to csys -> grad >= -1"
    csxs=np.arange(1,parCs)
    csys=np.arange(1,parCs)[::-1]
    if onlySteep:
        return np.stack((csxs, csys), axis=0).T[-(parCs/2):]
    else:
        return np.stack((csxs, csys), axis=0).T

#%% THESE POTENTIALLY REDUNDANT
def cThetaToRho(c, theta):
    return c*np.cos((np.pi/2.)-theta)  
    
def gradToTheta(grad):
    "grad of line to corresponding theta for Hough transform, simple trig. \
    np.inf returns np.pi, -np.inf returns 0.0"
    return (np.pi/2.)-np.arctan(-grad) #in radians

def grads(chargeTot, onlySteep=True):
    "calculates gradients of m/c lines for total charge chargeTot related \
    charge states"
    assert type(chargeTot)==int
    csx=np.arange(1,chargeTot)
    csy=np.arange(1,chargeTot)[::-1]
    grads=-(csx/csy.astype(float))
    stacked=np.vstack((csx, csy, grads)).T
    if onlySteep:
        return stacked[-(chargeTot/2):] # only return gradients at -1 or
    else:                                # or steeper
        return stacked

#%% needs testing
#def hn2si(rho, theta): #slope-intercept gets tricky with horizontal
#    #and vertical lines (i.e. neutral loss lines)
#    "Hesse normal form to slope intercept form"
#    m=-(1./np.tan(theta))
#    c=rho*1./np.sin(theta)
#    return m, c

#%% THIS MAY NOT RUN PROPERLY, FUNCTIONS IT CALLS HAVE BEEN CHANGED SINCE LAST
#TIME IT WAS RUN
#points=np.load(r"D:\Taran\Data\TopDown\Ubiquitin\6+\20171211\Turbo_binned_4\3000feats_jackknifeResampled_defaultParams_100rays.npy") #points
##points=np.load(r"D:\Taran\Data\ME4_2+\20160503\Turbo\NumScansAnalysis_20170206\3000feats_10000scans_jackknifeResampled_20170206.npy")
##points=np.load(r"D:\Taran\Data\ME16_3+\20160428\Turbo\NumScansAnalysis_20170920\3000feats_10000scans_jackknifeResampled_20170920.npy")
##points=np.load(r"D:\Taran\Data\TopDown\UN19_5+\20171122\HighMassRange\NCE70_sampled_4\3000feats_jackknifeResampled_defaultParams_100rays.npy")
#
#points=p2.sortList(p2.removeACFeats(points))[:100, :2]
#points=np.concatenate((points, np.flip(points, axis=1)))

###############################################################################
##%% this if you only want to worry about lines at known angles
#uniqLines=[]
#chargeTot=6
#
#for theta in gradToTheta(grads(chargeTot, onlySteep=False)):
#    if theta==np.pi/4.: # here, the same points will lie on the m/c line each 
#        linesToAdd=HoughLines(points, accThresh=4, thetaMin=theta, \
#                              thetaMax=theta) #side of the a/c line
#        print 'line at 45 deg!' 
#    linesToAdd=HoughLines(points, accThresh=8, thetaMin=theta, thetaMax=theta)
#    linesToAdd=splitLines(linesToAdd, 57, 0.01)
#    uniqLines.append(linesToAdd)
#uniqLines=np.concatenate(uniqLines)
###############################################################################
#%%
###############################################################################
##%% this if you want to do the full Hough transform over all angles          
#chargeTot=6
#lines=HoughLines(points, accThresh=8, plotTransform=True)
#diffTheta=np.nanmin(abs(np.diff(gradToTheta(grads(chargeTot)[:,2])))) #minimum 
##angular difference between mass conservation lines for two different charge 
##state partitions
#diffTheta-=-2*np.pi/180. # -2 degrees to allow for small deviation in tilt of 
#                         #lines
#uniqLines=splitLines(lines, 57, diffTheta) #57 is Mr of glycine
###############################################################################

#%%
## now plot defined lines on scatter plot
#fig1=plt.figure(figsize=(8, 8))
#ax1=fig1.add_subplot(111)
#plot1=ax1.scatter(points[:,0], points[:,1])
#
#for line in uniqLines:
#    xVals=np.linspace(np.nanmin(points), np.nanmax(points))
#    ax1.plot(xVals, xVals*-(np.cos(line[0])/np.sin(line[0]))+\
#                 line[1]/np.sin(line[0]))
#plt.xlim(np.nanmin(points)-10, np.nanmax(points)+10)
#plt.ylim(np.nanmin(points)-10, np.nanmax(points)+10)
#
## Now check the points found to be lying on each line are correct
#colours=['g', 'r', 'c', 'm', 'y', 'k']
#cInd=0
#for line2 in uniqLines:
#    pointsOnLine=on_line(points, (line2[1], line2[0]))
#    assert len(pointsOnLine)==line2[2] #same condition used for HoughLines
#    #as for 'pointsOnLine' function
#    ax1.scatter(pointsOnLine[:,0], pointsOnLine[:,1], s=120, c=colours[cInd], marker='*')
#    cInd+=1
#    
##ax1.scatter(1.501e+03,1.063e+03, s=200, c=colours[cInd], marker='*')
##ax1.scatter(1.501e+03, 1.0456e+03, s=200, c=colours[cInd], marker='*')
#
#plt.show()

#%% This uses least squares fit to properly centre the mass conservation lines
# Maybe not sensible because points accidentally labelled as m/c points could
# drag value out. For the minute this cell only prints out, nothing else.
#for line3 in uniqLines:
#    
#    theta=line3[0]
#    rhoInit=line3[1]
#    
#    pointsOnLine=on_line(points, (line3[1], line3[0]))
#    
#    funcTofit=lambda rho, x: -(np.cos(theta)/np.sin(theta))*x + \
#                             rho/np.sin(theta)
#    
#    rhoFinal=lstSqs(funcTofit, pointsOnLine[:,0], pointsOnLine[:,1], \
#                      np.array([rhoInit]))[0][0]
#    print 'from ' +str(rhoInit) + ' to ' + str(np.round(rhoFinal,2))
