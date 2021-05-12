# -*- coding: utf-8 -*-
"""
@author: Taran Driver

This code for matching experimental correlations with database peptides.

Iterates over all database peptides, calculating the number of experimental
correlations that match with each type of expected correlation from the
in silico fragmentation of the database peptide sequence.

For each database sequence, increments the score for each of the three 
correlation types (b&y, b&y then small molecule neutral loss, internal b-type 
& terminal) according to the normalised pC-2DMS correlation score of each 
matching experimental correlation.

Final pC-2DMS correlation score calculated for each database peptide 
calculated as weighted sum of these scores for the three separate 
correlation types.
"""
from __future__ import division #IMPORTANT! __future__ imports have to come 1st
import numpy as np
import pC2DMStools as pT
import dbMSE as dbMSE
import time
import os
import argparse

#%%
def formatMatch(matchArray, matchType, testPep, cs0, cs1, nl0=None,\
                nl1=None):
    "return string describing a match"            
    ionMarker=np.where(matchArray)[0][0]
    
    if matchType=='b_y':
        ionIndices=testPep.b_yAnnot(ionMarker)
        matchString0='b'+str(ionIndices[0])
        matchString1='y'+str(ionIndices[1])
        
    elif matchType=='b_int':
        ionIndices=testPep.b_intAnnot(ionMarker)
        matchString0='b'+str(ionIndices[0])
        matchString1='bi{'+str(ionIndices[1])+'-'+str(ionIndices[2])+'}'
        
    elif matchType=='int_y':
        ionIndices=testPep.int_yAnnot(ionMarker)
        matchString0='bi{'+str(ionIndices[0])+'-'+str(ionIndices[1])+'}'
        matchString1='y'+str(ionIndices[2])
        
    if nl0!=None:
        matchString0='['+matchString0+'-'+nl0+']'
        
    if nl1!=None:
        matchString1='['+matchString1+'-'+nl1+']'
        
    chargeStr0='' if cs0==1 else str(cs0)
    chargeStr1='' if cs1==1 else str(cs1)
    
    return matchString0+'('+chargeStr0+'+) & '+matchString1+'('+chargeStr1+'+)'

def totScore(scoreLs, (byWt, nlWt, intWt)): #weights for b/y, nl, internal
    'calculate total pC-2DMS correlation score given certain weights for each\
    correlation type. Default currently 0.8, 0, 1'
    scoreLs=np.asarray(scoreLs)
    return byWt*scoreLs[:,0]+nlWt*scoreLs[:,1]+intWt*scoreLs[:,2]
    
#%% SET PARAMETERS
parser=argparse.ArgumentParser()

parser.add_argument("--expCorrsIn","-e", type=str, required=True) #pathname for
#experimental correlations
parser.add_argument("--parCs","-z", type=int, required=True) #parent ion
#charge state
parser.add_argument("--parSeqs","-p", type=str, required=True) #pathname for
#where parent sequences are stored in a text file
parser.add_argument("--saveTo", "-s", type=str, required=True) #pathname to
#save results to

parser.add_argument("--intWOCL", type=bool, required=False, default=False)
#include internal ions without charged loss?
parser.add_argument("--acThresh", type=float, required=False, default=7.)
#threshold for determination of autocorrelation peaks
parser.add_argument("--takeTop", type=int, required=False, default=50)
#take the top N pC-2DMS correlation score-ordered correlations
parser.add_argument("--fragIonTol", type=float, required=False, default=0.8)
#tolerance for matching of experimental correlations with correlations
#generated from database sequences
args=parser.parse_args()

#%% SET SEQS: read sequences from a text file where each line holds a 
#new sequence

seqLs=[]
with open(args.parSeqs) as f:
    for line in f:
        seqLs.append(line.rstrip()) #rstrip removes whitespace characters,
        #i.e. the '\n' here
f.close()
seqLs=list(set(seqLs)) #make sure there are no duplicates in there

seqLs=[seq for seq in seqLs if not ('X' in seq or 'B' in seq or 'Z' in seq or\
                                    'U' in seq or 'O' in seq)]
#filters out X, B, Z, U and O

#%% Define how many analysed database sequences you want to print progress
# after
numSeqs=len(seqLs)

if numSeqs>50000:
    checkAt=int(numSeqs/10000)
elif numSeqs>50000:
    checkAt=int(numSeqs/1000)
elif numSeqs>5000:
    checkAt=int(numSeqs/100)
else:
    checkAt=int(numSeqs/10)

#%% checks on the directory the results are to be saved to
if os.listdir(args.saveTo)!=[]:
    print '\n********WARNING - SPECIFIED saveTo ALREADY HAS SOME FILES/FOLDERS IN!********'
    
checkSavePath=raw_input('have you checked the save path is what you want?: '\
                    +args.saveTo+'? - y/n\n')
if checkSavePath!='y':
    raise ValueError, 'ensure save path is correct'
    
seqsInfo=raw_input('provide information about the sequence set tested:\n')
    
assert os.path.isdir(args.saveTo)

#%%
#Load experimental correlations
expCorrs=np.load(args.expCorrsIn)
    
expCorrs=pT.sortList(pT.removeACFeats(expCorrs, args.acThresh))

if args.takeTop!='all':
    expCorrs=pT.sortList(pT.removeACFeats(expCorrs, args.acThresh))\
    [:args.takeTop]
    
expCorrs[:,3]=expCorrs[:,3]/(expCorrs[:,3].sum()) #normalise so that
#significances add up to 1 for all the features

#%% initialise containers holding results
seqLsScores=[[0, 0, 0] for x in seqLs] #b/y scores, NL scores, internal scores
seqLsCounts=[[0, 0, 0] for x in seqLs] #b/y scores, NL scores, internal scores
matchList=[[[],[],[]] for x in seqLs] #list of details for each correlation
#match for each database peptide
extraMatchList=[[[],[],[]] for x in seqLs] #as above but for further matches

#%%
time0=time.time()

#%%
for i in range(numSeqs): #iterate over each database sequence being tested
    seq=seqLs[i]
    
    try:
        if i%checkAt==0:
            timeNow=time.time()
            print 'checking sequence '+seq+' ('+str(i)+'/'+str(numSeqs)+' - '\
            +str(round(i/(numSeqs/100.), 2))+'%), estimated time to go = '+\
            str(round(((timeNow-time0)/float(i))*(numSeqs-i)*(1/60), 2))+\
            ' minutes'
    except:
        pass #i=0 case
    
    testPep_=dbMSE.testPep(np.array(dbMSE.resMass(seq)), args.parCs) #this 
    #step essentially performs the in silico fragmentation by initialising
    #a dbMSE.testPep() object
   
    for expCorr in expCorrs: #for each sequence, iterate over all the
        #experimental correlations to find which match with the theoretical
        #correlations from the database sequence
        
        #here you can have different correlations and order them to be in 
        #b/int order, etc. Using the sorting in the old orderKnown=False kwarg
        #to testPep.xxxmatches() destroys the info of what exactly is matching
        matched=False
        #%%
        mz0, mz1= expCorr[0], expCorr[1]
        
        csIndex=0 #initialised at 0, first used at -1
        for sumCharge in reversed(np.arange(2, args.parCs+1)):    
            for csIon0 in reversed(np.arange(1, sumCharge)): #cS0 goes up to 
            #1 less than sumCharge
                csIndex-=1 #index counting back in e.g. testPep_.b_yMzs row
                #indices. This prioritises no charge loss
                csIon1=sumCharge-csIon0 #charge state of 'ion' 0 [b in b/y, \
                #b in b/int, int in int/y] is cs0, and 'ion' 1 is csIon1  
                
                #below a length-2 tuple is made so to allow for each 
                #experimental ion to be one of the pair    

                #define the fundamental m/z differences between this
                #experimental correlation and the expected correlations of 
                #the database peptide                                               
                bDiffs=((mz0-testPep_.b_yMzs[csIndex,:,0]), \
                       (mz1-testPep_.b_yMzs[csIndex,:,0]))        
                yDiffs=((mz0-testPep_.b_yMzs[csIndex,:,1]),\
                       (mz1-testPep_.b_yMzs[csIndex,:,1]))
                
                #%% HERE, SPECIFY WHICH TYPES OF IONs/NEUTRAL LOSSES TO TAKE INTO ACCOUNT
                #(diff_array1, diff_array2, (correlation type string, correlation type index), \
                #(neutral loss 1 mass change, neutral loss 1 string), (neutral loss 2 mass change, neutral loss 2 string))
                #order in testCorrs list defines order of precedence for matching.
                testCorrs=[(bDiffs, yDiffs, ('b_y', 0), (0, None), (0, None)),\
                   
                   (bDiffs, yDiffs, ('b_y', 1), (18.01056, 'H2O'), (0, None)),\
                   (bDiffs, yDiffs, ('b_y', 1), (0, None), (18.01056, 'H2O')),\
                   
                   (bDiffs, yDiffs, ('b_y', 1), (17.02653, 'NH3'), (0, None)),\
                   (bDiffs, yDiffs, ('b_y', 1), (0, None), (17.02653, 'NH3')),\
                   
                   (bDiffs, yDiffs, ('b_y', 1), (18.01056, 'H2O'), (18.01056, 'H2O')),\
                   (bDiffs, yDiffs, ('b_y', 1), (17.02653, 'NH3'), (17.02653, 'NH3')),\
                   
                   (bDiffs, yDiffs, ('b_y', 1), (18.01056, 'H2O'), (17.02653, 'NH3')),\
                   (bDiffs, yDiffs, ('b_y', 1), (17.02653, 'NH3'), (18.01056, 'H2O')),\
                   
                   #Now with a-ions from the b-ion
                   (bDiffs, yDiffs, ('b_y', 1), (27.99492, 'CO'), (0, None)),\
                   
                   (bDiffs, yDiffs, ('b_y', 1), (18.01056+27.99492, 'CO-H2O'), (0, None)),\
                   (bDiffs, yDiffs, ('b_y', 1), (27.99492, 'CO'), (18.01056, 'H2O')),\
                   
                   (bDiffs, yDiffs, ('b_y', 1), (17.02653+27.99492, 'CO-NH3'), (0, None)),\
                   (bDiffs, yDiffs, ('b_y', 1), (27.99492, 'CO'), (17.02653, 'NH3')),\
                   
                   (bDiffs, yDiffs, ('b_y', 1), (18.01056+27.99492, 'CO-H2O'), (18.01056, 'H2O')),\
                   (bDiffs, yDiffs, ('b_y', 1), (17.02653+27.99492, 'CO-NH3'), (17.02653, 'NH3')),\
                   
                   (bDiffs, yDiffs, ('b_y', 1), (18.01056+27.99492, 'CO-H2O'), (17.02653, 'NH3')),\
                   (bDiffs, yDiffs, ('b_y', 1), (17.02653+27.99492, 'CO-NH3'), (18.01056, 'H2O'))]
                
                if args.parCs==2 or sumCharge<args.parCs or args.intWOCL: #for anything with
                #charge state 3+ or higher, eliminates (e.g. for 3+ ions)
                #terminal/internal with 2+/1+ or 1+/2+, neither of which 
                #we observe at our experimental conditions
                
                    #define further fundamental m/z differences between this
                    #experimental correlation and the expected correlations of 
                    #the database peptide             
                    b_bIntDiffs=((mz0-testPep_.b_intMzs[csIndex,:,0]), \
                                 (mz1-testPep_.b_intMzs[csIndex,:,0]))
                    int_bIntDiffs=((mz0-testPep_.b_intMzs[csIndex,:,1]), \
                                   (mz1-testPep_.b_intMzs[csIndex,:,1]))
                    
                    int_intyDiffs=((mz0-testPep_.int_yMzs[csIndex,:,0]), \
                                   (mz1-testPep_.int_yMzs[csIndex,:,0]))  
                    y_intyDiffs=((mz0-testPep_.int_yMzs[csIndex,:,1]), \
                                 (mz1-testPep_.int_yMzs[csIndex,:,1]))
                
                    testCorrs=[(b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (0, None), (0, None)),\
                   
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (18.01056, 'H2O'), (0, None)),\
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (0, None), (18.01056, 'H2O')),\
                   
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (17.02653, 'NH3'), (0, None)),\
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (0, None), (17.02653, 'NH3')),\
                   
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (18.01056, 'H2O'), (18.01056, 'H2O')),\
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (17.02653, 'NH3'), (17.02653, 'NH3')),\
                   
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (18.01056, 'H2O'), (17.02653, 'NH3')),\
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (17.02653, 'NH3'), (18.01056, 'H2O')),\
                   
                   ###
                   (int_intyDiffs, y_intyDiffs, ('int_y', 2), (0, None), (0, None)),\
                   
                   (int_intyDiffs, y_intyDiffs, ('int_y', 2), (18.01056, 'H2O'), (0, None)),\
                   (int_intyDiffs, y_intyDiffs, ('int_y', 2), (0, None), (18.01056, 'H2O')),\
                   
                   (int_intyDiffs, y_intyDiffs, ('int_y', 2), (17.02653, 'NH3'), (0, None)),\
                   (int_intyDiffs, y_intyDiffs, ('int_y', 2), (0, None), (17.02653, 'NH3')),\
                   
                   (int_intyDiffs, y_intyDiffs, ('int_y', 2), (18.01056, 'H2O'), (18.01056, 'H2O')),\
                   (int_intyDiffs, y_intyDiffs, ('int_y', 2), (17.02653, 'NH3'), (17.02653, 'NH3')),\
                   
                   (int_intyDiffs, y_intyDiffs, ('int_y', 2), (18.01056, 'H2O'), (17.02653, 'NH3')),\
                   (int_intyDiffs, y_intyDiffs, ('int_y', 2), (17.02653, 'NH3'), (18.01056, 'H2O')),\
                   
                   #Now with a-ion loss from the b-ion
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (27.99492, 'CO'), (0, None)),\
                   
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (18.01056+27.99492, 'CO-H2O'), (0, None)),\
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (27.99492, 'CO'), (18.01056, 'H2O')),\
                   
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (17.02653+27.99492, 'CO-NH3'), (0, None)),\
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (27.99492, 'CO'), (17.02653, 'NH3')),\
                   
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (18.01056+27.99492, 'CO-H2O'), (18.01056, 'H2O')),\
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (17.02653+27.99492, 'CO-NH3'), (17.02653, 'NH3')),\
                   
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (18.01056+27.99492, 'CO-H2O'), (17.02653, 'NH3')),\
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (17.02653+27.99492, 'CO-NH3'), (18.01056, 'H2O')),\
                
                   #Now with a-ions from the internal ion
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (0, None), (27.99492, 'CO')),\
                   
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (18.01056, 'H2O'), (27.99492, 'CO')),\
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (0, None), (18.01056+27.99492, 'CO-H2O')),\
                   
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (17.02653, 'NH3'), (27.99492, 'CO')),\
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (0, None), (17.02653+27.99492, 'CO-NH3')),\
                   
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (18.01056, 'H2O'), (18.01056+27.99492, 'CO-H2O')),\
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (17.02653, 'NH3'), (17.02653+27.99492, 'CO-NH3')),\
                   
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (18.01056, 'H2O'), (17.02653+27.99492, 'CO-NH3')),\
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (17.02653, 'NH3'), (18.01056+27.99492, 'CO-H2O')),\
                   
                   ###
                   (int_intyDiffs, y_intyDiffs, ('int_y', 2), (27.99492, 'CO'), (0, None)),\
                   
                   (int_intyDiffs, y_intyDiffs, ('int_y', 2), (18.01056+27.99492, 'C0-H2O'), (0, None)),\
                   (int_intyDiffs, y_intyDiffs, ('int_y', 2), (27.99492, 'CO'), (18.01056, 'H2O')),\
                   
                   (int_intyDiffs, y_intyDiffs, ('int_y', 2), (17.02653+27.99492, 'CO-NH3'), (0, None)),\
                   (int_intyDiffs, y_intyDiffs, ('int_y', 2), (27.99492, 'CO'), (17.02653, 'NH3')),\
                   
                   (int_intyDiffs, y_intyDiffs, ('int_y', 2), (18.01056+27.99492, 'CO-H2O'), (18.01056, 'H2O')),\
                   (int_intyDiffs, y_intyDiffs, ('int_y', 2), (17.02653+27.99492, 'CO-NH3'), (17.02653, 'NH3')),\
                   
                   (int_intyDiffs, y_intyDiffs, ('int_y', 2), (18.01056+27.99492, 'CO-H2O'), (17.02653, 'NH3')),\
                   (int_intyDiffs, y_intyDiffs, ('int_y', 2), (17.02653+27.99492, 'CO-NH3'), (18.01056, 'H2O')),\
                   
                   # Now with a-ions from the b-ion and the internal ion
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (27.99492, 'CO'), (27.99492, 'CO')),\
                   
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (18.01056+27.99492, 'CO-H2O'), (27.99492, 'CO')),\
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (27.99492, 'CO'), (18.01056+27.99492, 'CO-H2O')),\
                   
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (17.02653+27.99492, 'CO+NH3'), (27.99492, 'CO')),\
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (27.99492, 'CO'), (17.02653+27.99492, 'CO-NH3')),\
                   
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (18.01056+27.99492, 'CO-H2O'), (18.01056+27.99492, 'CO-H2O')),\
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (17.02653+27.99492, 'CO-NH3'), (17.02653+27.99492, 'CO-NH3')),\
                   
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (18.01056+27.99492, 'CO-H2O'), (17.02653+27.99492, 'CO-NH3')),\
                   (b_bIntDiffs, int_bIntDiffs, ('b_int', 2), (17.02653+27.99492, 'CO-NH3'), (18.01056+27.99492, 'CO-H2O'))] + testCorrs
                   #so that internal correlations take priority in being matched
                
                #%%
                for diffArray0, diffArray1, category, nl0, nl1 in testCorrs:
                    for z0, z1 in [(0,1), (1,0)]: #numbers in the tuples
                    #correspond to which experimental correlation was matched
                        
                        match=(abs(diffArray0[z0]+(nl0[0]/float(csIon0)))<=\
                               args.fragIonTol)*(abs(diffArray1[z1]+(nl1[0]/\
                                              float(csIon1)))<=args.fragIonTol)
                        
                        if np.any(match): #given a sensible mass tolerance 
                        #will only ever be only one match per ion correlation
                            matchEntry=[formatMatch(match, category[0], testPep_,\
                                       cs0=csIon0, cs1=csIon1, nl0=nl0[1], nl1=nl1[1]), \
                                       round(expCorr[z0],2), round(expCorr[z1], 2)]
                            if not matched:
                                matchList[i][category[1]].append(matchEntry)
                                seqLsScores[i][category[1]]+=expCorr[3] #this
                                #is pC-2DMS correlation score
                                seqLsCounts[i][category[1]]+=1
                                matched=True
                                #print matchEntry
                            else:
                                extraMatchList[i][category[1]].append(matchEntry)

#%% now save results
np.save(args.saveTo+'/scores.npy',np.array(seqLsScores))

np.save(args.saveTo+'/matches.npy', matchList)
np.save(args.saveTo+'/extraMatches.npy', extraMatchList)    
np.save(args.saveTo+'/sequences.npy', seqLs) #same sequences for everything

with open(args.saveTo+'/info.txt', 'a') as f_info:
    f_info.write('expCorrs found in='+args.expCorrsIn+'\n')
    f_info.write('acThresh='+str(args.acThresh)+'\n')
    f_info.write('takeTop='+str(args.takeTop)+'\n')
    f_info.write('fragIonTol='+str(args.fragIonTol)+'\n')
    f_info.write('parCs='+str(args.parCs)+'\n')
    f_info.write('information on origin of sequences: '+seqsInfo)
f_info.close()