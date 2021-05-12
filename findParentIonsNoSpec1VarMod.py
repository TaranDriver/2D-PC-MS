# -*- coding: utf-8 -*-
"""
@author: Taran Driver

This code identifies all peptides from a database that fall within a specified
ppm m/z tolerance of a given m/z value, at a given charge state.

- read in protein sequences from protein database
- digest database with no enzymatic specificity
- find all possible peptide sequences within set ppm m/z tolerance of given
  m/z value, at given parent charge state
- save these sequences to file

It handles either one type of variable modification, or no modification.
"""

import numpy as np
import time
import itertools
import argparse
import dbProts as dbP

#%% Define masses of amino acids and modifications
#Dictionary of monoisotopic weights of Amino Acids in Da, to 5 d.p.
resMassDict = {'A': 71.03711, 'R': 156.10111, 'N': 114.04293,
'D': 115.02694, 'C': 103.00919, 'E': 129.04259, 'Q': 128.05858, 'G': 57.02146,
'H': 137.05891, 'I': 113.08406, 'L': 113.08406, 'K': 128.09496, 
'M': 131.04049, 'F': 147.06841, 'P': 97.05276, 'S': 87.03203, 'T': 101.04768,
'W': 186.07931, 'Y': 163.06333, 'V': 99.06841,
'X': 110, 'B': 114.53494, 'Z': 128.55059, 'U':150.95363} #'Z' is ave of 
#Glu & Gln, 'B' is ave of Asp & Asn, 'X' is (roughly) weighted average 
#mass of all residues, 'U' is selenocysteine from 
#http://www.matrixscience.com/help/aa_help.html

#Dictionary of monoisotopic mass changes for PTMs
ptmMassDict={'p': 79.9663, 'Me': 14.0157, 'Ac': 42.0106, 
 'pm': 238.2297, 's': 79.95682,\
 'Me2': 28.0314, 'Me3': 42.0471,\
 'Myr': 210.1984, 'nitro': 44.98508,\
 'None': 0.0} #None is string because argparser class expects str for this 
               #variable

#%% Define some functions
def findChar(string, char): #find character in a string
    return [i for i, x in enumerate(string) if x==char]

def insStr(baseStr, insert, indices): #insert string
    'indices is iterable of indices, others are strings'
    indices=list(indices)
    indices.sort() #otherwise the incrementing of offset won't work
    offset=0
    offsetInc=len(insert) #offset increment
    for x in indices:
        baseStr=baseStr[:x+offset+1]+insert+baseStr[x+offset+1:] #adding after
        #the Python index of the letter
        offset+=offsetInc
    return baseStr #maybe yield to save memory

def resMass(seq):
    return [resMassDict[res] for res in seq]

#%%
parser=argparse.ArgumentParser()

parser.add_argument("--dbPath","-p", type=str, required=True) #database path
parser.add_argument("--saveTo","-s", type=str, required=True) #save sequences
#to?
parser.add_argument("--parCs", type=int, required=True) #parent charge 
#state
parser.add_argument("--parMz", type=float, required=True) #parent m/z value
parser.add_argument("--parTol", type=int, required=True) #tolerance for
#selection of parent ion, in ppm
parser.add_argument("--modRes", type=str, required=True) #modified residue
parser.add_argument("--mod", type=str, required=True) #modification

parser.add_argument("--minLen", type=int, required=False, default=5) #minimum 
#length of sequences for digest, in residues

args=parser.parse_args()

#%%
modMass=ptmMassDict[args.mod] #will take small amount of look-up time to find
#so let's define it now

#%%
parCs=float(abs(args.parCs)) #to ensure division is okay later on. This also
#accounts for negative charge states. m/z is always positive in search 
#engines, we are carrying this on here
modMZ=modMass/parCs
parTolAbs=args.parMz*(args.parTol/1e6)

peps=[]
protCount=0
keyErrors=[]

#%% Now import proteins from database
prots=dbP.importSeqs(dbPath=args.dbPath)

#%% Now the non-specific digest, looping over each protein sequence and 
# creating a list of peptides with suitable m/z at provided charge state 
#(pepLs) that can be created from  non-specific digestion
# of that particular sequence. This list is then added to the list of all
# possible peptide sequences from all tested proteins (peps)
time0=time.time()
for prot in prots:
    protCount+=1
    if protCount%1000==0:
        print str(protCount)+'('+str(len(peps))+') in '+\
        str(round((time.time()-time0)/60, 2))+' mins'
    pepLs=[]
    try:
        massList=np.array(resMass(prot))
    except:
        keyErrors.append(prot)
        continue
    numRes=len(prot)
    
    for i in np.arange(numRes-args.minLen+1):
        mzSum=(np.cumsum(massList[i:])+17.00274+1.00782+parCs*1.00728)/parCs
        #1.00727647 is the mass of a proton
        for j in range(args.minLen-1, numRes-i):
            diff0=mzSum[j]-args.parMz
            if diff0<=parTolAbs: #this on the assumption that the modification
            #will only increase the mass of the residue it's modifying 
                segment=prot[i:i+j+1]
                countThrough=1 if args.mod=='None' else \
                             segment.count(args.modRes)+1
                for numMods in range(countThrough): #from 0 
                    #to full number of mods
                    diff=diff0+numMods*modMZ   
                    if diff<-parTolAbs:
                        continue
                    elif abs(diff)<=parTolAbs:
                        pepLs.append((segment, numMods)) #tuple is hashable
                        #for removal of repeats using list(set())
                    else:
                        break
            else:
                break
    peps+=pepLs

#%%
pepsNr=list(set(peps)) #to remove repeats - 'peps no repeats'

pepsNr=[tup for tup in pepsNr if not ('X' in tup[0] or 'B' in tup[0] or \
                                      'Z' in tup[0] or 'U' in tup[0] or \
                                      'O' in tup[0])] #remove ambiguous
                                       #residues for the time being

#%% now add string of modification for peptides identified as modified
if args.mod=='None':
    outputPeps=[x[0] for x in pepsNr]
else:
    outputPeps=[]
    for y in pepsNr:
        if y[1]==0:
            outputPeps.append(y[0])
        else:
            for modInds in itertools.combinations(findChar(y[0],args.modRes),\
                                                  y[1]):
                outputPeps.append(insStr(y[0], '('+args.mod+')', modInds)) 
                #here, don't need args.mod to be in parentheses, unlike
                #previous versions

#%% now save peptide sequences in a text file
with open(args.saveTo, 'w') as f:
    for x in outputPeps:
        f.write(x+'\n')
    f.close() #should be unnecessary because of 'with' statement,
              #in here anyway