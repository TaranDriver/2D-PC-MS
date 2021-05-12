 # -*- coding: utf-8 -*-
"""
Created on Thu Jun 01 13:49:43 2017

@author: Taran Driver

Some functions and classes for matching peptides in a database with 
experimental data.

"""

from __future__ import division #IMPORTANT! __future__ imports have to come 1st
import dbProts as dbP
import numpy as np
from scipy.sparse import csr_matrix

#Dictionary of monoisotopic masses of Amino Acids in Da, to 5 d.p.
resMassDict = {'A': 71.03711, 'R': 156.10111, 'N': 114.04293,
'D': 115.02694, 'C': 103.00919, 'E': 129.04259, 'Q': 128.05858, \
'G': 57.02146, 'H': 137.05891, 'I': 113.08406, 'L': 113.08406, \
'K': 128.09496, 'M': 131.04049, 'F': 147.06841, 'P': 97.05276, \
'S': 87.03203, 'T': 101.04768, 'W': 186.07931, 'Y': 163.06333, \
'V': 99.06841}

#Dictionary of monoisotopic mass changes due to modifications, to 4 or 5 d.p.
modMassDict={'p': 79.9663, 'Me': 14.0157, 'Ac': 42.0106, 'pm': 238.2297, \
's': 79.95682, 'NH2': -0.98402, 'Me2': 28.0314, 'Me3': 42.0471, \
'Myr': 210.1984, 'nitro': 44.98508}

def resMass(seq):
    'calculate the summed mass of the residues + modifications in seq'
    seqSpl=seq.split('(')
    massList=[resMassDict[res] for res in seqSpl[0]]
    if len(seqSpl)==1:
        return massList    
    else:
        for sect in seqSpl[1:]:
            sectSpl=sect.split(')') #sectSpl will always be length 2, every
            #opening bracket is succeeded by exactly one closing bracket
            massList[-1]+=modMassDict[sectSpl[0]]
            massList+=[resMassDict[res] for res in sectSpl[1]]
            #if modification is to the C-terminal residue, sectSpl[1]=='' so the
            #massList is extended by [], i.e. remains the same
        return massList

def mrPar(baseMr): #returns parent Mr from base mr (=sum of residue masses)
    return baseMr+17.00274+1.00782

def mr_b(baseMr): #returns b-type fragment Mr from base Mr (=sum of residue \
    #masses)
    return baseMr #mz() will add protons for ions

def mr_y(baseMr): #returns y-type fragment Mr from base Mr (=sum of residue \
    #masses)
    return baseMr+17.00274+1.00782 #mz() will add protons for ions

def mz(mr, chargeState):
    "Uses Mr and charge state, protonates/deprotonates and returns m/z"
    #Requires from __future__ import division at top of file
    return (mr+chargeState*1.00728)/abs(chargeState)
    #charge state absolute so neg ion mode works too

class protDatabase(object):
    'database consisting of full protein sequences'
    def __init__(self, dbPath):
        self.seqs=dbP.importSeqs(dbPath)
        self.numSeqs=len(self.seqs)
        
    def digest(self, cleave=['R', 'K'], notCleave=['P'], miss=0, term='C', \
               reportPCent=1): #reportPCent is percentage to print report at
        "Default is tryptic digest, no missed cleavages"
        reportAt=int(self.numSeqs/(100/reportPCent))
        digSeqs=[] #digest sequences - result of digestion. Initialisation
        
        count=0
        for seq in self.seqs:
            digSeqs+=dbP.digest(seq, cleave=cleave, notCleave=notCleave, \
                                miss=miss, term=term)
            count+=1
            if count%reportAt==0:
                print 'digested '+str(count)+'/'+str(self.numSeqs)+\
                ' proteins ('+'%.2f'%((count*100)/float(self.numSeqs))+'%)'
                
        print '\ndigested ALL (100%) proteins ('+str(len(digSeqs))+' peptides)'
        
        origin=self, cleave, notCleave, miss, term
        
        return digSeqs, origin
    
    def filtXBZUO(self):
        'filter all sequences containing the ambiguous amino acid codes XBZUO'
        newSeqs=[] #this is the database to digest
        for seq in self.seqs:
            if not ('X' in seq or 'B' in seq or 'Z' in seq or 'U' in \
                    seq or 'O' in seq):
                newSeqs.append(seq)
        
        self.seqs=newSeqs
        self.numSeqs=len(self.seqs)
    
class pepPool(object):
    "'peptide pool'. Disordered set of peptide sequences from a digest, with \
    min length and max Mr"
    
    def __init__(self, digestOut, minLen=5, maxMr=4000):
        "digestOut the output of digestion of a protein database \
        (protDatabase.digest())"      
        
        seqs, origin=digestOut
        self.protdB=origin[0]
        self.cleave=origin[1]
        self.notCleave=origin[2]
        self.miss=origin[3]
        self.term=origin[4]
        
        self.setMinLen=minLen
        self.setMaxMr=maxMr  

        self.compileMassList(seqs)
        
    def compileMassList(self, seqs):
        'produce sparse array of mass of each residue for each sequence in the\
        peptide pool, for efficient further computation'
        newSeqs=[]
        rows=[]
        cols=[]
        resMasses=[]
        
        self.maxSeqLen=0 #initialisation
        newSeqCount=0 #initialisation
        
        for seq in seqs:
            seqLen=len(seq)
            if seqLen>=self.setMinLen:
                resMassLs=resMass(seq)
                if mrPar(sum(resMassLs))<self.setMaxMr:
                    newSeqs+=[seq]
                    rows+=[newSeqCount for _ in range(seqLen)] #new row for 
                    #each new sequence
                    cols+=range(seqLen) #each residue mass in different column
                    resMasses+=resMassLs
                    
                    self.maxSeqLen=max(self.maxSeqLen, seqLen)
                    newSeqCount+=1
            
        self.seqs=newSeqs
        self.numSeqs=newSeqCount
        
        print 'Now converting lists to numpy arrays to make sparse matrix of \
 mass list...'
        
        self.rows=np.asarray(rows)#convert to array for sparse matrix creation
        self.cols=np.asarray(cols)#convert to array for sparse matrix creation
        self.resMasses=np.asarray(resMasses) #convert to array for sparse
        #matrix creation
        self.matShape=np.array([self.numSeqs, self.maxSeqLen]) #shape of 
        #sparse matrix
        
        print 'Now making sparse array of mass list...'
        self.massList=(csr_matrix((self.resMasses, (self.rows,self.cols)),\
                                  shape=self.matShape)) #csr allows fancy
        #indexing with list of indices indexing (with bool it doesn't quite 
        #work as expected), is efficient with this across rows, and is 
        #relatively quick to construct. It also allows very quick summation 
        #across rows for the parMrs array. coo is quicker to construct but 
        #doesn't allow the indexing or arithmetic operations, lil is meant 
        #to be better for fancy indexing but csr works for it too and lil is 
        #restrictively slow to construct (minutes) and not as efficient
        #on the arithmetic operations. See:
        #https://docs.scipy.org/doc/scipy-0.18.1/reference/sparse.html
        self.parMrs=np.asarray(self.massList.sum(axis=1)+1.00782+17.00274)
        #this above could be calculated using mrPar() (defined above) I think,
        #but going to leave it as it is for now
        
    def randInds(self, sampSize):
        "Gives a list of random indices, with no replacement"
        return np.random.choice(self.numSeqs, sampSize)

    def coIso(self, parMz, cS, parTol=7, returnInds=False): #parent ion 
        #tolerance in ppm
        "Returns all peptides in pepPool that would be co-isolated with a \
        parent ion of m/z parMz and charge state cS at tolerance parTol ppm"
        tolAbs=parTol*(parMz/float(1e6))
        diffs=parMz-mz(self.parMrs, cS)
        
        match=abs(diffs)<tolAbs
        whereMatch=np.where(match) #useful for track of which peptides too
        #Now should be small enough to be array
        if returnInds: #added returnInds on 20200211
            return self.massList[whereMatch[0]], whereMatch[0]
        else:
            return self.massList[whereMatch[0]] #list of indices used instead
        #of boolean array for fancy indexing into self.massList, because fancy 
        #indexing for csr_matrix doesn't seem to work as expected with 
        #boolean arrays.
    
class testPep(object):
    'class defining potential fragment ions and some related operations for a \
    protonated peptide sequence'
    def __init__(self, massList, cS):
        'initialises by defining masses of canonical fragment ions and their \
        correlations'
        if type(massList)==csr_matrix:
            massList=massList.toarray()[0,:]
        elif type(massList)!=np.ndarray:
            raise TypeError, 'massList must be numpy array or csr_matrix'
        massList=massList[massList!=0.0] #when taken from sparse array there
        #there may be zero padding at end of mass list
        self.massList=massList
        self.length=len(massList)
        self.cS=cS
        
        numInts=int(((self.length-3)*(self.length-2))/2) #number of internal 
        #ions - sum of natural numbers up to n is n(n+1)/2
        cSRanges=np.arange(1, abs(self.cS)+1)
        if self.cS<0: cSRanges*=-1 #if this is a negative ion
        
        self.bMrs=np.cumsum(self.massList)[1:] #can't have b1 ion so start
        #from b2
        self.bMzs=mz(self.bMrs, cSRanges[:,np.newaxis])
        #row 0 is singly charged, row 1 doubly charged, etc.
        self.yMrs=mr_y(np.cumsum(self.massList[::-1])[::-1])
        #including y1 ion
        self.yMzs=mz(self.yMrs, cSRanges[:,np.newaxis])
        #row 0 is singly charged, row 1 doubly charged, etc.
        refInds=np.arange(1, self.length-2)[::-1] #reference indices for
        #filling of internal ion array etc.
        
        self.intMrs=np.zeros(numInts)
        self.bMzsStretch=np.repeat(self.bMzs[:,:-3], refInds[1:], axis=1)
        #stretch to 'fit against' intMzs
        self.yMzsStretch=np.zeros((abs(self.cS), numInts)) #stretch to 'fit 
        #against' intMzs, this will be filled in the coming loop

        startInd=0 #starting index (index is where to fill receiving arrays 
                   #from)
        startRes=1 #starting residue (residue of sequence from which to take 
                   #first residue of internal ion)
        
        for i1 in refInds:
            self.intMrs[startInd:startInd+i1]=\
            np.cumsum(self.massList[startRes:self.length-1])[1:]
            #fill internal ion array as follows: sit on N-terminal cleavage,
            #find mass of all internal ions from length 2 to max length 
            #(last with penultimate residue in parent sequence)
            self.yMzsStretch[:,startInd:startInd+i1]=self.yMzs[:,startRes+2:]
            #this sets the y mzs up against their corresponding internal mzs
            startRes+=1
            startInd+=i1
            
        self.intMzs=mz(self.intMrs, cSRanges[:,np.newaxis]) 
        #convert internal mrs to mzs. row 0 is singly charged, row 1 doubly 
        #charged, etc.
        numCsCombs=int(((abs(self.cS)-1)*abs(self.cS))/2) #number of charge  
        #state combinations
        
        #Following arrays have dimensions (charge state combination, ion index, 
        #ion type)
        self.b_intMzs=np.zeros((numCsCombs, self.bMzsStretch.shape[1], 2))
        #initialisation
        self.int_yMzs=np.zeros((numCsCombs, numInts, 2)) #initialisation
        self.b_yMzs=np.zeros((numCsCombs, self.length-2, 2)) #initialisation.
        #Can't have b1 ions, so correlations of b2 up to b(n-1)
        
        i2=0
        for fullCharge in np.arange(2, abs(self.cS)+1):
            for cSi in range(fullCharge-1): #cSi is charge state INDEX. This
                #is an index, it does not correspond to an actual charge state!
                self.b_intMzs[i2,:,0]=self.bMzsStretch[cSi]
                self.b_intMzs[i2,:,1]=self.intMzs[fullCharge-2-cSi, \
                             self.length-3:]
                #exclude internal ions complementary to b1 ions
                self.int_yMzs[i2,:,0]=self.intMzs[cSi]
                self.int_yMzs[i2,:,1]=self.yMzsStretch[fullCharge-2-cSi]
                
                self.b_yMzs[i2,:,0]=self.bMzs[cSi, :-1]
                self.b_yMzs[i2,:,1]=self.yMzs[fullCharge-2-cSi, 2:]
                #exclude y ions complementary to b1 ions
                i2+=1
        
    def intInds(self, i):
        'maps position of internal ion m/z in self.int_yMzs to correct \
        residue numbering'
        lenInts=len(self.intMrs)
        x=lenInts-i
        triRoot=(np.sqrt((8*x)+1)-1)/2.0 #triangular root
        
        nextUp=np.ceil(triRoot)
        nInd=nextUp+1
        nextDown=nextUp-1
        diff=x-(nextDown*(nextDown+1)/2)
        
        assert diff>1e-8 #in case np.ceil (to make nextUp) ceilinged 
        #x.000000001 when it should have been x.0
        return self.length-nInd, self.length-nInd+(nextUp-diff+1)
        
    def b_yAnnot(self, i): 
        'maps index of b&y correlation in self.b_yMzs to correct residue \
        numbering'
        return int(i+2), int(self.length-(i+2)) #b-index, y-index
    
    def b_intAnnot(self, i):
        'maps index of b&int correlation in self.b_intMzs to correct residue \
        numbering'
        x=self.intInds(i+(self.length-3)) #because no b1 ions
        return int(x[0]-1), int(x[0]), int(x[1]) #b-index, biFrom, biTo
    
    def int_yAnnot(self, i):
        'maps index of int&y correlation in self.int_yMzs to correct residue \
        numbering'
        x=self.intInds(i)
        return int(x[0]), int(x[1]), int(self.length-x[1]) #biFrom, biTo, 
        #yindex