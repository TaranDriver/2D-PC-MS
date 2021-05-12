# -*- coding: utf-8 -*-
"""
@author: Taran Driver

Some functions to handle proteins for database searching etc.
"""

def digest(prot, cleave=['R', 'K'], notCleave=['P'], miss=0, term='C'):
    "Returns a list of digested peptides. Default parameters for trypsin.\
    term: cleave C- or N- terminal to the given residues?"
    
    if term=='C':
        notCleavePs=[x+y for x in cleave for y in notCleave]
    elif term=='N':
        notCleavePs=[y+x for x in cleave for y in notCleave]
    
    for y in notCleavePs:
        prot=prot.replace(y, y.lower()) 
    #http://www.matrixscience.com/blog/non-standard-amino-acid-residues.html -
    #only capitals used in SwissProt, NCBInr, etc.

    for x in cleave:
        if term=='C':
            prot=prot.replace(x, x+'/')
        elif term=='N':
            prot=prot.replace(x, '/'+x)
    
    pepLs=prot.split('/')
    
    if term=='C' and prot.endswith('/'):
        pepLs=pepLs[:-1] #because last element will be an empty string
    elif term=='N' and prot.startswith('/'):
        pepLs=pepLs[1:] #because first element will be an empty string
        
    pepLs=[pep.upper() for pep in pepLs]
    
    #Now missed cleavages. This doesn't add trivial duplicates (i.e. if the
    #number of possible cleavages is greater than the number of specified maximum
    #cleavages)
    missPeps=[]
    for x in range(miss): #if miss=0, this doesn't run
        x+=1 #so x is now the actual number of missed cleavages
        missPeps+=[''.join(pepLs[i:i+x+1]) for i in range(len(pepLs)-x)]
    pepLs+=missPeps
    
    return pepLs

def importSeqs(dbPath):
    "Import protein sequences only from a given database in the same format \
    as SwissProt. Returns a list of protein sequences."
            
    count=0
    seqs=[]
    inProt=False 
    with open(dbPath) as f:   
        for i, line in enumerate(f):
            if line=='//\n':
                inProt=False
                seqs.append(seq) #it looks like appending takes about the same
                #amount of time as initialising a list so don't worry about it 
                #here!
                count+=1
                if count%50000==0:
                    print 'imported sequence '+str(count)+' from '+dbPath
            elif inProt:
                seq+=(''.join(line.split()))
            elif line.split()[0]=='SQ':
                inProt=True
                seq='' #empty string which can be added to
    f.close()
    
    return seqs