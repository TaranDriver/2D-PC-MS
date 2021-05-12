# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 16:03:52 2018

@author: vaverbuk

Plot bar charts from search engine results
"""

import numpy as np
from matplotlib import pyplot as plt

scoresIn=r"D:\Vitali\Data\20160905_1151_UN14_2+_1nM_CID_NCE35_maxInjectTime1000ms_Turbo\SearchResults\scores.npy"
seqsIn=r"D:\Vitali\Data\20160905_1151_UN14_2+_1nM_CID_NCE35_maxInjectTime1000ms_Turbo\SearchResults\sequences.npy"

scores=np.load(scoresIn)
seqs=np.load(seqsIn)

fig0=plt.figure()
ax0=fig0.add_subplot(111)
ax0.hist(scores[:])
plt.show()