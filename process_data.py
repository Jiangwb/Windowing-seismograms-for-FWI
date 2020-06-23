#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:59:11 2020

@author: xichengpeng
"""
import obspy
import numpy as np
from obspy.core import read
import matplotlib.pyplot as plt
from obspy.core import Trace
from obspy.io.sac import SACTrace
from tqdm import tqdm
import os


def Bandpass(datapath):
    # for j in tqdm(range(10),desc = "prosessing"):
    filepath = datapath + "/predict/syn/Z/"
    overpath = datapath + "/predict/syn/Z/"
    os.chdir(filepath)
    
    for i in range(300):
        st = obspy.read(filepath + str(i) + '.sac')
        tr = st[0]
        tr.filter('bandpass', freqmin=8, freqmax=15, corners=4, zerophase=True)
        # tr.filter('highpass', freq=8, corners=4, zerophase=True)
        tr = (tr.data) / np.max(abs(tr.data))
        sacfile = Trace()
        sacfile.data = tr[:]
        
        sac = SACTrace.from_obspy_trace(sacfile)
        sac_data = sac.data
        sac.stla=35
        sac.stlo=110+(80+72*500+i*25)/111000
       
        sac.delta=0.0006
        sac.evla=35
        sac.evlo=110+(6568+500*72)/111000
        sac.evdp=0.05
        sac.write(overpath + str(72*520+i) + ".sac")
def normalization(datapath):            
    filepath = datapath + '/predict/syn/Z/'
    for i in tqdm(range(300),desc = 'processing'):
        st = read(filepath + str(i) + '.sac')
        tr = (st[0].data) / np.max(abs(st[0].data))
        sacfile = Trace()
        sacfile.data = tr[:]
        sacfile.write(filepath + str(i) + ".sac",format = "SAC")    
if __name__ == "__main__":
    datapath = "./data"
    Bandpass(datapath)
    normalization(datapath)
