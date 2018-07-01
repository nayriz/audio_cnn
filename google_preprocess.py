#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 11:54:34 2018

@author: john
"""

import os
import numpy as np
import scipy.io.wavfile as wav
import json
path1 = '/media/john/github/audio_cnn/data/google_speech/'

listdir1 = os.listdir(path1)
listdir1.sort()
sample_list = []
y = []
word_class = 0
word_dict = {}

sample_nbr = 0
for word in listdir1 :
    
    listdir_tmp = os.listdir(path1 + word)    
    listdir_tmp.sort()    
    
    internal_letter_class = 0    
    for sample in listdir_tmp:
        
        path2 = path1 +  word + '/' + sample
        sample_rate, samples = wav.read(path2)
        
        if sample_rate != 16000:
            STOP

        if len(samples) < 16000:
            
            diff = 16000 - len(samples)
            before = int(diff/2)
            after = diff - before
            samples = np.pad(samples,(before,after), 'constant' )
        
        word_dict[sample_nbr] = word
        sample_nbr += 1 
        
        sample_list.append(samples)   
        y.append(word_class)

    word_class += 1
           
        
        
X = np.array(sample_list)  
np.save('/media/john/github/audio_cnn/data/X',X)
y = np.array(y)
np.save('/media/john/github/audio_cnn/data/y',y)

with open('/media/john/github/audio_cnn/data/word_dict.json', 'w') as f:
    json.dump(word_dict, f)    
f.close()