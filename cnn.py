#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 14:47:53 2018

@author: john
"""

import numpy as np
import tensorflow as tf

Xtrain = np.load('MNIST/X_train_MNIST.npy')
ytrain = np.load('MNIST/y_train_MNIST.npy')

Xtest = np.load('MNIST/X_test_MNIST.npy')
ytest = np.load('MNIST/y_test_MNIST.npy')
