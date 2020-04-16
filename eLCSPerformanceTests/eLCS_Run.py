"""
Name:        eLCS_Run.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     November 1, 2013
Description: To run e-LCS, run this module.  A properly formatted configuration file, including all run parameters must be included with the path to that 
             file given below.  In this example, the configuration file has been included locally, so only the file name is required.
             
---------------------------------------------------------------------------------------------------------------------------------------------------------
eLCS: Educational Learning Classifier System - A basic LCS coded for educational purposes.  This LCS algorithm uses supervised learning, and thus is most 
similar to "UCS", an LCS algorithm published by Ester Bernado-Mansilla and Josep Garrell-Guiu (2003) which in turn is based heavily on "XCS", an LCS 
algorithm published by Stewart Wilson (1995).  

Copyright (C) 2013 Ryan Urbanowicz 
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the 
Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABLILITY 
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
"""
from eLCS_Timer import Timer
from eLCS_ParamParser import ParamParser
from eLCS_Offline_Environment import Offline_Environment
from eLCS_Algorithm import eLCS
from eLCS_Constants import *
import numpy as np
import time

def runOriginaleLCS(dataFile,labelPhenotype,learningIterations,randomSeed,cv=False):
    #Run the e-LCS algorithm.
    if cv == False:
        ParamParser(dataFile,cv=cv,labelPhenotype=labelPhenotype,learningIterations=learningIterations,randomSeed=randomSeed)
        timer = Timer()
        cons.referenceTimer(timer)
        env = Offline_Environment()
        cons.referenceEnv(env)
        cons.parseIterations()
        e = eLCS()
        return np.array([e.trainEval[0],cons.timer.globalDeletion,cons.timer.globalEvaluation,cons.timer.globalMatching,cons.timer.globalSelection,cons.timer.globalSubsumption,cons.timer.globalTime])
    else:
        l = []
        ParamParser(dataFile,cv=cv,labelPhenotype=labelPhenotype,learningIterations=learningIterations,randomSeed=randomSeed)
        for i in range(cv):
            cons.setCV()
            timer = Timer()
            cons.referenceTimer(timer)
            env = Offline_Environment()
            cons.referenceEnv(env)
            cons.parseIterations()
            e = eLCS()
            l.append(e.testEval[0])
        return np.mean(np.array(l))

import skeLCS
import pandas as pd
from sklearn.model_selection import cross_val_score

def runScikiteLCS(dataFile,classLabel,learningIterations,randomSeed,cv=False):
    data = pd.read_csv(dataFile)
    dataFeatures = data.drop(classLabel,axis=1).values
    dataPhenotypes = data[classLabel].values
    model = skeLCS.eLCS(learningIterations = learningIterations,randomSeed = randomSeed)
    random.seed(randomSeed)

    if cv == False:
        model.fit(dataFeatures,dataPhenotypes)
        score = model.score(dataFeatures,dataPhenotypes)
        return np.array([score,model.timer.globalDeletion,model.timer.globalEvaluation,model.timer.globalMatching,model.timer.globalSelection,model.timer.globalSubsumption,model.timer.globalTime])
    else:
        formatted = np.insert(dataFeatures,dataFeatures.shape[1],dataPhenotypes,1)
        random.shuffle(formatted)
        dataFeatures = np.delete(formatted,-1,axis=1)
        dataPhenotypes = formatted[:,-1]
        return np.mean(cross_val_score(model,dataFeatures,dataPhenotypes,cv=cv))

randomSeeds = [0]

# avgOriginal = 0
# for seed in randomSeeds:
#     o = runOriginaleLCS('Datasets/Multiplexer11.csv','Class','1000',seed,cv=3)
#     print(o)
#     avgOriginal += o
# avgOriginal /= len(randomSeeds)
# print("Average Testing Accuracy: "+str(avgOriginal))
#
# avgScikit = 0
# for seed in randomSeeds:
#     s = runScikiteLCS('Datasets/Multiplexer11.csv','Class',1000,seed,cv=3)
#     print(s)
#     avgScikit += s
# avgScikit /= len(randomSeeds)
# print("Average Testing Accuracy: "+str(avgScikit))
#
# o = runOriginaleLCS('Datasets/Multiplexer11.csv','Class','1000',seed)[0]
# s = runScikiteLCS('Datasets/Multiplexer11.csv','Class',1000,seed)[0]
# print(o)
# print(s)

import matplotlib.pyplot as plt

fig,axes = plt.subplots(ncols=3,sharey=True)
fig.subplots_adjust(wspace=0)

A = [[0,1,2],[3,4,5]]
B = [[1,0,1],[8,9,10]]
C = [[2,10,12],[5,0,5]]
data = {}
data['6-bit'] = A
data['11-bit'] = B
data['20-bit'] = C

for ax,name in zip(axes,['6-bit','11-bit','20-bit']):
    ax.boxplot([data[name][item] for item in [0, 1]])
    ax.set(xticklabels=['Original', 'scikit'], xlabel=name)
    ax.margins(0.05) # Optional
    if name == '6-bit':
        ax.set_ylabel('y axis label')
plt.show()