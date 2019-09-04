import pandas as pd

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from eLCS import eLCS
from skrebate import ReliefF
from Classifier import *


'''Separates out features into np_array of shape [number of items, number of features per item] 
and labels into np_array of shape [number of items]'''

data = pd.read_csv('Multiplexer6.csv',sep=',') #Puts data from csv into indexable np arrays
dataFeatures, dataPhenotypes = data.drop('class', axis=1).values, data['class'].values
clf = eLCS(learningIterations=1000) #Can add parameters if you want, but default values are preprogrammed
print(np.mean(cross_val_score(clf, dataFeatures, dataPhenotypes)))


#clf = clf.fit(dataFeatures,dataPhenotypes)
for i in range(clf.popStatObjs.size):
    print(clf.popStatObjs[i].trainingAccuracy)
print()
for i in range(clf.trackingObjs.size):
    print(clf.trackingObjs[i].aveGenerality)
#print(clf.score(dataFeatures,dataPhenotypes))

