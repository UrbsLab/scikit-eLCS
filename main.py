import pandas as pd

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from eLCS import eLCS
from skrebate import ReliefF
from Classifier import *


'''Separates out features into np_array of shape [number of items, number of features per item] 
and labels into np_array of shape [number of items]'''
#clf = eLCS(learningIterations=5000,discreteAttributeLimit=3) #Can add parameters if you want, but default values are preprogrammed

#clf = make_pipeline(ReliefF(n_features_to_select=4, n_neighbors=30),eLCS(learningIterations=400, discreteAttributeLimit=3))
clf = eLCS(learningIterations=2000, discreteAttributeLimit=10)
#dataFeatures, dataPhenotypes = clf.preFit("FakeMissingData3.csv","NA","class",np.array(["N4"]),np.array(['N1']),"d")
dataFeatures, dataPhenotypes = clf.preFit("FakeMissingData3.csv","NA","class",np.array(["N4"]))


#clf = ReliefF(n_features_to_select=4, n_neighbors=30)
#clf = clf.fit(dataFeatures,dataPhenotypes)
#print(clf.feature_importances_)
#print(np.mean(cross_val_score(clf, dataFeatures, dataPhenotypes,cv=2)))


clf = clf.fit(dataFeatures,dataPhenotypes)
# print(clf.population.popSet.size);
clf.printAccuratePopSet(1,1)
# print()
#
# for i in range(clf.popStatObjs.size):
#     print(clf.popStatObjs[i].trainingAccuracy)
# print()
# for i in range(clf.trackingObjs.size):
#    print(clf.trackingObjs[i].aveGenerality)
# print(clf.score(dataFeatures,dataPhenotypes))

