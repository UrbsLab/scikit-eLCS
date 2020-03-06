from skrebate import ReliefF
from DataCleanup import *
from eLCS import *
import time
import numpy as np
from sklearn.model_selection import cross_val_score
from RuleCompaction import RuleCompacter

'''Separates out features into np_array of shape [number of items, number of features per item] 
and labels into np_array of shape [number of items]'''

converter = StringEnumerator("Datasets/Real/ContinuousAndNonBinaryDiscreteAttributes.csv","Class")
headers,classLabel,dataFeatures,dataPhenotypes = converter.getParams()

formatted = np.insert(dataFeatures,dataFeatures.shape[1],dataPhenotypes,1)
np.random.shuffle(formatted)
dataFeatures = np.delete(formatted,-1,axis=1)
dataPhenotypes = formatted[:,-1]

model = eLCS(learningIterations = 5000,evalWhileFit=True,learningCheckpoints=np.array([99,999,3999]),trackingFrequency=100)
model.fit(dataFeatures,dataPhenotypes)

model.exportFinalRulePopulationToCSV(ALKR=True)