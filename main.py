from skrebate import ReliefF
from DataCleanup import *
from eLCS import *
import time
import numpy as np
from sklearn.model_selection import cross_val_score
from RuleCompaction import RuleCompacter

'''Separates out features into np_array of shape [number of items, number of features per item] 
and labels into np_array of shape [number of items]'''

converter = StringEnumerator("test/Datasets/Real/Multiplexer6.csv","class")
headers,classLabel,dataFeatures,dataPhenotypes = converter.getParams()

formatted = np.insert(dataFeatures,dataFeatures.shape[1],dataPhenotypes,1)
np.random.shuffle(formatted)
dataFeatures = np.delete(formatted,-1,axis=1)
dataPhenotypes = formatted[:,-1]

model = eLCS(learningIterations = 1000,evalWhileFit=True,learningCheckpoints=np.array([99,999,3999]),trackingFrequency=100)

print(np.mean(cross_val_score(model, dataFeatures, dataPhenotypes,cv=3)))

