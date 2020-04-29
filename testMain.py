import pandas as pd
from skeLCS.eLCS import eLCS
from skeLCS.DataCleanup import StringEnumerator
import time
import random
import numpy as np
from sklearn.model_selection import cross_val_score

converter = StringEnumerator("test/DataSets/Real/Multiplexer6.csv","class")
headers,classLabel,dataFeatures,dataPhenotypes = converter.getParams()

model = eLCS(learningIterations=5000,randomSeed=0)
formatted = np.insert(dataFeatures,dataFeatures.shape[1],dataPhenotypes,1)
random.shuffle(formatted)
dataFeatures = np.delete(formatted,-1,axis=1)
dataPhenotypes = formatted[:,-1]

model.fit(dataFeatures,dataPhenotypes)
print(model.score(dataFeatures,dataPhenotypes))
print(model.getFinalAccuracy())
model.exportIterationTrackingData('defaultExportDir/tracking.csv')

model.pickleModel('defaultExportDir/pickled')

newModel = eLCS(learningIterations=5000,rebootFilename='defaultExportDir/pickled')
newModel.fit(dataFeatures,dataPhenotypes)
print(newModel.score(dataFeatures,dataPhenotypes))
newModel.exportIterationTrackingData('defaultExportDir/tracking2.csv')
