from skrebate import ReliefF
from DataCleanup import *
from eLCS import *
import time
import numpy as np
from sklearn.model_selection import cross_val_score

'''Separates out features into np_array of shape [number of items, number of features per item] 
and labels into np_array of shape [number of items]'''

converter = StringEnumerator("Datasets/Real/ContinuousAndNonBinaryDiscreteAttributesMissing.csv","Class")
headers, classLabel, dataFeatures,dataPhenotypes = converter.getParams()
t = time.time()
print("start")
clf = eLCS(learningIterations=10000,doSubsumption=True)

#A manual shuffle is needed to perform a proper CV, because CV trains on the first 2/3 of instances, and tests on the last 1/3 of instances. While the algo will shuffle
#the 2/3 of instances, the original set needs to be shuffled as well.

# formatted = np.insert(dataFeatures,dataFeatures.shape[1],dataPhenotypes,1)
# np.random.shuffle(formatted)
# dataFeatures = np.delete(formatted,-1,axis=1)
# dataPhenotypes = formatted[:,-1]

clf.fit(dataFeatures,dataPhenotypes)

print(clf.score(dataFeatures,dataPhenotypes))
print(clf.timer.reportTimes())
print("total time:"+str(time.time()-t))
# clf.printPopSet()

#print(np.mean(cross_val_score(clf,dataFeatures,dataPhenotypes,scoring="balanced_accuracy"))) #Example use of external scorer

#print(np.mean(cross_val_score(clf,dataFeatures,dataPhenotypes)))



