from skrebate import ReliefF
from DataCleanup import *
from eLCS import *
from sklearn.model_selection import cross_val_score

'''Separates out features into np_array of shape [number of items, number of features per item] 
and labels into np_array of shape [number of items]'''

converter = StringEnumerator("Datasets/Real/Multiplexer11.csv","class")
headers, classLabel, dataFeatures,dataPhenotypes = converter.getParams()

clf = eLCS(learningIterations=5000,doSubsumption=True)
clf.fit(dataFeatures,dataPhenotypes)
print(clf.score(dataFeatures,dataPhenotypes))
clf.printPopSet()
#print(np.mean(cross_val_score(clf,dataFeatures,dataPhenotypes)))

# clf.fit(dataFeatures,dataPhenotypes)
#
#
#
# print(clf.population.popSet.size);
#
# for i in range(clf.popStatObjs.size):
#     print(clf.popStatObjs[i].trainingAccuracy)
# print()
# for i in range(clf.trackingObjs.size):
#    print(clf.trackingObjs[i].aveGenerality)
# print(clf.score(dataFeatures,dataPhenotypes))

