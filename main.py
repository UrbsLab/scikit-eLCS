from skrebate import ReliefF
from DataCleanup import *
from eLCS import *
import time
import numpy as np
from sklearn.model_selection import cross_val_score
from RuleCompaction import RuleCompacter

'''Separates out features into np_array of shape [number of items, number of features per item] 
and labels into np_array of shape [number of items]'''

clf = eLCS(learningIterations=5000)
RuleCompacter(clf,"Datasets/Real/Multiplexer11.csv","class")

# converter = StringEnumerator("Datasets/Real/Multiplexer11.csv", "class")
# headers, classLabel, dataFeatures, dataPhenotypes = converter.getParams()
#
# clf = eLCS(learningIterations=1000,evalWhileFit=True,trackingFrequency=100,randomSeed=100)
# clf.fit(dataFeatures,dataPhenotypes)
# clf.exportIterationTrackingDataToCSV(filename='DataSets/Tests/RandomTests/track1.csv')
# clf.exportFinalRulePopulationToCSV(headerNames=headers,className=classLabel,ALKR=True,filename='DataSets/Tests/RandomTests/pop1.csv')
# clf.exportFinalPopStatsToCSV(headerNames=headers,filename="DataSets/Tests/RandomTests/popStats1.csv")
#
# clf2 = eLCS(learningIterations=1000,evalWhileFit=True,trackingFrequency=100,randomSeed=100)
# clf2.fit(dataFeatures,dataPhenotypes)
# clf2.exportIterationTrackingDataToCSV(filename='DataSets/Tests/RandomTests/track2.csv')
# clf2.exportFinalRulePopulationToCSV(headerNames=headers,className=classLabel,ALKR=True,filename='DataSets/Tests/RandomTests/pop2.csv')
# clf2.exportFinalPopStatsToCSV(headerNames=headers,filename="DataSets/Tests/RandomTests/popStats2.csv")