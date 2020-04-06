import pandas as pd
from skeLCS.eLCS import eLCS
from skeLCS.DataCleanup import StringEnumerator
import time

converter = StringEnumerator("test/DataSets/Real/Multiplexer20.csv","class")
headers,classLabel,dataFeatures,dataPhenotypes = converter.getParams()

model = eLCS(learningIterations=10000,randomSeed=0)
t = time.time()
model.fit(dataFeatures,dataPhenotypes)
print(time.time()-t)
print("Total Time:"+str(model.timer.globalTime))
model.exportIterationTrackingDataToCSV("testTrackingData.csv")
print("Deletion Time:"+str(model.timer.globalDeletion))
print("Evaluation Time:"+str(model.timer.globalEvaluation))
print("Matching Time:"+str(model.timer.globalMatching))
print("Selection Time:"+str(model.timer.globalSelection))
print("Subsumption Time:"+str(model.timer.globalSubsumption))
print("Total Time:"+str(model.timer.globalTime))
print(model.score(dataFeatures,dataPhenotypes))
