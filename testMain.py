import pandas as pd
from skeLCS.eLCS import eLCS
from skeLCS.DataCleanup import StringEnumerator

converter = StringEnumerator("test/DataSets/Real/Multiplexer20.csv","class")
headers,classLabel,dataFeatures,dataPhenotypes = converter.getParams()

model = eLCS(learningIterations=10000,evalWhileFit=True)
model.fit(dataFeatures,dataPhenotypes)
print(model.score(dataFeatures,dataPhenotypes))
model.exportIterationTrackingDataToCSV("testTrackingData.csv")