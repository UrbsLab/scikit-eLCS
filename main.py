import pandas as pd

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from eLCS import eLCS
from skrebate import ReliefF
from Classifier import *
from DataCleanup import *


'''Separates out features into np_array of shape [number of items, number of features per item] 
and labels into np_array of shape [number of items]'''

converter = StringEnumerator("ChronicMedicationsSurvey.csv","From a scale of 1 to 10, how much do you like the idea of this program?")
converter.deleteAllInstancesWithoutHeaderData("How often do you visit your main prescriber?")
converter.deleteAttribute("Timestamp")
converter.deleteAttribute("Worker ID (on the top left corner of your mTurk dashboard):")
converter.addAttributeConverter("How many unique medications do you take regularly that require you to follow a specific dosing schedule (for example: once a day, once every four hours, etc…)?",
                                np.array(["1 medication","2 medications","3 to 5 medications","More than 5 medications"]))
converter.addAttributeConverter("Do you take any addictive medication? (opioids, antidepressants, stimulants, benzodiazepines)",np.array(["No","Yes"]))
converter.addAttributeConverter("How often do you visit the pharmacy?",np.array(["More than once a week","Once a week","2 to 3 times a month","Once a month","A few times a year","Once a year, or less"]))
converter.addAttributeConverter("How often do you visit your main prescriber?",np.array(["More than once a week","Once a week","2 to 3 times a month","Once a month","A few times a year","Once a year, or less"]))
converter.addAttributeConverter("If your pharmacy offers this program, will you opt-in?",np.array(["No","Yes"]))
converter.addAttributeConverter("Gender",np.array(["Male","Female"]))
converter.addAttributeConverter("Ethnicity",np.array(["Asian/Pacific Islander","Caucasian","African American","Hispanic/Latino","Other"]))
converter.addAttributeConverter("Household Income",np.array(["Less than 25k/year","25-50k/year","50k-100k/year","100-200k/year","More than 200k/year"]))
converter.addAttributeConverter("Age",np.array(["Less than 20","20 to 34","35 to 49","50 to 64","65 and over"]))
converter.addAttributeConverter("Health insurance:",np.array(["Privately Insured but not through my employer","Medicare/Medicaid/Tricare","Privately Insured through my employer","Not Insured"]))
converter.addAttributeConverter("Medication Treatment Coverage:",np.array(["None of my medications are covered by insurance","My medications are mostly covered by insurance, but I have high copays/high deductibles"
                                                                           ,"My medications are mostly covered by insurance, and I have low copays/low deductibles","My medications are fully covered by insurance"]))

converter.convertAllAttributes()
for instanceIndex in range(len(converter.dataFeatures)):
    for attribute in converter.dataFeatures[instanceIndex]:
        print(int(attribute),end="\t\t")
    print(int(converter.dataPhenotypes[instanceIndex]))

#clf = eLCS(learningIterations=5000,discreteAttributeLimit=3) #Can add parameters if you want, but default values are preprogrammed

#clf = make_pipeline(ReliefF(n_features_to_select=4, n_neighbors=30),eLCS(learningIterations=400, discreteAttributeLimit=3))
#clf = eLCS(learningIterations=2000, discreteAttributeLimit=10)
#dataFeatures, dataPhenotypes = clf.preFit("FakeMissingData3.csv","NA","class",np.array(["N4"]),np.array(['N1']),"d")
#dataFeatures, dataPhenotypes = clf.preFit("FakeMissingData3.csv","NA","class",np.array(["N4"]))
#dataFeatures, dataPhenotypes = clf.preFit("Multiplexer20.csv","NA","class")


#clf = ReliefF(n_features_to_select=4, n_neighbors=30)
#clf = clf.fit(dataFeatures,dataPhenotypes)
#print(clf.feature_importances_)
#print(np.mean(cross_val_score(clf, dataFeatures, dataPhenotypes,cv=2)))


#clf = clf.fit(dataFeatures,dataPhenotypes)
# print(clf.population.popSet.size);
#clf.printAccuratePopSet(1,1)
# print()
#
# for i in range(clf.popStatObjs.size):
#     print(clf.popStatObjs[i].trainingAccuracy)
# print()
# for i in range(clf.trackingObjs.size):
#    print(clf.trackingObjs[i].aveGenerality)
# print(clf.score(dataFeatures,dataPhenotypes))

