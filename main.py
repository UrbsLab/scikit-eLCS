from skrebate import ReliefF
from DataCleanup import *


'''Separates out features into np_array of shape [number of items, number of features per item] 
and labels into np_array of shape [number of items]'''

converter = StringEnumerator("ChronicMedicationsSurvey.csv","From a scale of 1 to 10, how much do you like the idea of this program?")
#converter = StringEnumerator("ChronicMedicationsSurvey.csv","If your pharmacy offers this program, will you opt-in?")
converter.deleteAllInstancesWithoutHeaderData("How often do you visit your main prescriber?")
converter.deleteAttribute("Timestamp")
converter.deleteAttribute("Worker ID (on the top left corner of your mTurk dashboard):")

converter.changeHeaderName("How many unique medications do you take regularly that require you to follow a specific dosing schedule (for example: once a day, once every four hours, etc…)?","# of Meds")
converter.changeHeaderName("Do you take any addictive medication? (opioids, antidepressants, stimulants, benzodiazepines)","Take Addictive Meds?")
converter.changeHeaderName("How often do you visit the pharmacy?","Pharmacy Visit Freq")
converter.changeHeaderName("How often do you visit your main prescriber?","Physician Visit Freq")
converter.changeHeaderName("I forget to take my medication","Forgets?")
converter.changeHeaderName("My medication regimen is right for me (drug works, prescribed dosage works)","Right Regimen?")
converter.changeHeaderName("I think it is important to stick to my prescription","Adherence Importance?")
converter.changeHeaderName("I consider myself to be extremely adherent to all of my prescriptions (follows dosage correctly)","Personal Adherence Rating")
converter.changeHeaderName("Will being a part of this program improve your prescription adherence?","Will Program Improve Adherence?")
converter.changeHeaderName("From a scale of 1 to 10, how much do you like the idea of this program?","Program NPS")
converter.changeHeaderName("If your pharmacy offers this program, will you opt-in?","Opt in")

converter.addAttributeConverter("# of Meds",np.array(["1 medication","2 medications","3 to 5 medications","More than 5 medications"]))
converter.addAttributeConverter("Take Addictive Meds?",np.array(["No","Yes"]))
converter.addAttributeConverter("Pharmacy Visit Freq",np.array(["More than once a week","Once a week","2 to 3 times a month","Once a month","A few times a year","Once a year, or less"]))
converter.addAttributeConverter("Physician Visit Freq",np.array(["More than once a week","Once a week","2 to 3 times a month","Once a month","A few times a year","Once a year, or less"]))
converter.addAttributeConverter("Opt in",np.array(["No","Yes"]))
converter.addAttributeConverterRandom("Gender")
converter.addAttributeConverterRandom("Ethnicity")
converter.addAttributeConverter("Household Income",np.array(["Less than 25k/year","25-50k/year","50k-100k/year","100-200k/year","More than 200k/year"]))
converter.addAttributeConverter("Age",np.array(["Less than 20","20 to 34","35 to 49","50 to 64","65 and over"]))
converter.addAttributeConverterRandom("Health insurance:")
converter.addAttributeConverter("Medication Treatment Coverage:",np.array(["None of my medications are covered by insurance","My medications are mostly covered by insurance, but I have high copays/high deductibles"
                                                                           ,"My medications are mostly covered by insurance, and I have low copays/low deductibles","My medications are fully covered by insurance"]))
#converter.addClassConverter(np.array(["No","Yes"]))
converter.convertAllAttributes()
converter.print()

dataFeatures,dataPhenotypes = converter.dataFeatures,converter.dataPhenotypes
dataFeatures = dataFeatures = dataFeatures.astype('float64')

#clf = eLCS(learningIterations=5000,discreteAttributeLimit=3) #Can add parameters if you want, but default values are preprogrammed

#clf = make_pipeline(ReliefF(n_features_to_select=4, n_neighbors=30),eLCS(learningIterations=400, discreteAttributeLimit=3))
#clf = eLCS(learningIterations=2000, discreteAttributeLimit=10)
#dataFeatures, dataPhenotypes = clf.preFit("Multiplexer20.csv","NA","class")


clf = ReliefF(n_features_to_select=16, n_neighbors=30)
clf = clf.fit(dataFeatures,dataPhenotypes)
print(clf.feature_importances_)
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

