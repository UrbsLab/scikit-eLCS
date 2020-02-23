import csv
import numpy as np

class IterationRecord():
    '''
    IterationRecord keeps track of all relevant data that comes with each iteration if the evalWhileFit param is True in eLCS object.
    It also allows for easy access to data via access methods and export to CSV options

    IterationRecord Tracks 2 dictionaries:
    1) Tracking Dict: Cursory Iteration Evaluation. Frequency determined by trackingFrequency param in eLCS. For each iteration evaluated, it saves:
        KEY-iteration number
        0-accuracy (approximate from correct array in eLCS)
        1-average population generality
        2-macropopulation size
        3-micropopulation size
        4-match set size
        5-correct set size
        6-average iteration age of correct set classifiers
        7-number of classifiers subsumed (in iteration)
        8-number of crossover operations performed (in iteration)
        9-number of mutation operations performed (in iteration)
        10-number of covering operations performed (in iteration)
        11-number of deletion operations performed (in iteration)
        12-total global time at end of iteration
        13-total matching time at end of iteration
        14-total deletion time at end of iteration
        15-total subsumption time at end of iteration
        16-total selection time at end of iteration
        17-total evaluation time at end of iteration

    2) Evaluation Dict: Full Iteration Evaluation. Frequency determined by learningCheckpoints. For each evaluation, it saves:
        KEY-iteration number
        0-evaluation accuracy (from full training data evaluation)
        1-instance coverage
        2-population set (list of population classifiers)

    '''

    def __init__(self):
        self.trackingDict = {}
        self.evaluationDict = {}

    def addToTracking(self,iterationNumber,accuracy,avgPopGenerality,macroSize,microSize,mSize,cSize,iterAvg,
                      subsumptionCount,crossoverCount,mutationCount,coveringCount,deletionCount,
                      globalTime,matchingTime,deletionTime,subsumptionTime,selectionTime,evaluationTime):

        self.trackingDict[iterationNumber] = [iterationNumber,accuracy,avgPopGenerality,macroSize,microSize,mSize,cSize,iterAvg,
                                   subsumptionCount,crossoverCount,mutationCount,coveringCount,deletionCount,
                                   globalTime,matchingTime,deletionTime,subsumptionTime,selectionTime,evaluationTime]


    def addToEval(self,iterationNumber,evalAccuracy,instanceCoverage,fullPopSet):
        self.evaluationDict[iterationNumber] = [iterationNumber,evalAccuracy,instanceCoverage,fullPopSet]

    def exportTrackingToCSV(self):
        #Exports each entry in Tracking Array as a column
        with open('iterationData.csv',mode='w') as file:
            writer = csv.writer(file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)

            writer.writerow(["Iteration","Accuracy (approx)", "Average Population Generality","Macropopulation Size",
                             "Micropopulation Size", "Match Set Size", "Correct Set Size", "Average Iteration Age of Correct Set Classifiers",
                             "# Classifiers Subsumed in Iteration","# Crossover Operations Performed in Iteration","# Mutation Operations Performed in Iteration",
                             "# Covering Operations Performed in Iteration","# Deletion Operations Performed in Iteration",
                             "Total Global Time","Total Matching Time","Total Deletion Time","Total Subsumption Time","Total Selection Time","Total Evaluation Time"])

            for k,v in sorted(self.trackingDict.items()):
                writer.writerow([k,v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12],v[13],v[14],v[15],v[16],v[17]])

    def exportEvaluationToCSV(self,iterationNumber,headerNames=np.array([]),className='phenotype'):
        '''
        Assumes at least 1 classifier is in population.
        :param iterationNumber: Which iteration to export evaluation data for
        :param headerNames: Must be ndarray. Header Names of attributes, in order. MUST have equal # of headers as there are attributes
        :param className: Name of class
        :return:
        '''

        numAttributes = len(self.evaluationDict.items()[iterationNumber][2][0].conditionType)

        headerNames = headerNames.tolist() #Convert to Python List

        #Default headerNames if none provided
        if isinstance(headerNames,np.ndarray):
            for i in range(numAttributes):
                headerNames.append("N"+str(i))

        if len(headerNames) != numAttributes:
            raise Exception("# of Header Names provided does not match the number of attributes in dataset instances.")

        if iterationNumber in self.evaluationDict:
            with open('populationData.csv', mode='w') as file:
                writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                writer.writerow(["Training Accuracy","Instance Coverage"])
                writer.writerow([self.evaluationDict[iterationNumber][0],self.evaluationDict[iterationNumber][1]])
                writer.writerow([])
                writer.writerow(headerNames+[className]+["Fitness","Accuracy","Numerosity","Iteration Initialized","Specificity"])
                classifiers = self.evaluationDict.items()[iterationNumber][2]
                for classifier in classifiers:
                    a = []
                    for attributeIndex in range(numAttributes):
                        if attributeIndex in classifier.specifiedAttList:
                            specifiedLocation = classifier.specifiedAttList.index(attributeIndex)
                            if classifier.conditionType[specifiedLocation] == 0: #if discrete
                                a.append(classifier.conditionDiscrete[specifiedLocation])
                            else: #if continuous
                                conditionCont = classifier.conditionContinuous[specifiedLocation] #cont array [min,max]
                                s = str(conditionCont[0])+","+str(conditionCont[1])
                                a.append(s)
                    else:
                        a.append("#")

                    if isinstance(classifier.phenotype,list):
                        s = str(classifier.phenotype[0])+","+str(classifier.phenotype[1])
                        a.append(s)
                    else:
                        a.append(classifier.phenotype)
                    a.append(classifier.fitness)
                    a.append(classifier.accuracy)
                    a.append(classifier.numerosity)
                    a.append(classifier.initTimeStamp)
                    a.append(len(classifier.specifiedAttList)/numAttributes)
                    writer.writerow(a)

        else:
            raise Exception("No Evaluation Data Exists for this iteration. If you want to have evaluation data for this iteration, make sure it was included in the learningCheckpoints param in eLCS")

    def exportFinalEvaluationToCSV(self,headerNames=np.array([]),className='phenotype'):
        self.exportEvaluationToCSV(sorted(self.evaluationDict.items())[len(self.evaluationDict.items())-1],headerNames,className)