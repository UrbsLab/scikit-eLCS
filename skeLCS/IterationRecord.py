import csv
import numpy as np

class IterationRecord():
    '''
    IterationRecord Tracks 1 dictionary:
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
        11-number of deleted macroclassifiers performed (in iteration)
        12-total global time at end of iteration
        13-total matching time at end of iteration
        14-total deletion time at end of iteration
        15-total subsumption time at end of iteration
        16-total selection time at end of iteration
        17-total evaluation time at end of iteration
    '''

    def __init__(self):
        self.trackingDict = {}

    def addToTracking(self,iterationNumber,accuracy,avgPopGenerality,macroSize,microSize,mSize,cSize,iterAvg,
                      subsumptionCount,crossoverCount,mutationCount,coveringCount,deletionCount,
                      globalTime,matchingTime,deletionTime,subsumptionTime,selectionTime,evaluationTime):

        self.trackingDict[iterationNumber] = [accuracy,avgPopGenerality,macroSize,microSize,mSize,cSize,iterAvg,
                                   subsumptionCount,crossoverCount,mutationCount,coveringCount,deletionCount,
                                   globalTime,matchingTime,deletionTime,subsumptionTime,selectionTime,evaluationTime]

    def exportTrackingToCSV(self,filename='iterationData.csv'):
        #Exports each entry in Tracking Array as a column
        with open(filename,mode='w') as file:
            writer = csv.writer(file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)

            writer.writerow(["Iteration","Accuracy (approx)", "Average Population Generality","Macropopulation Size",
                             "Micropopulation Size", "Match Set Size", "Correct Set Size", "Average Iteration Age of Correct Set Classifiers",
                             "# Classifiers Subsumed in Iteration","# Crossover Operations Performed in Iteration","# Mutation Operations Performed in Iteration",
                             "# Covering Operations Performed in Iteration","# Deletion Operations Performed in Iteration",
                             "Total Global Time","Total Matching Time","Total Deletion Time","Total Subsumption Time","Total Selection Time","Total Evaluation Time"])

            for k,v in sorted(self.trackingDict.items()):
                writer.writerow([k,v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12],v[13],v[14],v[15],v[16],v[17]])
        file.close()

    def exportPop(self,elcs,headerNames=np.array([]),className='phenotype',filename='populationData.csv'):
        numAttributes = elcs.env.formatData.numAttributes

        headerNames = headerNames.tolist() #Convert to Python List

        #Default headerNames if none provided
        if len(headerNames) == 0:
            for i in range(numAttributes):
                headerNames.append("N"+str(i))

        if len(headerNames) != numAttributes:
            raise Exception("# of Header Names provided does not match the number of attributes in dataset instances.")

        with open(filename, mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            writer.writerow(headerNames+[className]+["Fitness","Accuracy","Numerosity","Avg Match Set Size","TimeStamp GA","Iteration Initialized","Specificity","Deletion Probability","Correct Count","Match Count"])
            classifiers = elcs.population.popSet
            for classifier in classifiers:
                a = []
                for attributeIndex in range(numAttributes):
                    if attributeIndex in classifier.specifiedAttList:
                        specifiedLocation = classifier.specifiedAttList.index(attributeIndex)
                        if not isinstance(classifier.condition[specifiedLocation],list): #if discrete
                            a.append(classifier.condition[specifiedLocation])
                        else: #if continuous
                            conditionCont = classifier.condition[specifiedLocation] #cont array [min,max]
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
                a.append(classifier.aveMatchSetSize)
                a.append(classifier.timeStampGA)
                a.append(classifier.initTimeStamp)
                a.append(len(classifier.specifiedAttList)/numAttributes)
                a.append(classifier.deletionProb)
                a.append(classifier.correctCount)
                a.append(classifier.matchCount)
                writer.writerow(a)
        file.close()

    def exportPopDCAL(self,elcs,headerNames=np.array([]),className='phenotype',filename='populationData.csv'):

        numAttributes = elcs.env.formatData.numAttributes

        headerNames = headerNames.tolist() #Convert to Python List

        #Default headerNames if none provided
        if len(headerNames) == 0:
            for i in range(numAttributes):
                headerNames.append("N"+str(i))

        if len(headerNames) != numAttributes:
            raise Exception("# of Header Names provided does not match the number of attributes in dataset instances.")

        with open(filename, mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            writer.writerow(["Specified Values","Specified Attribute Names"]+[className]+["Fitness","Accuracy","Numerosity","Avg Match Set Size","TimeStamp GA","Iteration Initialized","Specificity","Deletion Probability","Correct Count","Match Count"])

            classifiers = elcs.population.popSet
            for classifier in classifiers:
                a = []

                #Add attribute information
                headerString = ""
                valueString = ""
                for attributeIndex in range(numAttributes):
                    if attributeIndex in classifier.specifiedAttList:
                        specifiedLocation = classifier.specifiedAttList.index(attributeIndex)
                        headerString+=str(headerNames[attributeIndex])+", "
                        if not isinstance(classifier.condition[specifiedLocation],list): #if discrete
                            valueString+= str(classifier.condition[specifiedLocation])+", "
                        else: #if continuous
                            conditionCont = classifier.condition[specifiedLocation] #cont array [min,max]
                            s = "["+str(conditionCont[0])+","+str(conditionCont[1])+"]"
                            valueString+= s+", "

                a.append(valueString[:-2])
                a.append(headerString[:-2])

                #Add phenotype information
                if isinstance(classifier.phenotype, list):
                    s = str(classifier.phenotype[0]) + "," + str(classifier.phenotype[1])
                    a.append(s)
                else:
                    a.append(classifier.phenotype)

                #Add statistics
                a.append(classifier.fitness)
                a.append(classifier.accuracy)
                a.append(classifier.numerosity)
                a.append(classifier.aveMatchSetSize)
                a.append(classifier.timeStampGA)
                a.append(classifier.initTimeStamp)
                a.append(len(classifier.specifiedAttList) / numAttributes)
                a.append(classifier.deletionProb)
                a.append(classifier.correctCount)
                a.append(classifier.matchCount)
                writer.writerow(a)
        file.close()