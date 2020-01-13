import numpy as np
from DynamicNPArray import TupleArray

class Prediction():
    def __init__(self,elcs,population):
        self.decision = None

        #Discrete Phenotypes
        if elcs.env.formatData.discretePhenotype:
            self.vote = {}
            self.tieBreak_Numerosity = {}
            self.tieBreak_TimeStamp = {}

            for eachClass in elcs.env.formatData.phenotypeList.getArray():
                self.vote[eachClass] = 0.0
                self.tieBreak_Numerosity[eachClass] = 0.0
                self.tieBreak_TimeStamp[eachClass] = 0.0
            #print(population.matchSet)
            for ref in population.matchSet.getArray():
                cl = population.popSet.a[ref,0]
                self.vote[cl.phenotype] += cl.fitness * cl.numerosity
                self.tieBreak_Numerosity[cl.phenotype] += cl.numerosity
                self.tieBreak_TimeStamp[cl.phenotype] += cl.initTimeStamp

            highVal = 0.0
            bestClass = TupleArray(k=1)
            for thisClass in elcs.env.formatData.phenotypeList.getArray():
                if self.vote[thisClass] >= highVal:
                    highVal = self.vote[thisClass]

            for thisClass in elcs.env.formatData.phenotypeList.getArray():
                if self.vote[thisClass] == highVal:  # Tie for best class
                    bestClass.append(thisClass)

            if highVal == 0.0:
                self.decision = None

            elif bestClass.size() > 1:
                bestNum = 0
                newBestClass = TupleArray(k=1)
                for thisClass in bestClass.getArray():
                    if self.tieBreak_Numerosity[thisClass] >= bestNum:
                        bestNum = self.tieBreak_Numerosity[thisClass]

                for thisClass in bestClass.getArray():
                    if self.tieBreak_Numerosity[thisClass] == bestNum:
                        newBestClass.append(thisClass)

                if newBestClass.size() > 1:
                    bestStamp = 0
                    newestBestClass = TupleArray(k=1)
                    for thisClass in newBestClass.getArray():
                        if self.tieBreak_TimeStamp[thisClass] >= bestStamp:
                            bestStamp = self.tieBreak_TimeStamp[thisClass]

                    for thisClass in newBestClass.getArray():
                        if self.tieBreak_TimeStamp[thisClass] == bestStamp:
                            newestBestClass.append(thisClass)
                    # -----------------------------------------------------------------------
                    if newestBestClass.size() > 1:  # Prediction is completely tied - eLCS has no useful information for making a prediction
                        self.decision = 'Tie'
                else:
                    self.decision = newBestClass.a[0,0]
            else:
                self.decision = bestClass.a[0,0]

        #Continuous Phenotypes
        else:
            if population.matchSet.size() < 1:
                self.decision = None
            else:
                phenotypeRange = elcs.env.formatData.phenotypeList.a[1,0] - elcs.env.formatData.phenotypeList.a[0,0]
                predictionValue = 0
                valueWeightSum = 0
                for ref in population.matchSet.getArray():
                    cl = population.popSet.a[ref,0]
                    localRange = cl.phenotype.a[1,0] - cl.phenotype.a[0,0]
                    valueWeight = (phenotypeRange / float(localRange))
                    localAverage = cl.phenotype.a[1,0] + cl.phenotype.a[0,0] / 2.0

                    valueWeightSum += valueWeight
                    predictionValue += valueWeight * localAverage
                if valueWeightSum == 0.0:
                    self.decision = None
                else:
                    self.decision = predictionValue / float(valueWeightSum)

    def getFitnessSum(self, population, low, high):
        """ Get the fitness sum of rules in the rule-set. For continuous phenotype prediction. """
        fitSum = 0
        for ref in population.matchSet.getArray():
            cl = population.popSet.a[ref,0]
            if cl.phenotype.a[0,0] <= low and cl.phenotype.a[1,0] >= high:  # if classifier range subsumes segment range.
                fitSum += cl.fitness
        return fitSum

    def getDecision(self):
        """ Returns prediction decision. """
        return self.decision