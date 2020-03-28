import numpy as np

class Prediction():
    def __init__(self,elcs,population):
        self.decision = None
        self.probabilities = {}

        #Discrete Phenotypes
        if elcs.env.formatData.discretePhenotype:
            self.vote = {}
            self.tieBreak_Numerosity = {}
            self.tieBreak_TimeStamp = {}

            for eachClass in elcs.env.formatData.phenotypeList:
                self.vote[eachClass] = 0.0
                self.tieBreak_Numerosity[eachClass] = 0.0
                self.tieBreak_TimeStamp[eachClass] = 0.0

            for ref in population.matchSet:
                cl = population.popSet[ref]
                self.vote[cl.phenotype] += cl.fitness * cl.numerosity
                self.tieBreak_Numerosity[cl.phenotype] += cl.numerosity
                self.tieBreak_TimeStamp[cl.phenotype] += cl.initTimeStamp

            #Populate Probabilities
            sProb = 0
            for k,v in sorted(self.vote.items()):
                self.probabilities[k] = v
                sProb += v
            if sProb == 0: #In the case the match set doesn't exist
                for k, v in sorted(self.probabilities.items()):
                    self.probabilities[k] = 0
            else:
                for k,v in sorted(self.probabilities.items()):
                    self.probabilities[k] = v/sProb

            highVal = 0.0
            bestClass = []
            for thisClass in elcs.env.formatData.phenotypeList:
                if self.vote[thisClass] >= highVal:
                    highVal = self.vote[thisClass]

            for thisClass in elcs.env.formatData.phenotypeList:
                if self.vote[thisClass] == highVal:  # Tie for best class
                    bestClass.append(thisClass)

            if highVal == 0.0:
                self.decision = None

            elif len(bestClass) > 1:
                bestNum = 0
                newBestClass = []
                for thisClass in bestClass:
                    if self.tieBreak_Numerosity[thisClass] >= bestNum:
                        bestNum = self.tieBreak_Numerosity[thisClass]

                for thisClass in bestClass:
                    if self.tieBreak_Numerosity[thisClass] == bestNum:
                        newBestClass.append(thisClass)

                if len(newBestClass) > 1:
                    bestStamp = 0
                    newestBestClass = []
                    for thisClass in newBestClass:
                        if self.tieBreak_TimeStamp[thisClass] >= bestStamp:
                            bestStamp = self.tieBreak_TimeStamp[thisClass]

                    for thisClass in newBestClass:
                        if self.tieBreak_TimeStamp[thisClass] == bestStamp:
                            newestBestClass.append(thisClass)
                    # -----------------------------------------------------------------------
                    if len(newestBestClass) > 1:  # Prediction is completely tied - eLCS has no useful information for making a prediction
                        self.decision = 'Tie'
                else:
                    self.decision = newBestClass[0]
            else:
                self.decision = bestClass[0]

        #Continuous Phenotypes
        else:
            if len(population.matchSet) < 1:
                self.decision = None
            else:
                phenotypeRange = elcs.env.formatData.phenotypeList[1] - elcs.env.formatData.phenotypeList[0]
                predictionValue = 0
                valueWeightSum = 0
                for ref in population.matchSet:
                    cl = population.popSet[ref]
                    localRange = cl.phenotype[1] - cl.phenotype[0]
                    valueWeight = (phenotypeRange / float(localRange))
                    localAverage = cl.phenotype[1] + cl.phenotype[0] / 2.0

                    valueWeightSum += valueWeight
                    predictionValue += valueWeight * localAverage
                if valueWeightSum == 0.0:
                    self.decision = None
                else:
                    self.decision = predictionValue / float(valueWeightSum)

    def getFitnessSum(self, population, low, high):
        """ Get the fitness sum of rules in the rule-set. For continuous phenotype prediction. """
        fitSum = 0
        for ref in population.matchSet:
            cl = population.popSet[ref][0]
            if cl.phenotype[0] <= low and cl.phenotype[1] >= high:  # if classifier range subsumes segment range.
                fitSum += cl.fitness
        return fitSum

    def getDecision(self):
        """ Returns prediction decision. """
        return self.decision

    def getProbabilities(self):
        ''' Returns probabilities of each phenotype from the decision'''
        a = np.empty(len(sorted(self.probabilities.items())))
        counter = 0
        for k,v in sorted(self.probabilities.items()):
            a[counter] = v
            counter += 1
        return a