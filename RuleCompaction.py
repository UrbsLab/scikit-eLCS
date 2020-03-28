from skeLCS.eLCS import *
import numpy as np
import pandas as pd
from skeLCS.DataCleanup import StringEnumerator
from sklearn.model_selection import train_test_split
import copy

class RuleCompacter():
    def __init__(self,LCS,dataSet,classLabel,cv=3):

        converter = StringEnumerator(dataSet, classLabel)
        headers, classLabel, dataFeatures, dataPhenotypes = converter.getParams()

        formatted = np.insert(dataFeatures, dataFeatures.shape[1], dataPhenotypes, 1)
        np.random.shuffle(formatted)
        dataFeatures = np.delete(formatted, -1, axis=1)
        dataPhenotypes = formatted[:, -1]

        x_train,x_test,y_train,y_test = train_test_split(dataFeatures,dataPhenotypes,test_size=1/cv)

        self.trainingX = x_train
        self.trainingY = y_train
        self.testX = x_test
        self.testY = y_test
        self.cv = cv
        LCS.fit(self.trainingX,self.trainingY)

        self.baseTestScore = LCS.score(self.testX,self.testY)
        print("Original Score: "+str(self.baseTestScore))

        self.ranking = np.zeros(len(LCS.population.popSet))
        amountRemove = int(len(LCS.population.popSet) / 4)
        self.originalPopulation = copy.deepcopy(LCS.population.popSet)
        oLength = len(LCS.population.popSet)

        for i in range(oLength):

            removedIndexes = []
            self.population = copy.deepcopy(self.originalPopulation)
            if i + amountRemove <= oLength:
                for j in range(amountRemove):
                    self.population.pop(i)
                    removedIndexes.append(i+j)
            else:
                popEndAmount = oLength - i
                popBeginAmount = amountRemove - popEndAmount
                for j in range(popEndAmount):
                    self.population.pop(i)
                    removedIndexes.append(i+j)
                for j in range(popBeginAmount):
                    self.population.pop(0)
                    removedIndexes.append(j)

            LCS.population.popSet = copy.deepcopy(self.population)
            newScore = LCS.score(self.testX,self.testY)

            scoreDiff = self.baseTestScore - newScore
            for j in removedIndexes:
                self.ranking[j] += scoreDiff
        LCS.population.popSet = self.originalPopulation
        self.ranking /= amountRemove #Normalize rankings
        self.indexes = np.argsort(self.ranking) #indexes of worst to best rules

        print()
        self.worst = self.indexes[:int(oLength/2)]
        self.best = self.indexes[int(oLength*0.5):]

        newPopNoW = []
        for i in self.indexes:
            if i not in self.worst:
                newPopNoW.append(self.originalPopulation[i])

        LCS.population.popSet = newPopNoW
        newScore = LCS.score(self.testX, self.testY)
        print("Score after taking away worst: "+str(newScore))

        newPopNoB = []
        for i in self.indexes:
            if i not in self.best:
                newPopNoB.append(self.originalPopulation[i])

        LCS.population.popSet = newPopNoB
        newScore = LCS.score(self.testX, self.testY)
        print("Score after taking away best: "+str(newScore))

        LCS.population.popSet = copy.deepcopy(self.originalPopulation)


