
from eLCS import *
import random
import copy
import sys
import numpy as np
from Classifier import *

class ClassifierSet:
    def __init__(self,elcs):
        #Major Parameters
        self.popSet = np.array([])
        self.matchSet = np.array([],dtype='int64')
        self.correctSet = np.array([],dtype='int64')
        self.microPopSize = 0

        #Evaluation Parameters
        self.aveGenerality = 0.0
        self.expRules = 0.0
        self.attributeSpecList = np.array([])
        self.attributeAccList = []
        self.avePhenotypeRange = 0.0

        self.makePop()

    def makePop(self):
        self.popSet = np.array([])

    def makeMatchSet(self,state_phenotype,exploreIter,elcs):
        state = state_phenotype.attributeList
        phenotype = state_phenotype.phenotype
        doCovering = True
        setNumerositySum = 0

        #Matching
        for i in range(len(self.popSet)):
            cl = self.popSet[i]
            if cl.match(state,elcs):
                self.matchSet = np.append(self.matchSet,i)
                setNumerositySum += cl.numerosity

                #Covering Check
                if elcs.parameters['env'].formatData.discretePhenotype:
                    if cl.phenotype == phenotype:
                        doCovering = False
                    else:
                        if float(cl.phenotype[0]) <= float(phenotype) <= float(cl.phenotype[1]):
                            doCovering = False

        #Covering
        while doCovering:
            newCl = Classifier(elcs,setNumerositySum+1,exploreIter,state,phenotype)
            self.addClassifierToPopulation(newCl,True)
            self.matchSet = np.append(self.matchSet,self.popSet.size - 1)
            doCovering = False

    def getIdenticalClassifier(self,newCl):
        for cl in self.popSet:
            if newCl.equals(cl):
                return cl
        return None

    def addClassifierToPopulation(self,cl,covering):
        oldCl = None
        if not covering:
            oldCl = self.getIdenticalClassifier(cl)
        if oldCl != None:
            oldCl.updateNumerosity(1)
            self.microPopSize += 1
        else:
            self.popSet = np.append(self.popSet,cl)
            self.microPopSize += 1

    def makeCorrectSet(self,elcs,phenotype):
        for i in range(self.matchSet.size):
            ref = self.matchSet[i]
            #Discrete Phenotype
            if elcs.parameters['env'].formatData.discretePhenotype:
                if self.popSet[ref].phenotype == phenotype:
                    self.correctSet = np.append(self.correctSet,ref)

            #Continuous Phenotype
            else:
                if float(phenotype) <= float(self.popSet[ref].phenotype[1]) and float(phenotype) >= float(self.popSet[ref].phenotype[0]):
                    self.correctSet = np.append(self.correctSet,ref)

    def updateSets(self,elcs,exploreIter):
        matchSetNumerosity = 0
        for ref in self.matchSet:
            matchSetNumerosity += self.popSet[ref].numerosity

        for ref in self.matchSet:
            self.popSet[ref].updateExperience()
            self.popSet[ref].updateMatchSetSize(elcs,matchSetNumerosity)
            if ref in self.correctSet:
                self.popSet[ref].updateCorrect()

            self.popSet[ref].updateAccuracy()
            self.popSet[ref].updateFitness(elcs)



    def doCorrectSetSubsumption(self,elcs):
        subsumer = None
        for ref in self.correctSet:
            cl = self.popSet[ref]
            if cl.isSubsumer(elcs):
                if subsumer == None or cl.isMoreGeneral(subsumer,elcs):
                    subsumer = cl

        if subsumer != None:
            i = 0
            while i < self.correctSet.size:
                ref = self.correctSet[i]
                if subsumer.isMoreGeneral(self.popSet[ref],elcs):
                    subsumer.updateNumerosity(self.popSet[ref].numerosity)
                    self.removeMacroClassifier(ref)
                    self.deleteFromMatchSet(ref)
                    self.deleteFromCorrectSet(ref)
                    i -= 1
                i+=1

    def removeMacroClassifier(self,ref):
        self.popSet = np.delete(self.popSet,ref)

    def deleteFromMatchSet(self,deleteRef):
        if deleteRef in self.matchSet:
            index = np.where(self.matchSet == deleteRef)[0][0]
            self.matchSet = np.delete(self.matchSet,index)

        for j in range(self.matchSet.size):
            ref = self.matchSet[j]
            if ref > deleteRef:
                self.matchSet[j] -= 1

    def deleteFromCorrectSet(self,deleteRef):
        if deleteRef in self.correctSet:
            index = np.where(self.correctSet == deleteRef)[0][0]
            self.correctSet = np.delete(self.correctSet,index)

        for j in range(self.correctSet.size):
            ref = self.correctSet[j]
            if ref > deleteRef:
                self.correctSet[j] -= 1

    def runGA(self,elcs,exploreIter,state,phenotype):
        #GA Run Requirement
        if (exploreIter - self.getIterStampAverage()) < elcs.parameters['theta_GA']:
            return

        self.setIterStamps(exploreIter)
        changed = False

        #Select Parents
        if elcs.parameters['selectionMethod'] == "roulette":
            selectList = self.selectClassifierRW()
            clP1 = selectList[0]
            clP2 = selectList[1]
        elif elcs.parameters['selectionMethod'] == "tournament":
            selectList = self.selectClassifierT(elcs)
            clP1 = selectList[0]
            clP2 = selectList[1]

        #Initialize Offspring
        cl1 = Classifier(clP1,exploreIter)
        if clP2 == None:
            cl2 = Classifier(clP1, exploreIter)
        else:
            cl2 = Classifier(clP2, exploreIter)

        #Crossover Operator (uniform crossover)
        if not cl1.equals(cl2) and random.random() < elcs.parameters['chi']:
            changed = cl1.uniformCrossover(elcs,cl2)

        #Initialize Key Offspring Parameters
        if changed:
            cl1.setAccuracy((cl1.accuracy + cl2.accuracy) / 2.0)
            cl1.setFitness(elcs.parameters['fitnessReduction'] * (cl1.fitness + cl2.fitness) / 2.0)
            cl2.setAccuracy(cl1.accuracy)
            cl2.setFitness(cl1.fitness)
        else:
            cl1.setFitness(elcs.parameters['fitnessReduction'] * cl1.fitness)
            cl2.setFitness(elcs.parameters['fitnessReduction'] * cl2.fitness)

        #Mutation Operator
        nowchanged = cl1.Mutation(elcs,state,phenotype)
        howaboutnow = cl2.Mutation(elcs,state,phenotype)

        #Add offspring to population
        if changed or nowchanged or howaboutnow:
            self.insertDiscoveredClassifiers(elcs,cl1, cl2, clP1, clP2, exploreIter)  # Subsumption


    def getIterStampAverage(self):
        sumCl = 0.0
        numSum = 0.0
        for i in range(self.correctSet.size):
            ref = self.correctSet[i]
            sumCl += self.popSet[ref].timeStampGA * self.popSet[ref].numerosity
            numSum += self.popSet[ref].numerosity
        return sumCl/float(numSum)

    def setIterStamps(self,exploreIter):
        for i in range(self.correctSet.size):
            ref = self.correctSet[i]
            self.popSet[ref].updateTimeStamp(exploreIter)

    def selectClassifierRW(self):
        setList = copy.deepcopy(self.correctSet)

        if setList.size > 2:
            selectList = np.array([None,None])
            currentCount = 0

            while currentCount < 2:
                fitSum = self.getFitnessSum(setList)

                choiceP = random.random() * fitSum
                i = 0
                sumCl = self.popSet[setList[i]].fitness
                while choiceP > sumCl:
                    i = i + 1
                    sumCl += self.popSet[setList[i]].fitness

                selectList[currentCount] = self.popSet[setList[i]]
                index = np.where(setList == setList[i])[0][0];
                setList = np.delete(setList,index)
                currentCount += 1

        elif setList.size == 2:
            selectList = np.array([self.popSet[setList[0]], self.popSet[setList[1]]])
        elif setList.size == 1:
            selectList = np.array([self.popSet[setList[0]], self.popSet[setList[0]]])

        return selectList

    def selectClassifierT(elcs,self):
        selectList = np.array([None, None])
        currentCount = 0
        setList = self.correctSet

        while currentCount < 2:
            tSize = int(setList.size * elcs.parameters['theta_sel'])

            #Select tSize elements from correctSet
            copyList = copy.deepcopy(self.correctSet)
            posList = np.array([])
            for i in range(tSize):
                choice = np.random.choice(copyList)
                index = np.where(c==choice)[0][0]
                posList = np.append(posList,choice)
                c = np.delete(c,index)

            bestF = 0
            bestC = self.correctSet[0]
            for j in posList:
                if self.popSet[j].fitness > bestF:
                    bestF = self.popSet[j].fitness
                    bestC = j

            selectList[currentCount] = self.popSet[bestC]
            currentCount += 1

        return selectList

    def insertDiscoveredClassifiers(self,elcs,cl1,cl2,clP1,clP2,exploreIter):
        if elcs.parameters['doSubsumption']:
            if cl1.specifiedAttList.size > 0:
                self.subsumeClassifier(elcs,cl1,clP1,clP2)
            if cl2.specifiedAttList.size > 0:
                self.subsumeClassifier(elcs,cl2, clP1, clP2)
        else:
            if cl1.specifiedAttList.size > 0:
                self.addClassifierToPopulation(cl1,False)
            if cl2.specifiedAttList.size > 0:
                self.addClassifierToPopulation(cl2, False)

    def subsumeClassifier(self,elcs,cl=None,cl1P=None,cl2P=None):
        if cl1P != None and cl1P.subsumes(elcs,cl):
            self.microPopSize += 1
            cl1P.updateNumerosity(1)
        elif cl2P != None and cl2P.subsumes(elcs,cl):
            self.microPopSize += 1
            cl2P.updateNumerosity(1)
        else:
            self.subsumeClassifier2(cl)  # Try to subsume in the correct set.

    def subsumeClassifier2(self,cl):
        choices = np.array([])
        for ref in self.correctSet:
            if self.popSet[ref].subsumes(cl):
                choices = np.append(choices,ref)

        if choices.size > 0:
            choice = int(random.random()*choices.size)
            self.popSet[choices[choice]].updateNumerosity(1)
            self.microPopSize += 1
            return
        self.addClassifierToPopulation(cl,False)

    def deletion(self,elcs,exploreIter):
        while (self.microPopSize > elcs.parameters['N']):
            self.deleteFromPopulation(elcs)

    def deleteFromPopulation(self,elcs):
        meanFitness = self.getPopFitnessSum() / float(self.microPopSize)

        sumCl = 0.0
        voteList = np.array([])
        for cl in self.popSet:
            vote = cl.getDelProp(elcs,meanFitness)
            sumCl += vote
            voteList = np.append(voteList,vote)

        choicePoint = sumCl * random.random()  # Determine the choice point

        newSum = 0.0
        for i in range(voteList.size):
            cl = self.popSet[i]
            newSum = newSum + voteList[i]
            if newSum > choicePoint:  # Select classifier for deletion
                # Delete classifier----------------------------------
                cl.updateNumerosity(-1)
                self.microPopSize -= 1
                if cl.numerosity < 1:  # When all micro-classifiers for a given classifier have been depleted.
                    self.removeMacroClassifier(i)
                    self.deleteFromMatchSet(i)
                    self.deleteFromCorrectSet(i)
                return
        return

    def getPopFitnessSum(self):
        """ Returns the sum of the fitnesses of all classifiers in the set. """
        sumCl = 0.0
        for cl in self.popSet:
            sumCl += cl.fitness * cl.numerosity
        return sumCl

    def clearSets(self):
        """ Clears out references in the match and correct sets for the next learning iteration. """
        self.matchSet = np.array([],dtype="int64")
        self.correctSet = np.array([],dtype="int64")