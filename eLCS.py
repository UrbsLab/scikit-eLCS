'''
Name: eLCS.py
Authors: Robert Zhang in association with Ryan Urbanowicz
Contact: robertzh@wharton.upenn.edu
Description: This module implements the necessary methods for a Scikit-learn package: fit, transform, and fit_transform
for the eLCS training model.
'''

import numpy as np
import time
import warnings
import sys
import math
from sklearn.base import BaseEstimator
from sklearn.externals.joblib import Parallel, delayed
import pandas as pd
from OfflineEnvironment import OfflineEnvironment
from ClassifierSet import *
from Prediction import *
from Timer import *
from ClassAccuracy import *
import copy
from enum import Enum,auto

class eLCS(BaseEstimator):

    def __init__(self,learningIterations = 10000,trackingFrequency = 0, learningEvalCheckpoints = np.array([10000]),N = 1000,p_spec=0.5,labelMissingData='NA',discreteAttributeLimit=10,nu=5,chi=0.8,upsilon=0.04,theta_GA=25,theta_del=20,theta_sub=20,acc_sub=0.99,beta=0.2,delta=0.1,init_fit=0.01,fitnessReduction=0.1,doSubsumption=1,selectionMethod='tournament',theta_sel=0.5):
        """Sets up eLCS model with given parameters
        """
        self.learningIterations = learningIterations
        self.N = N
        self.p_spec = p_spec
        self.labelMissingData = labelMissingData
        self.discreteAttributeLimit = discreteAttributeLimit
        self.nu = nu
        self.chi = chi
        self.upsilon = upsilon
        self.theta_GA = theta_GA
        self.theta_del = theta_del
        self.theta_sub = theta_sub
        self.acc_sub = acc_sub
        self.beta = beta
        self.delta = delta
        self.init_fit = init_fit
        self.fitnessReduction = fitnessReduction
        self.doSubsumption = doSubsumption
        self.selectionMethod = selectionMethod
        self.theta_sel = theta_sel
        self.trackingFrequency = trackingFrequency
        self.learningCheckpoints = np.array([learningIterations])
        self.timer = Timer()
        self.trackingObjs = np.array([])
        self.popStatObjs = np.array([])
        self.dataHeaders = np.array([])
        self.explicitlyDiscreteAttributeIndexes = np.array([])
        self.explicitlyContinuousAttributeIndexes = np.array([])
        self.explicitPhenotype = ""

    def fit(self, X, y):
        """Scikit-learn required: Computes the feature importance scores from the training data.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels

        Returns
        __________
        self

        """

        # for i in range(X.shape[0]):
        #     for j in range(X.shape[1]):
        #         if (math.isnan(X[i,j])):
        #             X[i,j] = self.labelMissingData
        #
        # for i in range(y.shape[0]):
        #     if (math.isnan(y[i])):
        #         y[i] = self.labelMissingData


        self.env = OfflineEnvironment(X, y, self)

        if self.trackingFrequency == 0:
            self.trackingFrequency = self.env.formatData.numTrainInstances

        self.population = ClassifierSet(self)
        self.explorIter = 0
        self.correct = np.empty(self.trackingFrequency)
        self.correct.fill(0)

        while self.explorIter < self.learningIterations:
            #Get New Instance and Run a learning algorithm
            state_phenotype = self.env.getTrainInstance()

            self.runIteration(state_phenotype,self.explorIter)

            #Evaluations of Algorithm
            self.timer.startTimeEvaluation()

            if (self.explorIter%self.trackingFrequency) == (self.trackingFrequency-1) and self.explorIter > 0:
                self.population.runPopAveEval(self.explorIter,self)
                trackedAccuracy = np.sum(self.correct)/float(self.trackingFrequency)
                newObj = TrackingEvalObj(trackedAccuracy,self.explorIter,self.trackingFrequency,self)
                self.trackingObjs = np.append(self.trackingObjs,newObj)
            self.timer.stopTimeEvaluation()

            if (self.explorIter + 1) in self.learningCheckpoints:
                self.timer.startTimeEvaluation()
                self.population.runPopAveEval(self.explorIter,self)
                self.population.runAttGeneralitySum(True,self)
                self.env.startEvaluationMode()  #Preserves learning position in training data

                #Only a training file is available
                if self.env.formatData.discretePhenotype:
                    trainEval = self.doPopEvaluation()
                else:
                    trainEval = self.doContPopEvaluation()

                self.env.stopEvaluationMode()  # Returns to learning position in training data
                self.timer.stopTimeEvaluation()
                self.timer.returnGlobalTimer()
                newEvalObj = PopStatObj(trainEval,self.explorIter+1,self.population,self.correct,self)
                self.popStatObjs = np.append(self.popStatObjs,newEvalObj)

            #Incremenet Instance & Iteration
            self.explorIter+=1
            self.env.newInstance()
        #
        return self

    def score(self,X,y):
        return self.popStatObjs[self.popStatObjs.size-1].trainingAccuracy


    def transform(self, X):
        """Not needed for eLCS"""
        return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    ##Helper Functions
    def runIteration(self,state_phenotype,exploreIter):
        #print("ITERATION:"+str(self.explorIter))
        #print("Data Instance:" ,end=" ")
       # for i in range(state_phenotype.attributeList.size):
       #     print(state_phenotype.attributeList[i].value,end=" ")
       # print(" w/ Phenotype: ",state_phenotype.phenotype)
        #print("Population Set Size: "+str(self.population.popSet.size))

        #Form [M]
        self.population.makeMatchSet(state_phenotype,exploreIter,self)

        #Print [M]
        #self.printMatchSet()

        #Make a Prediction
        self.timer.startTimeEvaluation()
        prediction = Prediction(self,self.population)
        phenotypePrediction = prediction.getDecision()

        if phenotypePrediction == None or phenotypePrediction == 'Tie':
            if self.env.formatData.discretePhenotype:
                phenotypePrediction = random.choice(self.env.formatData.phenotypeList)
            else:
                phenotypePrediction = random.randrange(self.env.formatData.phenotypeList[0],self.env.formatData.phenotypeList[1],(self.env.formatData.phenotypeList[1]-self.env.formatData.phenotypeList[0])/float(1000))
        else:
            if self.env.formatData.discretePhenotype:
                if phenotypePrediction == state_phenotype.phenotype:
                    self.correct[exploreIter%self.trackingFrequency] = 1
                else:
                    self.correct[exploreIter%self.trackingFrequency] = 0
            else:
                predictionError = math.fabs(phenotypePrediction-float(state_phenotype.phenotype))
                phenotypeRange = self.env.formatData.phenotypeList[1] - self.env.formatData.phenotypeList[0]
                accuracyEstimate = 1.0 - (predictionError / float(phenotypeRange))
                self.correct[exploreIter%self.trackingFrequency] = accuracyEstimate

        self.timer.stopTimeEvaluation()

        #Form [C]
        self.population.makeCorrectSet(self,state_phenotype.phenotype)

        #Print [C]
        #self.printCorrectSet()

        #Update Parameters
        self.population.updateSets(self,exploreIter)

        #Perform Subsumption
        if self.doSubsumption:
            self.timer.startTimeSubsumption()
            self.population.doCorrectSetSubsumption(self)
            self.timer.stopTimeSubsumption()

        #Perform GA
        self.population.runGA(self,exploreIter,state_phenotype.attributeList,state_phenotype.phenotype)

        #Run Deletion
        self.population.deletion(self,exploreIter)

        #Clear [M] and [C]
        self.population.clearSets()

        #print("________________________________________")

    def printClassifier(self,classifier):
        attributeCounter = 0

        for attribute in range(self.env.formatData.numAttributes):
            if attribute in classifier.specifiedAttList:
                specifiedLocation = np.where(classifier.specifiedAttList == attribute)[0][0]
                if self.env.formatData.attributeInfo[attributeCounter].type == 0:  # isDiscrete
                    print(classifier.condition[specifiedLocation].value, end="\t\t\t\t")
                else:
                    print("[", end="")
                    print(
                        round(classifier.condition[specifiedLocation].list[0] * 10) / 10,
                        end=", ")
                    print(
                        round(classifier.condition[specifiedLocation].list[1] * 10) / 10,
                        end="")
                    print("]", end="\t\t")
            else:
                print("#", end="\t\t\t\t")
            attributeCounter += 1
        if self.env.formatData.discretePhenotype:
            print(classifier.phenotype,end="\t\t\t\t")
        else:
            print("[", end="")
            print(round(classifier.phenotype[0] * 10) / 10, end=", ")
            print(round(classifier.phenotype[1] * 10) / 10, end="")
            print("]",end="\t\t")
        print(classifier.fitness)

    def printMatchSet(self):
        print("Match Set Size: "+str(self.population.matchSet.size))
        for classifierRef in self.population.matchSet:
            self.printClassifier(self.population.popSet[classifierRef])
        print()

    def printCorrectSet(self):
        print("Correct Set Size: " + str(self.population.correctSet.size))
        for classifierRef in self.population.correctSet:
            self.printClassifier(self.population.popSet[classifierRef])
        print()

    def printPopSet(self):
        print("Population Set Size: " + str(self.population.popSet.size))
        for classifier in self.population.popSet:
            self.printClassifier(classifier)
        print()

    def printAccuratePopSet(self,threshold,exp):
        print("Population Set Size: " + str(self.population.popSet.size))
        for classifier in self.population.popSet:
            if classifier.fitness >= threshold and classifier.correctCount >= exp:
                self.printClassifier(classifier)

        print()

    def doPopEvaluation(self):
        noMatch = 0  # How often does the population fail to have a classifier that matches an instance in the data.
        tie = 0  # How often can the algorithm not make a decision between classes due to a tie.
        self.env.resetDataRef()  # Go to the first instance in dataset
        phenotypeList = self.env.formatData.phenotypeList

        classAccDict = {}
        for each in phenotypeList:
            classAccDict[each] = ClassAccuracy()
        # ----------------------------------------------

        instances = self.env.formatData.numTrainInstances

        # ----------------------------------------------------------------------------------------------
        for inst in range(instances):
            state_phenotype = self.env.getTrainInstance()
            self.population.makeEvalMatchSet(state_phenotype.attributeList,self)
            prediction = Prediction(self,self.population)
            phenotypeSelection = prediction.getDecision()

            if phenotypeSelection == None:
                noMatch += 1
            elif phenotypeSelection == 'Tie':
                tie += 1
            else:  # Instances which failed to be covered are excluded from the accuracy calculation
                for each in phenotypeList:
                    thisIsMe = False
                    accuratePhenotype = False
                    truePhenotype = state_phenotype.phenotype
                    if each == truePhenotype:
                        thisIsMe = True
                    if phenotypeSelection == truePhenotype:
                        accuratePhenotype = True
                    classAccDict[each].updateAccuracy(thisIsMe, accuratePhenotype)

            self.env.newInstance()  # next instance
            self.population.clearSets()

        # Calculate Standard Accuracy--------------------------------------------
        instancesCorrectlyClassified = classAccDict[phenotypeList[0]].T_myClass + classAccDict[phenotypeList[0]].T_otherClass
        instancesIncorrectlyClassified = classAccDict[phenotypeList[0]].F_myClass + classAccDict[phenotypeList[0]].F_otherClass
        standardAccuracy = float(instancesCorrectlyClassified) / float(instancesCorrectlyClassified + instancesIncorrectlyClassified)

        # Calculate Balanced Accuracy---------------------------------------------
        T_mySum = 0
        T_otherSum = 0
        F_mySum = 0
        F_otherSum = 0
        for each in phenotypeList:
            T_mySum += classAccDict[each].T_myClass
            T_otherSum += classAccDict[each].T_otherClass
            F_mySum += classAccDict[each].F_myClass
            F_otherSum += classAccDict[each].F_otherClass
        balancedAccuracy = ((0.5 * T_mySum / (float(T_mySum + F_otherSum)) + 0.5 * T_otherSum / (float(T_otherSum + F_mySum))))  # BalancedAccuracy = (Specificity + Sensitivity)/2

        # Adjustment for uncovered instances - to avoid positive or negative bias we incorporate the probability of guessing a phenotype by chance (e.g. 50% if two phenotypes)
        predictionFail = float(noMatch) / float(instances)
        predictionTies = float(tie) / float(instances)
        instanceCoverage = 1.0 - predictionFail
        predictionMade = 1.0 - (predictionFail + predictionTies)

        adjustedStandardAccuracy = (standardAccuracy * predictionMade) + ((1.0 - predictionMade) * (1.0 / float(len(phenotypeList))))
        adjustedBalancedAccuracy = (balancedAccuracy * predictionMade) + ((1.0 - predictionMade) * (1.0 / float(len(phenotypeList))))

        resultList = np.array([adjustedBalancedAccuracy,instanceCoverage])
        return resultList

    def doContPopEvaluation(self):
        noMatch = 0  # How often does the population fail to have a classifier that matches an instance in the data.
        accuracyEstimateSum = 0
        self.env.resetDataRef()  # Go to the first instance in dataset

        instances = self.env.formatData.numTrainInstances

        # ----------------------------------------------------------------------------------------------
        for inst in range(instances):
            state_phenotype = self.env.getTrainInstance()
            self.population.makeEvalMatchSet(state_phenotype.attributeList,self)
            prediction = Prediction(self, self.population)
            phenotypePrediction = prediction.getDecision()

            if phenotypePrediction == None:
                noMatch += 1
            else:
                predictionError = math.fabs(float(phenotypePrediction) - float(state_phenotype.phenotype))
                phenotypeRange = self.env.formatData.phenotypeList[1] - self.env.formatData.phenotypeList[0]
                accuracyEstimateSum += 1.0 - (predictionError / float(phenotypeRange))

            self.env.newInstance()  # next instance
            self.population.clearSets()

        # Accuracy Estimate
        if instances == noMatch:
            accuracyEstimate = 0
        else:
            accuracyEstimate = accuracyEstimateSum / float(instances - noMatch)

        # Adjustment for uncovered instances - to avoid positive or negative bias we incorporate the probability of guessing a phenotype by chance (e.g. 50% if two phenotypes)
        instanceCoverage = 1.0 - (float(noMatch) / float(instances))
        adjustedAccuracyEstimate = accuracyEstimateSum / float(instances)

        resultList = np.array([adjustedAccuracyEstimate, instanceCoverage])
        return resultList

    def preFit(self,csvFileName,missingDataLabel,classLabel,explicitlyDiscrete=np.array([]),explicitlyContinuous=np.array([]),explicitPhenotype = "n"):
        data = pd.read_csv(csvFileName, sep=',')  # Puts data from csv into indexable np arrays
        data = data.fillna(missingDataLabel)
        dataFeatures, dataPhenotypes, dataHeaders = data.drop(classLabel, axis=1).values, data[classLabel].values, data.drop(classLabel, axis=1).columns.values
        for d in explicitlyDiscrete:
            i = np.where(dataHeaders == d)
            self.explicitlyDiscreteAttributeIndexes = np.append(self.explicitlyDiscreteAttributeIndexes,i)
        for c in explicitlyContinuous:
            i = np.where(dataHeaders == c)
            self.explicitlyContinuousAttributeIndexes = np.append(self.explicitlyContinuousAttributeIndexes,i)
        self.dataHeaders = dataHeaders

        if explicitPhenotype == "c" or explicitPhenotype == "continuous":
            self.explicitPhenotype = "c"
        elif explicitPhenotype == "d" or explicitPhenotype == "discrete":
            self.explicitPhenotype = "d"
        else:
            self.explicitPhenotype = ""
        return dataFeatures,dataPhenotypes

class TrackingEvalObj():
    def __init__(self,accuracy,exploreIter,trackingFrequency,elcs):
        self.exploreIter = exploreIter
        self.popSetLength = elcs.population.popSet.size
        self.microPopSize = elcs.population.microPopSize
        self.accuracy = accuracy
        self.aveGenerality = elcs.population.aveGenerality
        self.time = elcs.timer.returnGlobalTimer()

class PopStatObj():
    def __init__(self,trainEval,exploreIter,pop,correct,elcs):
        self.trainingAccuracy = trainEval[0]
        self.trainingCoverage = trainEval[1]
        self.macroPopSize = pop.popSet.size
        self.microPopSize = pop.microPopSize
        self.aveGenerality = pop.aveGenerality
        self.attributeSpecList = copy.deepcopy(pop.attributeSpecList)
        self.attributeAccList = copy.deepcopy(pop.attributeAccList)
        self.times = copy.deepcopy(elcs.timer.reportTimes())
        self.correct = copy.deepcopy(correct)
