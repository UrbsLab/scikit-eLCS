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
from sklearn.base import BaseEstimator
from sklearn.externals.joblib import Parallel, delayed
import pandas as pd
from OfflineEnvironment import OfflineEnvironment
from ClassifierSet import *
from Prediction import *

class eLCS(BaseEstimator):

    def __init__(self,learningIterations = 10000,N = 1000,p_spec=0.5,labelMissingData='NA',discreteAttributeLimit=10,nu=5,chi=0.8,upsilon=0.04,theta_GA=25,theta_del=20,theta_sub=20,acc_sub=0.99,beta=0.2,delta=0.1,init_fit=0.01,fitnessReduction=0.1,doSubsumption=1,selectionMethod='tournament',theta_sel='0.5'):
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

        self.env = OfflineEnvironment(X, y, self)

        self.population = ClassifierSet(self)
        self.explorIter = 0
        self.correct = np.empty(self.learningIterations)
        self.correct.fill(0)

        while self.explorIter < self.learningIterations:

            #Get New Instance and Run a learning algorithm
            state_phenotype = self.env.getTrainInstance()

            self.runIteration(state_phenotype,self.explorIter)

            #Incremenet Instance & Iteration
            self.explorIter+=1
            self.env.newInstance()
        #
        # return self

    def transform(self, X):
        """Not needed for eLCS"""
        return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    ##Helper Functions
    def runIteration(self,state_phenotype,exploreIter):
        #Form [M]
        self.population.makeMatchSet(state_phenotype,exploreIter,self)

        #Print [M]
        self.printMatchSet()

        # #Make a Prediction
        # prediction = Prediction(self,self.population)
        # phenotypePrediction = prediction.getDecision()
        #
        # if phenotypePrediction == None or phenotypePrediction == 'Tie':
        #     if self.env.formatData.discretePhenotype:
        #         phenotypePrediction = random.choice(self.env.formatData.phenotypeList)
        #     else:
        #         phenotypePrediction = random.randrange(self.env.formatData.phenotypeList[0],self.env.formatData.phenotypeList[1],(self.env.formatData.phenotypeList[1]-self.env.formatData.phenotypeList[0])/float(1000))
        # else:
        #     if self.env.formatData.discretePhenotype:
        #         if phenotypePrediction == state_phenotype.phenotype:
        #             self.correct[exploreIter] = 1
        #         else:
        #             self.correct[exploreIter] = 0
        #     else:
        #         predictionError = math.fabs(phenotypePrediction-float(state_phenotype.phenotype))
        #         phenotypeRange = self.env.formatData.phenotypeList[1] - self.env.formatData.phenotypeList[0]
        #         accuracyEstimate = 1.0 - (predictionError / float(phenotypeRange))
        #         self.correct[exploreIter] = accuracyEstimate
        #
        # #Form [C]
        # self.population.makeCorrectSet(self,state_phenotype.phenotype)
        #
        # #Update Parameters
        # self.population.updateSets(self,exploreIter)
        #
        # #Perform Subsumption
        # if self.doSubsumption:
        #     self.population.doCorrectSetSubsumption(self)
        #
        # #Perform GA
        # self.population.runGA(self,exploreIter,state_phenotype.attributeList,state_phenotype.phenotype)
        #
        # #Run Deletion
        # self.population.deletion(self,exploreIter)

        #Clear [M] and [C]
        self.population.clearSets()

    def printMatchSet(self):
        print("ITERATION:"+str(self.explorIter))
        print(self.population.matchSet.size)
        for classifierRef in self.population.matchSet:
            specifiedCounter = 0
            attributeCounter = 0

            for attribute in range(self.env.formatData.numAttributes):
                if attribute in self.population.popSet[classifierRef].specifiedAttList:
                    if self.env.formatData.attributeInfo[attributeCounter].type == 0:  # isDiscrete
                        print(self.population.popSet[classifierRef].condition[specifiedCounter].value, end="\t\t\t\t")
                    else:
                        print("[", end="")
                        print(
                            round(self.population.popSet[classifierRef].condition[specifiedCounter].list[0] * 10) / 10,
                            end=", ")
                        print(
                            round(self.population.popSet[classifierRef].condition[specifiedCounter].list[1] * 10) / 10,
                            end="")
                        print("]", end="\t\t")
                    specifiedCounter += 1
                else:
                    print("#", end="\t\t\t\t")
                attributeCounter += 1
            if self.env.formatData.discretePhenotype:
                print(self.population.popSet[classifierRef].phenotype)
            else:
                print("[", end="")
                print(round(self.population.popSet[classifierRef].phenotype[0] * 10) / 10, end=", ")
                print(round(self.population.popSet[classifierRef].phenotype[1] * 10) / 10, end="")
                print("]")
            print()
        print("________________________________________")