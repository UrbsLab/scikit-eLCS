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
from Constants import Constants
from OfflineEnvironment import OfflineEnvironment
from ClassifierSet import *
from Prediction import *

class eLCS(BaseEstimator):

    def __init__(self,trainingFeatures,trainingPhenotypes,parameterNames,parameterValues):
        """Sets up eLCS model with default parameters from configFile, and training data
        """
        self.parameters = Constants()
        self.parameters = self.parameters.setConstants(parameterNames,parameterValues)


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

        env = OfflineEnvironment(X, y, self)
        self.parameters.referenceEnv(env)

        self.population = ClassifierSet()
        self.explorIter = 0
        self.correct = np.empty(self.parameters['learningIterations'])
        self.correct.fill(0)

        while self.explorIter < self.parameters['learningIterations']:

            #Get New Instance and Run a learning algorithm
            state_phenotype = self.parameters['env'].getTrainInstance()
            self.runIteration(state_phenotype,self.explorIter)

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

        #Make a Prediction
        prediction = Prediction(self,self.population)
        phenotypePrediction = prediction.getDecision()

        if phenotypePrediction == None or phenotypePrediction == 'Tie':
            if self.parameters['env'].formatData.discretePhenotype:
                phenotypePrediction = random.choice(self.parameters['env'].formatData.phenotypeList)
            else:
                phenotypePrediction = random.randrange(self.parameters['env'].formatData.phenotypeList[0],self.parameters['env'].formatData.phenotypeList[1],(self.parameters['env'].formatData.phenotypeList[1]-self.parameters['env'].formatData.phenotypeList[0])/float(1000))
        else:
            if self.parameters['env'].formatData.discretePhenotype:
                if phenotypePrediction == state_phenotype.phenotype:
                    self.correct[exploreIter] = 1
                else:
                    self.correct[exploreIter] = 0
            else:
                predictionError = math.fabs(phenotypePrediction-float(state_phenotype.phenotype))
                phenotypeRange = self.parameters['env'].formatData.phenotypeList[1] - self.parameters['env'].formatData.phenotypeList[0]
                accuracyEstimate = 1.0 - (predictionError / float(phenotypeRange))
                self.correct[exploreIter] = accuracyEstimate

        #Form [C]
        self.population.makeCorrectSet(state_phenotype.phenotype)

        #Update Parameters
        self.population.updateSets(self,exploreIter)

        #Perform Subsumption
        if self.parameters['doSubsumption']:
            self.population.doCorrectSetSubsumption(self)

        #Perform GA
        self.population.runGA(self,exploreIter,state_phenotype.attributeList,state_phenotype.phenotype)

        #Run Deletion
        self.population.deletion(self,exploreIter)

        #Clear [M] and [C]
        self.population.clearSets()
