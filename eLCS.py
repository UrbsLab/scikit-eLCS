from OfflineEnvironment import OfflineEnvironment
from ClassifierSet import *
from Prediction import *
from Timer import *
from ClassAccuracy import *
import copy
import random
from sklearn.base import BaseEstimator
import numpy as np
import math

class eLCS(BaseEstimator):
    def __init__(self, learningIterations=10000, trackingFrequency=0, learningCheckpoints=np.array([1,10,50,100,200,500,700,1000]), N=1000,
                 p_spec=0.5, discreteAttributeLimit=10, specifiedAttributes = np.array([]), discretePhenotypeLimit=10,nu=5, chi=0.8, upsilon=0.04, theta_GA=25,
                 theta_del=20, theta_sub=20, acc_sub=0.99, beta=0.2, delta=0.1, init_fit=0.01, fitnessReduction=0.1,
                 doSubsumption=1, selectionMethod='tournament', theta_sel=0.5,randomSeed = "none"):

        self.learningIterations = learningIterations
        self.N = N
        self.p_spec = p_spec
        self.discreteAttributeLimit = discreteAttributeLimit #Can be number, or "c" or "d"
        self.discretePhenotypeLimit = discretePhenotypeLimit
        self.specifiedAttributes = specifiedAttributes #Must be array of indices

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

        self.learningCheckpoints = learningCheckpoints

        self.randomSeed = randomSeed

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

        #Parameter Checking
        if self.selectionMethod != "tournament" and self.selectionMethod != "roulette":
            raise Exception("Invalid selection type. Must be tournament or roulette")

        # Check Specified Headers and Discrete Attr List for Validity
        try:
            int(self.discreteAttributeLimit)
        except:
            if self.discreteAttributeLimit != "c" or self.discreteAttributeLimit != "d":
                raise Exception("Discrete Attribute Limit is invalid. Must be integer, 'c' or 'd'")
            else:
                numAttr = X.shape[1]
                for a in self.specifiedAttributes:
                    if a >= numAttr or a < 0:
                        raise Exception("Indexes for at least one specified attribute is out of bounds")

        if self.discretePhenotypeLimit != "c" and self.discretePhenotypeLimit != "d" and self.discretePhenotypeLimit != 10:
            raise Exception("Invalid discrete phenotype limit")

        if np.array_equal(self.learningCheckpoints,np.array([])):
            self.learningCheckpoints = np.array([self.learningIterations])
        else:
            if self.learningCheckpoints.min() > self.learningIterations:
                raise Exception("At least 1 learning evaluation checkpoint must be below the number of learning iterations")

        self.timer = Timer()
        self.trackingObjs = np.array([])
        self.popStatObjs = np.array([])

        if self.randomSeed != "none":
            try:
                int(self.randomSeed)
                random.seed(int(self.randomSeed))
            except:
                raise Exception("Random seed must be a number")

        #Check if X and Y are numeric
        try:
            for instance in X:
                for value in instance:
                    if not(np.isnan(value)):
                        float(value)
            for value in y:
                if not(np.isnan(value)):
                    float(value)

        except:
            raise Exception("X and y must be fully numeric")

        self.env = OfflineEnvironment(X,y,self)

        if self.trackingFrequency <= 0:
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
                if phenotypePrediction == state_phenotype[1]:
                    self.correct[exploreIter%self.trackingFrequency] = 1
                else:
                    self.correct[exploreIter%self.trackingFrequency] = 0
            else:
                predictionError = math.fabs(phenotypePrediction-float(state_phenotype[1]))
                phenotypeRange = self.env.formatData.phenotypeList[1] - self.env.formatData.phenotypeList[0]
                accuracyEstimate = 1.0 - (predictionError / float(phenotypeRange))
                self.correct[exploreIter%self.trackingFrequency] = accuracyEstimate

        self.timer.stopTimeEvaluation()

        #Form [C]
        self.population.makeCorrectSet(self,state_phenotype[1])

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
        self.population.runGA(self,exploreIter,state_phenotype[0],state_phenotype[1])

        #Run Deletion
        self.population.deletion(self,exploreIter)

        #Clear [M] and [C]
        self.population.clearSets()

        #print("________________________________________")

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
            self.population.makeEvalMatchSet(state_phenotype[0],self)
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
                    truePhenotype = state_phenotype[1]
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
            self.population.makeEvalMatchSet(state_phenotype[0],self)
            prediction = Prediction(self, self.population)
            phenotypePrediction = prediction.getDecision()

            if phenotypePrediction == None:
                noMatch += 1
            else:
                predictionError = math.fabs(float(phenotypePrediction) - float(state_phenotype[1]))
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
