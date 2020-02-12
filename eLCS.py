from OfflineEnvironment import OfflineEnvironment
from ClassifierSet import *
from Prediction import *
from Timer import *
from ClassAccuracy import *
import copy
import random
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import math
from DynamicNPArray import ArrayFactory

class eLCS(BaseEstimator,ClassifierMixin, RegressorMixin):
    def __init__(self, learningIterations=10000, trackingFrequency=0, learningCheckpoints=np.array([1,10,50,100,200,500,700,1000]), evalWhileFit = False, N=1000,
                 p_spec=0.5, discreteAttributeLimit=10, specifiedAttributes = np.array([]), discretePhenotypeLimit=10,nu=5, chi=0.8, upsilon=0.04, theta_GA=25,
                 theta_del=20, theta_sub=20, acc_sub=0.99, beta=0.2, delta=0.1, init_fit=0.01, fitnessReduction=0.1,
                 doSubsumption=1, selectionMethod='tournament', theta_sel=0.5,randomSeed = "none"):

        self.learningIterations = learningIterations
        self.N = N
        self.p_spec = p_spec
        self.discreteAttributeLimit = discreteAttributeLimit #Can be number, or "c" or "d"
        self.discretePhenotypeLimit = discretePhenotypeLimit
        self.specifiedAttributes = specifiedAttributes #Must be array of indices
        self.evalWhileFit = evalWhileFit

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

        ##Debugging Tools
        self.iterationTrackingObjs = []
        self.printPSet = False
        self.printMSet = False
        self.printCSet = False
        self.printPopSize = False
        self.printGAMech = False
        self.printMisc = False
        self.printSubCount = False
        self.printMicroPopSize = False
        self.printCrossOver = False
        self.printMutation = False
        self.printCovering = False
        self.printGACount = False
        self.printIterStampAvg = False
        self.printCSize = False
        self.printMSize = False

        self.subsumptionCounter = 0
        self.crossOverCounter = 0
        self.mutationCounter = 0
        self.coveringCounter = 0
        self.gaCounter = 0

    def fit(self, X, y):
        """Scikit-learn required: Computes the feature importance scores from the training data.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances
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
        self.trackingObjs = []
        self.popStatObjs = []

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

            if self.evalWhileFit:
                #Evaluations of Algorithm
                self.timer.startTimeEvaluation()

                if (self.explorIter%self.trackingFrequency) == (self.trackingFrequency-1) and self.explorIter > 0:
                    self.population.runPopAveEval(self.explorIter,self)
                    trackedAccuracy = np.sum(self.correct)/float(self.trackingFrequency)
                    newObj = TrackingEvalObj(trackedAccuracy,self.explorIter,self.trackingFrequency,self)
                    self.trackingObjs.append(newObj)

                if (self.explorIter + 1) in self.learningCheckpoints:
                    self.population.runPopAveEval(self.explorIter,self)
                    self.population.runAttGeneralitySum(True,self)
                    self.env.startEvaluationMode()  #Preserves learning position in training data

                    #Only a training file is available
                    if self.env.formatData.discretePhenotype:
                        trainEval = self.doPopEvaluation()
                    else:
                        trainEval = self.doContPopEvaluation()

                    self.env.stopEvaluationMode()  # Returns to learning position in training data
                    newEvalObj = PopStatObj(trainEval,self.explorIter+1,self.population,self.correct,self)
                    self.popStatObjs.append(newEvalObj)
                self.timer.stopTimeEvaluation()

            #Incremenet Instance & Iteration
            self.explorIter+=1
            self.env.newInstance()
        #
        return self

    def predict(self, X):
        self.timer.startTimeEvaluation()
        """Scikit-learn required: Computes the feature importance scores from the training data.

            Parameters
               ----------
            X: array-like {n_samples, n_features}
                Test instances to classify


            Returns
            __________
            y: array-like {n_samples}
                Classifications
        """
        instances = X.shape[0]
        predList = ArrayFactory.createArray(k=1)

        # ----------------------------------------------------------------------------------------------
        for inst in range(instances):
            state = X[inst]
            self.population.makeEvalMatchSet(state, self)
            prediction = Prediction(self, self.population)
            phenotypeSelection = prediction.getDecision()
            if phenotypeSelection == None or phenotypeSelection == "Tie":
                l = self.env.formatData.phenotypeList
                phenotypeSelection = random.choice(l)
            predList.append(phenotypeSelection) #What to do if None or Tie?
            self.population.clearSets(self)
        self.timer.stopTimeEvaluation()
        return predList.getArray()


    '''Having this score method is not mandatory, since the eLCS inherits from ClassifierMixin, which by default has a parent score method. When doing CV,
    you could pass in "scorer = 'balanced_accuracy'" or any other scorer function which would override the default ClassifierMixin method. This can all be done w/o
    the score method existing below.
    
    However, this score method acts as a replacement for the ClassifierMixin method (so technically, ClassifierMixin doesn't need to be a parent now), so by default,
    balanced accuracy is the scoring method. In the future, this scorer can be made to be more sophisticated whenever necessary. You can still pass in an external 
    scorer like above to override this scoring method as well if you want.
    
    '''
    #Commented out score function so that RegressorMixin and ClassifierMixin default methods can be used appropriately
    # def score(self,X,y):
    #     predList = self.predict(X)
    #     return balanced_accuracy_score(y, predList) #Make it balanced accuracy

    def transform(self, X):
        """Not needed for eLCS"""
        return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    ##Helper Functions
    def runIteration(self,state_phenotype,exploreIter):
        iterTrack = IterationTrackingObj()

        if self.printMisc:
            print("ITERATION:"+str(self.explorIter))
            print("Data Instance:" ,end=" ")
            for i in range(state_phenotype[0].size()):
                print(state_phenotype[0][i],end=" ")
            print(" w/ Phenotype: ",state_phenotype[1])
        if self.printPopSize:
            #print("Population Set Size: "+str(self.population.popSet.size))
            iterTrack.macroPopSize = len(self.population.popSet)
            print(len(self.population.popSet))
        if self.printSubCount:
            iterTrack.subsumptionCount = self.subsumptionCounter
            print(self.subsumptionCounter)
        if self.printMicroPopSize:
            iterTrack.microPopSize = self.population.microPopSize
            print(self.population.microPopSize)
        if self.printCovering:
            iterTrack.coveringCount = self.coveringCounter
            print(self.coveringCounter)
        if self.printCrossOver:
            iterTrack.crossOverCount = self.crossOverCounter
            print(self.crossOverCounter)
        if self.printMutation:
            iterTrack.mutationCount = self.mutationCounter
            print(self.mutationCounter)
        if self.printGACount:
            iterTrack.GACount = self.gaCounter
            print(self.gaCounter)

        #Print [P]
        if self.printPSet:
            self.printPopSet()

        #Form [M]
        self.timer.startTimeMatching()
        self.population.makeMatchSet(state_phenotype,exploreIter,self)
        self.timer.stopTimeMatching()

        #Print [M]
        if self.printMSet:
            self.printMatchSet()

        if self.evalWhileFit:
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
        if self.printCSet:
            self.printCorrectSet()

        if self.printCSize:
            print(len(self.population.correctSet))
            iterTrack.correctSetSize = len(self.population.correctSet)

        if self.printMSize:
            print(len(self.population.matchSet))
            iterTrack.matchSetSize = len(self.population.matchSet)

        #Update Parameters
        self.population.updateSets(self,exploreIter)

        #Perform Subsumption
        if self.doSubsumption:
            self.timer.startTimeSubsumption()
            self.population.doCorrectSetSubsumption(self)
            self.timer.stopTimeSubsumption()

        if self.printIterStampAvg:
            if len(self.population.correctSet) >= 1:
                iterTrack.iterStampAvg = self.population.getIterStampAverage()-exploreIter
            else:
                iterTrack.iterStampAvg = 0

        #Perform GA
        self.population.runGA(self,exploreIter,state_phenotype[0],state_phenotype[1])

        #Run Deletion
        self.timer.startTimeDeletion()
        self.population.deletion(self,exploreIter)
        self.timer.stopTimeDeletion()

        #Clear [M] and [C]
        self.population.clearSets(self)

        if self.printMisc:
            print("________________________________________")

        if self.evalWhileFit:
            self.iterationTrackingObjs.append(iterTrack)

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

        resultList = [adjustedBalancedAccuracy,instanceCoverage]
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
            self.population.clearSets(self)

        # Accuracy Estimate
        if instances == noMatch:
            accuracyEstimate = 0
        else:
            accuracyEstimate = accuracyEstimateSum / float(instances - noMatch)

        # Adjustment for uncovered instances - to avoid positive or negative bias we incorporate the probability of guessing a phenotype by chance (e.g. 50% if two phenotypes)
        instanceCoverage = 1.0 - (float(noMatch) / float(instances))
        adjustedAccuracyEstimate = accuracyEstimateSum / float(instances)

        resultList = [adjustedAccuracyEstimate, instanceCoverage]
        return resultList

    def printClassifier(self,classifier):
        attributeCounter = 0

        for attribute in range(self.env.formatData.numAttributes):
            if attribute in classifier.specifiedAttList:
                specifiedLocation = classifier.specifiedAttList.index(attribute)
                if self.env.formatData.attributeInfoType[attributeCounter] == 0:  # isDiscrete
                    print(classifier.conditionDiscrete[specifiedLocation], end="\t\t\t\t")
                else:
                    print("[", end="")
                    print(
                        round(classifier.conditionContinuous[specifiedLocation][0] * 10) / 10,
                        end=", ")
                    print(
                        round(classifier.conditionContinuous[specifiedLocation][1] * 10) / 10,
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
        if round(classifier.fitness*1000)/1000 != classifier.fitness:
            print(round(classifier.fitness*1000)/1000,end="\t\t")
        else:
            print(round(classifier.fitness * 1000) / 1000, end="\t\t\t")

        if round(classifier.accuracy * 1000) / 1000 != classifier.accuracy:
            print(round(classifier.accuracy*1000)/1000,end="\t\t")
        else:
            print(round(classifier.accuracy * 1000) / 1000, end="\t\t\t")
        print(classifier.numerosity)

    def printMatchSet(self):
        print("Match Set Size: "+str(len(self.population.matchSet)))
        for classifierRef in self.population.matchSet:
            self.printClassifier(self.population.popSet[classifierRef])
        print()

    def printCorrectSet(self):
        print("Correct Set Size: " + str(len(self.population.correctSet)))
        for classifierRef in self.population.correctSet:
            self.printClassifier(self.population.popSet[classifierRef])
        print()

    def printPopSet(self):
        print("Population Set Size: " + str(len(self.population.popSet)))
        for classifier in self.population.popSet:
            self.printClassifier(classifier)
        print()

class TrackingEvalObj():
    def __init__(self,accuracy,exploreIter,trackingFrequency,elcs):
        self.exploreIter = exploreIter
        self.popSetLength = len(elcs.population.popSet)
        self.microPopSize = elcs.population.microPopSize
        self.accuracy = accuracy
        self.aveGenerality = elcs.population.aveGenerality
        self.time = elcs.timer.returnGlobalTimer()

class PopStatObj():
    def __init__(self,trainEval,exploreIter,pop,correct,elcs):
        self.trainingAccuracy = trainEval[0]
        self.trainingCoverage = trainEval[1]
        self.macroPopSize = len(pop.popSet)
        self.microPopSize = pop.microPopSize
        self.aveGenerality = pop.aveGenerality
        self.attributeSpecList = copy.deepcopy(pop.attributeSpecList)
        self.attributeAccList = copy.deepcopy(pop.attributeAccList)
        self.times = copy.deepcopy(elcs.timer.reportTimes())
        self.correct = copy.deepcopy(correct)

class IterationTrackingObj():
    def __init__(self):
        self.macroPopSize = 0
        self.subsumptionCount = 0
        self.microPopSize = 0
        self.crossOverCount = 0
        self.mutationCount = 0
        self.coveringCount = 0
        self.GACount = 0
        self.iterStampAvg = 0
        self.correctSetSize = 0
        self.matchSetSize = 0