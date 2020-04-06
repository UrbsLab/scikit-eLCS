from skeLCS.OfflineEnvironment import OfflineEnvironment
from skeLCS.ClassifierSet import ClassifierSet
from skeLCS.Prediction import Prediction
from skeLCS.Timer import Timer
from skeLCS.ClassAccuracy import ClassAccuracy
from skeLCS.DynamicNPArray import ArrayFactory
from skeLCS.IterationRecord import IterationRecord

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import balanced_accuracy_score
import copy
import random
import numpy as np
import math

class eLCS(BaseEstimator,ClassifierMixin, RegressorMixin):
    def __init__(self, learningIterations=10000, trackingFrequency=0, learningCheckpoints=np.array([]), evalWhileFit = False, N=1000,
                 p_spec=0.5, discreteAttributeLimit=10, specifiedAttributes = np.array([]), discretePhenotypeLimit=10,nu=5, chi=0.8, upsilon=0.04, theta_GA=25,
                 theta_del=20, theta_sub=20, acc_sub=0.99, beta=0.2, delta=0.1, init_fit=0.01, fitnessReduction=0.1,
                 doSubsumption=True, selectionMethod='tournament', theta_sel=0.5,randomSeed = "none",matchForMissingness=False):

        '''
        :param learningIterations:      Must be nonnegative integer. The number of training cycles to run.
        :param trackingFrequency:       Must be nonnegative integer. Relevant only if evalWhileFit param is true. Conducts accuracy approximations and population measurements every trackingFrequency iterations.
                                        If param == 0, tracking done once every epoch.
        :param learningCheckpoints:     Must be ndarray of nonnegative integers. Relevant only if evalWhileFit param is true. Conducts detailed evaluation of model performance by finding precise training accuracy
                                        at the specified iteration count. Iterations are 0 indexed.
        :param evalWhileFit:            Must be boolean. Determines if live tracking and evaluation is done during model training
        :param N:                       Must be nonnegative integer. Maximum micro classifier population size (sum of classifier numerosities).
        :param p_spec:                  Must be float from 0 - 1. Probability of specifying an attribute during the covering procedure. Advised: larger amounts of attributes => lower p_spec values
        :param discreteAttributeLimit:  Must be nonnegative integer OR "c" OR "d". Multipurpose param. If it is a nonnegative integer, discreteAttributeLimit determines the threshold that determines
                                        if an attribute will be treated as a continuous or discrete attribute. For example, if discreteAttributeLimit == 10, if an attribute has more than 10 unique
                                        values in the dataset, the attribute will be continuous. If the attribute has 10 or less unique values, it will be discrete. Alternatively,
                                        discreteAttributeLimit can take the value of "c" or "d". See next param for this.
        :param specifiedAttributes:     Must be an ndarray type of nonnegative integer attributeIndices (zero indexed).
                                        If "c", attributes specified by index in this param will be continuous and the rest will be discrete. If "d", attributes specified by index in this
                                        param will be discrete and the rest will be continuous.
                                        If this value is given, and discreteAttributeLimit is not "c" or "d", discreteAttributeLimit overrides this specification
        :param discretePhenotypeLimit:  Must be nonnegative integer OR "c" OR "d". Works similarly to discreteAttributeLimit. Multipurpose param. If it is a nonnegative integer, this param determines the
                                        continuous/discrete threshold for the phenotype. If it is "c" or "d", the phenotype is explicitly defined as continuous or discrete.
        :param nu:                      (v) Must be a float. Power parameter used to determine the importance of high accuracy when calculating fitness. (typically set to 5, recommended setting of 1 in noisy data)
        :param chi:                     (X) Must be float from 0 - 1. The probability of applying crossover in the GA. (typically set to 0.5-1.0)
        :param upsilon:                 (u) Must be float from 0 - 1. The probability of mutating an allele within an offspring.(typically set to 0.01-0.05)
        :param theta_GA:                Must be nonnegative float. The GA threshold. The GA is applied in a set when the average time (# of iterations) since the last GA in the correct set is greater than theta_GA.
        :param theta_del:               Must be a nonnegative integer. The deletion experience threshold; The calculation of the deletion probability changes once this threshold is passed.
        :param theta_sub:               Must be a nonnegative integer. The subsumption experience threshold
        :param acc_sub:                 Must be float from 0 - 1. Subsumption accuracy requirement
        :param beta:                    Must be float. Learning parameter; Used in calculating average correct set size
        :param delta:                   Must be float. Deletion parameter; Used in determining deletion vote calculation.
        :param init_fit:                Must be float. The initial fitness for a new classifier. (typically very small, approaching but not equal to zero)
        :param fitnessReduction:        Must be float. Initial fitness reduction in GA offspring rules.
        :param doSubsumption:           Must be boolean. Determines if subsumption is done in the learning process.
        :param selectionMethod:         Must be either "tournament" or "roulette". Determines GA selection method. Recommended: tournament
        :param theta_sel:               Must be float from 0 - 1. The fraction of the correct set to be included in tournament selection.
        :param randomSeed:              Must be an integer or "none". Set a constant random seed value to some integer (in order to obtain reproducible results). Put 'none' if none (for pseudo-random algorithm runs).
        :param matchForMissingness:     Must be boolean. Determines if eLCS matches for missingness (i.e. if a missing value can match w/ a specified value)
        '''

        '''
        Parameter Validity Checking
        Checks all parameters for valid values
        '''
        #learningIterations
        if not self.checkIsInt(learningIterations):
            raise Exception("learningIterations param must be nonnegative integer")

        if learningIterations < 0:
            raise Exception("learningIterations param must be nonnegative integer")

        #trackingFrequency
        if not self.checkIsInt(trackingFrequency):
            raise Exception("trackingFrequency param must be nonnegative integer")

        if trackingFrequency < 0:
            raise Exception("trackingFrequency param must be nonnegative integer")

        #learningCheckpoints
        if not (isinstance(learningCheckpoints,np.ndarray)):
            raise Exception("learningCheckpoints param must be ndarray")

        for learningCheckpt in learningCheckpoints:
            if not self.checkIsInt(learningCheckpt):
                raise Exception("All learningCheckpoints elements param must be nonnegative integers")
            if int(learningCheckpt) < 0:
                raise Exception("All learningCheckpoints elements param must be nonnegative integers")


        #evalWhileFit
        if not(isinstance(evalWhileFit,bool)):
            raise Exception("evalWhileFit param must be boolean")

        #N
        if not self.checkIsInt(N):
            raise Exception("N param must be nonnegative integer")

        if N < 0:
            raise Exception("N param must be nonnegative integer")

        #p_spec
        if not self.checkIsFloat(p_spec):
            raise Exception("p_spec param must be float from 0 - 1")

        if p_spec < 0 or p_spec > 1:
            raise Exception("p_spec param must be float from 0 - 1")

        #discreteAttributeLimit
        if discreteAttributeLimit != "c" and discreteAttributeLimit != "d":
            try:
                dpl = int(discreteAttributeLimit)
                if not self.checkIsInt(discreteAttributeLimit):
                    raise Exception("discreteAttributeLimit param must be nonnegative integer or 'c' or 'd'")
                if dpl < 0:
                    raise Exception("discreteAttributeLimit param must be nonnegative integer or 'c' or 'd'")
            except:
                raise Exception("discreteAttributeLimit param must be nonnegative integer or 'c' or 'd'")

        #specifiedAttributes
        if not (isinstance(specifiedAttributes,np.ndarray)):
            raise Exception("specifiedAttributes param must be ndarray")

        for spAttr in specifiedAttributes:
            if not self.checkIsInt(spAttr):
                raise Exception("All specifiedAttributes elements param must be nonnegative integers")
            if int(spAttr) < 0:
                raise Exception("All specifiedAttributes elements param must be nonnegative integers")

        #discretePhenotypeLimit
        if discretePhenotypeLimit != "c" and discretePhenotypeLimit != "d":
            try:
                dpl = int(discretePhenotypeLimit)
                if not self.checkIsInt(discretePhenotypeLimit):
                    raise Exception("discreteAttributeLimit param must be nonnegative integer or 'c' or 'd'")
                if dpl < 0:
                    raise Exception("discretePhenotypeLimit param must be nonnegative integer or 'c' or 'd'")
            except:
                raise Exception("discretePhenotypeLimit param must be nonnegative integer or 'c' or 'd'")

        #nu
        if not self.checkIsFloat(nu):
            raise Exception("nu param must be float")

        #chi
        if not self.checkIsFloat(chi):
            raise Exception("chi param must be float from 0 - 1")

        if chi < 0 or chi > 1:
            raise Exception("chi param must be float from 0 - 1")

        #upsilon
        if not self.checkIsFloat(upsilon):
            raise Exception("upsilon param must be float from 0 - 1")

        if upsilon < 0 or upsilon > 1:
            raise Exception("upsilon param must be float from 0 - 1")

        #theta_GA
        if not self.checkIsFloat(theta_GA):
            raise Exception("theta_GA param must be nonnegative float")

        if theta_GA < 0:
            raise Exception("theta_GA param must be nonnegative float")

        #theta_del
        if not self.checkIsInt(theta_del):
            raise Exception("theta_del param must be nonnegative integer")

        if theta_del < 0:
            raise Exception("theta_del param must be nonnegative integer")

        #theta_sub
        if not self.checkIsInt(theta_sub):
            raise Exception("theta_sub param must be nonnegative integer")

        if theta_sub < 0:
            raise Exception("theta_sub param must be nonnegative integer")

        #acc_sub
        if not self.checkIsFloat(acc_sub):
            raise Exception("acc_sub param must be float from 0 - 1")

        if acc_sub < 0 or acc_sub > 1:
            raise Exception("acc_sub param must be float from 0 - 1")

        #beta
        if not self.checkIsFloat(beta):
            raise Exception("beta param must be float")

        #delta
        if not self.checkIsFloat(delta):
            raise Exception("delta param must be float")

        #init_fit
        if not self.checkIsFloat(init_fit):
            raise Exception("init_fit param must be float")

        #fitnessReduction
        if not self.checkIsFloat(fitnessReduction):
            raise Exception("fitnessReduction param must be float")

        #doSubsumption
        if not(isinstance(doSubsumption,bool)):
            raise Exception("doSubsumption param must be boolean")

        #selectionMethod
        if selectionMethod != "tournament" and selectionMethod != "roulette":
            raise Exception("selectionMethod param must be 'tournament' or 'roulette'")

        #theta_sel
        if not self.checkIsFloat(theta_sel):
            raise Exception("theta_sel param must be float from 0 - 1")

        if theta_sel < 0 or theta_sel > 1:
            raise Exception("theta_sel param must be float from 0 - 1")

        #randomSeed
        if randomSeed != "none":
            try:
                if not self.checkIsInt(randomSeed):
                    raise Exception("randomSeed param must be integer or 'none'")
                random.seed(int(randomSeed))
                np.random.seed(int(randomSeed))
            except:
                raise Exception("randomSeed param must be integer or 'none'")

        #matchForMissingness
        if not (isinstance(matchForMissingness, bool)):
            raise Exception("matchForMissingness param must be boolean")

        '''
        Set params
        '''
        self.learningIterations = learningIterations
        self.N = N
        self.p_spec = p_spec
        self.discreteAttributeLimit = discreteAttributeLimit
        self.discretePhenotypeLimit = discretePhenotypeLimit
        self.specifiedAttributes = specifiedAttributes
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
        self.matchForMissingness = matchForMissingness

        '''
        Set tracking tools
        '''
        self.trackingObj = tempTrackingObj()
        self.record = IterationRecord()
        self.hasTrained = False
        self.evalWhileFitAfter = self.evalWhileFit

    def checkIsInt(self,num):
        try:
            n = float(num)
            if num - int(num) == 0:
                return True
            else:
                return False
        except:
            return False

    def checkIsFloat(self,num):
        try:
            n = float(num)
            return True
        except:
            return False

    def fit(self, X, y):
        """Scikit-learn required: Supervised training of eLCS

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances. ALL INSTANCE ATTRIBUTES MUST BE NUMERIC or NAN
        y: array-like {n_samples}
            Training labels. ALL INSTANCE PHENOTYPES MUST BE NUMERIC NOT NAN OR OTHER TYPE

        Returns
        __________
        self
        """
        #If trained already, raise Exception
        if self.hasTrained:
            raise Exception("Cannot train already trained model again")

        # Check if X and Y are numeric
        try:
            for instance in X:
                for value in instance:
                    if not (np.isnan(value)):
                        float(value)
            for value in y:
                float(value)

        except:
            raise Exception("X and y must be fully numeric")

        #Set up environment
        self.env = OfflineEnvironment(X,y,self)

        if not self.env.formatData.discretePhenotype:
            raise Exception("eLCS works best with classification problems. While we have the infrastructure to support continuous phenotypes, we have disabled it for this version.")

        # Modify certain params to default values
        if not (self.learningIterations - 1 in self.learningCheckpoints):
            self.learningCheckpoints = np.append(self.learningCheckpoints,self.learningIterations - 1)

        if self.trackingFrequency == 0:
            self.trackingFrequency = self.env.formatData.numTrainInstances

        self.timer = Timer()
        self.population = ClassifierSet(self)
        self.explorIter = 0
        self.correct = np.empty(self.trackingFrequency)
        self.correct.fill(0)

        while self.explorIter < self.learningIterations:
            #Get New Instance and Run a learning algorithm
            state_phenotype = self.env.getTrainInstance()

            self.runIteration(state_phenotype,self.explorIter)


            #Evaluations of Algorithm
            if not self.hasTrained:
                self.timer.startTimeEvaluation()

            if (((self.explorIter%self.trackingFrequency) == (self.trackingFrequency-1) and self.explorIter > 0) or self.explorIter == self.learningIterations-1) and self.evalWhileFit:
                self.population.runPopAveEval(self.explorIter,self)
                trackedAccuracy = np.sum(self.correct)/float(self.trackingFrequency)
                if not self.hasTrained:
                    self.timer.returnGlobalTimer()
                self.record.addToTracking(self.explorIter,trackedAccuracy,self.population.aveGenerality,
                                            self.trackingObj.macroPopSize,self.trackingObj.microPopSize,
                                            self.trackingObj.matchSetSize,self.trackingObj.correctSetSize,
                                            self.trackingObj.avgIterAge, self.trackingObj.subsumptionCount,
                                            self.trackingObj.crossOverCount, self.trackingObj.mutationCount,
                                            self.trackingObj.coveringCount,self.trackingObj.deletionCount,
                                            self.timer.globalTime,self.timer.globalMatching,
                                            self.timer.globalDeletion,self.timer.globalSubsumption,
                                            self.timer.globalSelection,self.timer.globalEvaluation)
            else: #If not detailed track, record regular easy to track data every iteration
                if not self.hasTrained:
                    self.timer.returnGlobalTimer()
                self.record.addToTracking(self.explorIter, "", "",
                                            self.trackingObj.macroPopSize, self.trackingObj.microPopSize,
                                            self.trackingObj.matchSetSize, self.trackingObj.correctSetSize,
                                            self.trackingObj.avgIterAge, self.trackingObj.subsumptionCount,
                                            self.trackingObj.crossOverCount, self.trackingObj.mutationCount,
                                            self.trackingObj.coveringCount, self.trackingObj.deletionCount,
                                            self.timer.globalTime, self.timer.globalMatching,
                                            self.timer.globalDeletion, self.timer.globalSubsumption,
                                            self.timer.globalSelection, self.timer.globalEvaluation)

            if self.evalWhileFit:
                if (self.explorIter) in self.learningCheckpoints: #0 indexed learning Checkpoints
                    self.population.runPopAveEval(self.explorIter,self)
                    self.population.runAttGeneralitySum(True,self)
                    self.env.startEvaluationMode()  #Preserves learning position in training data

                    if self.env.formatData.discretePhenotype:
                        trainEval = self.doPopEvaluation()
                    else:
                        trainEval = self.doContPopEvaluation()

                    self.record.addToEval(self.explorIter,trainEval[0],trainEval[1],copy.deepcopy(self.population.popSet),copy.deepcopy(self.population.attributeSpecList),copy.deepcopy(self.population.attributeAccList))

                    self.env.stopEvaluationMode()  # Returns to learning position in training data
            if not self.hasTrained:
                self.timer.stopTimeEvaluation()

            #Incremenet Instance & Iteration
            self.explorIter+=1
            self.env.newInstance()
        self.hasTrained = True
        return self

    def predict_proba(self, X):
        if not self.hasTrained:
            self.timer.startTimeEvaluation()
        """Scikit-learn required: Test Accuracy of eLCS

            Parameters
            ----------
            X: array-like {n_samples, n_features}
                Test instances to classify. ALL INSTANCE ATTRIBUTES MUST BE NUMERIC


            Returns
            __________
            y: array-like {n_samples}
                Classifications.
        """

        try:
            for instance in X:
                for value in instance:
                    if not (np.isnan(value)):
                        float(value)
        except:
            raise Exception("X must be fully numeric")

        instances = X.shape[0]
        predList = ArrayFactory.createArray(k=len(self.env.formatData.phenotypeList))

        # ----------------------------------------------------------------------------------------------
        for inst in range(instances):
            state = X[inst]
            self.population.makeEvalMatchSet(state, self)
            prediction = Prediction(self, self.population)
            probs = prediction.getProbabilities()
            predList.append(probs)
            self.population.clearSets(self)
        if not self.hasTrained:
            self.timer.stopTimeEvaluation()
        return predList.getArray()

    def predict(self, X):
        if not self.hasTrained:
            self.timer.startTimeEvaluation()
        """Scikit-learn required: Test Accuracy of eLCS

            Parameters
            ----------
            X: array-like {n_samples, n_features}
                Test instances to classify. ALL INSTANCE ATTRIBUTES MUST BE NUMERIC


            Returns
            __________
            y: array-like {n_samples}
                Classifications.
        """

        try:
            for instance in X:
                for value in instance:
                    if not (np.isnan(value)):
                        float(value)
        except:
            raise Exception("X must be fully numeric")

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
        if not self.hasTrained:
            self.timer.stopTimeEvaluation()
        return predList.getArray()


    '''Having this score method is not mandatory, since the eLCS inherits from ClassifierMixin, which by default has a parent score method. When doing CV,
    you could pass in "scorer = 'balanced_accuracy'" or any other scorer function which would override the default ClassifierMixin method. This can all be done w/o
    the score method existing below.
    
    However, this score method acts as a replacement for the ClassifierMixin method (so technically, ClassifierMixin doesn't need to be a parent now), so by default,
    balanced accuracy is the scoring method. In the future, this scorer can be made to be more sophisticated whenever necessary. You can still pass in an external 
    scorer like above to override this scoring method as well if you want.
    
    '''
    #Commented out score function if continuous phenotype is built in, so that RegressorMixin and ClassifierMixin default methods can be used appropriately
    def score(self,X,y):
        predList = self.predict(X)
        return balanced_accuracy_score(y, predList) #Make it balanced accuracy

    def transform(self, X):
        """Not needed for eLCS"""
        return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    ##Helper Functions
    def runIteration(self,state_phenotype,exploreIter):
        #Reset tracking object counters
        self.trackingObj.resetAll()

        #Form [M]
        self.population.makeMatchSet(state_phenotype,exploreIter,self)

        if self.evalWhileFit:
            #Make a Prediction
            if not self.hasTrained:
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
            if not self.hasTrained:
                self.timer.stopTimeEvaluation()

        #Form [C]
        self.population.makeCorrectSet(self,state_phenotype[1])

        #Update Parameters
        self.population.updateSets(self,exploreIter)

        #Perform Subsumption
        if self.doSubsumption:
            if not self.hasTrained:
                self.timer.startTimeSubsumption()
            self.population.doCorrectSetSubsumption(self)
            if not self.hasTrained:
                self.timer.stopTimeSubsumption()

        #Perform GA
        self.population.runGA(self,exploreIter,state_phenotype[0],state_phenotype[1])

        #Run Deletion
        if not self.hasTrained:
            self.timer.startTimeDeletion()
        self.population.deletion(self,exploreIter)
        if not self.hasTrained:
            self.timer.stopTimeDeletion()

        self.trackingObj.macroPopSize = len(self.population.popSet)
        self.trackingObj.microPopSize = self.population.microPopSize
        self.trackingObj.matchSetSize = len(self.population.matchSet)
        self.trackingObj.correctSetSize = len(self.population.correctSet)
        self.trackingObj.avgIterAge = self.population.getIterStampAverage()

        #Clear [M] and [C]
        self.population.clearSets(self)

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
            self.population.clearSets(self)

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

    def exportIterationTrackingDataToCSV(self,filename='iterationData.csv'):
        if self.hasTrained:
            self.record.exportTrackingToCSV(filename)
        else:
            raise Exception("There is no tracking data to export, as the eLCS model has not been trained")

    '''
    If evalWhiteFit was turned off, as long as the iterationNumber is the final iteration, an immediate evaluation will be run
    on the population and an export will be made. Past unsaved rule populations are not obviously not valid for evaluation or export.
    '''
    def exportRulePopulationAtIterationToCSV(self,iterationNumber,headerNames=np.array([]),className='phenotype',filename='populationData.csv',ALKR=False):
        if self.evalWhileFitAfter or iterationNumber != self.learningIterations - 1:
            if ALKR:
                self.record.exportEvaluationToCSVALKR(self, iterationNumber, headerNames, className,filename)
            else:
                self.record.exportEvaluationToCSV(self, iterationNumber, headerNames, className, filename)
        else:
            self.exportFinalRulePopulationToCSV(headerNames,className,filename,ALKR)

    '''
    Even if evalWhileFit was turned off, this will run an immediate evaluation and export it.
    '''
    def exportFinalRulePopulationToCSV(self,headerNames=np.array([]),className="phenotype",filename='populationData.csv',ALKR=False):
        if self.hasTrained:
            if self.evalWhileFitAfter:
                if ALKR:
                    self.record.exportFinalRulePopulationToCSVALKR(self,headerNames,className,filename)
                else:
                    self.record.exportFinalRulePopulationToCSV(self, headerNames, className, filename)
            else:
                self.population.runPopAveEval(self.explorIter, self)
                self.population.runAttGeneralitySum(True, self)
                self.env.startEvaluationMode()  # Preserves learning position in training data

                if self.env.formatData.discretePhenotype:
                    trainEval = self.doPopEvaluation()
                else:
                    trainEval = self.doContPopEvaluation()

                self.record.addToEval(self.explorIter-1, trainEval[0], trainEval[1], copy.deepcopy(self.population.popSet),copy.deepcopy(self.population.attributeSpecList),copy.deepcopy(self.population.attributeAccList))
                self.evalWhileFitAfter = True #So it doesn't run this else again if this is invoked again
                self.env.stopEvaluationMode()  # Returns to learning position in training data
                if ALKR:
                    self.record.exportFinalRulePopulationToCSVALKR(self,headerNames, className,filename)
                else:
                    self.record.exportFinalRulePopulationToCSV(self, headerNames, className, filename)
        else:
            raise Exception("There is no rule population to export, as the eLCS model has not been trained")

    def exportPopStatsToCSV(self,iterationNumber,headerNames=np.array([]),filename='popStats.csv'):
        if self.evalWhileFitAfter or iterationNumber != self.learningIterations - 1:
            self.record.exportSumsToCSV(self, iterationNumber, headerNames,filename)
        else:
            self.exportFinalPopStatsToCSV(headerNames,filename)

    '''
    Even if evalWhileFit was turned off, this will run an immediate evaluation and export it.
    '''
    def exportFinalPopStatsToCSV(self,headerNames=np.array([]),filename='popStats.csv'):
        if self.hasTrained:
            if self.evalWhileFitAfter:
                self.record.exportFinalSumsToCSV(self,headerNames,filename)
            else:
                self.population.runPopAveEval(self.explorIter, self)
                self.population.runAttGeneralitySum(True, self)
                self.env.startEvaluationMode()  # Preserves learning position in training data

                if self.env.formatData.discretePhenotype:
                    trainEval = self.doPopEvaluation()
                else:
                    trainEval = self.doContPopEvaluation()

                self.record.addToEval(self.explorIter-1, trainEval[0], trainEval[1], copy.deepcopy(self.population.popSet),copy.deepcopy(self.population.attributeSpecList),copy.deepcopy(self.population.attributeAccList))
                self.evalWhileFitAfter = True #So it doesn't run this else again if this is invoked again
                self.env.stopEvaluationMode()  # Returns to learning position in training data

                self.record.exportFinalSumsToCSV(self,headerNames,filename)
        else:
            raise Exception("There is no rule population to export, as the eLCS model has not been trained")

    '''
    Note that all iterationNumbers are zero indexed (i.e. if learningIterations = 1000, the last iteration would be 999)
    '''
    def getMacroPopulationSize(self,iterationNumber):
        return self.record.getMacroPopulationSize(iterationNumber)

    def getFinalMacroPopulationSize(self):
        return self.record.getFinalMacroPopulationSize()

    def getMicroPopulationSize(self, iterationNumber):
        return self.record.getMicroPopulationSize(iterationNumber)

    def getFinalMicroPopulationSize(self):
        return self.record.getFinalMicroPopulationSize()

    def getPopAvgGenerality(self, iterationNumber):
        return self.record.getPopAvgGenerality(iterationNumber)

    def getFinalPopAvgGenerality(self):
        return self.record.getFinalPopAvgGenerality()

    def getTimeToTrain(self, iterationNumber):
        return self.record.getTimeToTrain(iterationNumber)

    def getFinalTimeToTrain(self):
        return self.record.getFinalTimeToTrain()

    '''
    If evalWhiteFit was turned off, as long as the iterationNumber is the final iteration, an immediate evaluation will be run
    on the population. Past unsaved rule populations are not obviously not valid for evaluation.
    '''
    def getAccuracy(self, iterationNumber):
        if self.evalWhileFitAfter or iterationNumber != self.learningIterations - 1:
            return self.record.getAccuracy(iterationNumber)
        else:
            self.getFinalAccuracy()

    def getInstanceCoverage(self,iterationNumber):
        if self.evalWhileFitAfter or iterationNumber != self.learningIterations - 1:
            return self.record.getInstanceCoverage(iterationNumber)
        else:
            self.getFinalInstanceCoverage()


    '''
    Even if evalWhileFit was turned off, this will run an immediate evaluation and give an accuracy.
    '''
    def getFinalAccuracy(self):
        if self.evalWhileFitAfter:
            return self.record.getFinalAccuracy()
        else:
            self.population.runPopAveEval(self.explorIter, self)
            self.population.runAttGeneralitySum(True, self)
            self.env.startEvaluationMode()  # Preserves learning position in training data
            if self.env.formatData.discretePhenotype:
                trainEval = self.doPopEvaluation()
            else:
                trainEval = self.doContPopEvaluation()

            self.record.addToEval(self.explorIter-1, trainEval[0], trainEval[1], copy.deepcopy(self.population.popSet),copy.deepcopy(self.population.attributeSpecList),copy.deepcopy(self.population.attributeAccList))

            self.env.stopEvaluationMode()  # Returns to learning position in training data
            self.evalWhileFitAfter = True #So it doesn't run this again when this is invoked again
            return self.record.getFinalAccuracy()

    def getFinalInstanceCoverage(self):
        if self.evalWhileFitAfter:
            return self.record.getFinalInstanceCoverage()
        else:
            self.population.runPopAveEval(self.explorIter, self)
            self.population.runAttGeneralitySum(True, self)
            self.env.startEvaluationMode()  # Preserves learning position in training data
            if self.env.formatData.discretePhenotype:
                trainEval = self.doPopEvaluation()
            else:
                trainEval = self.doContPopEvaluation()

            self.record.addToEval(self.explorIter-1, trainEval[0], trainEval[1], copy.deepcopy(self.population.popSet),copy.deepcopy(self.population.attributeSpecList),copy.deepcopy(self.population.attributeAccList))

            self.env.stopEvaluationMode()  # Returns to learning position in training data
            self.evalWhileFitAfter = True #So it doesn't run this again when this is invoked again
            return self.record.getFinalInstanceCoverage()

    #######################################################PRINT METHODS FOR DEBUGGING################################################################################

    def printClassifier(self,classifier):
        attributeCounter = 0

        for attribute in range(self.env.formatData.numAttributes):
            if attribute in classifier.specifiedAttList:
                specifiedLocation = classifier.specifiedAttList.index(attribute)
                if self.env.formatData.attributeInfoType[attributeCounter] == 0:  # isDiscrete
                    print(classifier.condition[specifiedLocation], end="\t\t\t\t")
                else:
                    print("[", end="")
                    print(
                        round(classifier.condition[specifiedLocation][0] * 10) / 10,
                        end=", ")
                    print(
                        round(classifier.condition[specifiedLocation][1] * 10) / 10,
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

    #######################################################TEMPORARY STORAGE OBJECTS################################################################################
class tempTrackingObj():
    #Tracks stats of every iteration (except accuracy, avg generality, and times)
    def __init__(self):
        self.macroPopSize = 0
        self.microPopSize = 0
        self.matchSetSize = 0
        self.correctSetSize = 0
        self.avgIterAge = 0
        self.subsumptionCount = 0
        self.crossOverCount = 0
        self.mutationCount = 0
        self.coveringCount = 0
        self.deletionCount = 0

    def resetAll(self):
        self.macroPopSize = 0
        self.microPopSize = 0
        self.matchSetSize = 0
        self.correctSetSize = 0
        self.avgIterAge = 0
        self.subsumptionCount = 0
        self.crossOverCount = 0
        self.mutationCount = 0
        self.coveringCount = 0
        self.deletionCount = 0
