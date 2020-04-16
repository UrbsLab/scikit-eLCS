"""
Name:        eLCS_Algorithm.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     November 1, 2013
Description: The major controlling module of eLCS.  Includes the major run loop which controls learning over a specified number of iterations.  Also includes
             periodic tracking of estimated performance, and checkpoints where complete evaluations of the eLCS rule population is performed.
             
---------------------------------------------------------------------------------------------------------------------------------------------------------
eLCS: Educational Learning Classifier System - A basic LCS coded for educational purposes.  This LCS algorithm uses supervised learning, and thus is most 
similar to "UCS", an LCS algorithm published by Ester Bernado-Mansilla and Josep Garrell-Guiu (2003) which in turn is based heavily on "XCS", an LCS 
algorithm published by Stewart Wilson (1995).  

Copyright (C) 2013 Ryan Urbanowicz 
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the 
Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABLILITY 
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, 
Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#Import Required Modules-------------------------------
from eLCS_Constants import *
from eLCS_ClassifierSet import ClassifierSet
from eLCS_Prediction import *
from eLCS_ClassAccuracy import ClassAccuracy
import copy
import random
import math
#------------------------------------------------------

class eLCS:
    def __init__(self):
        """ Initializes the eLCS algorithm """
        #Global Parameters-------------------------------------------------------------------------------------
        self.population = None          # The rule population (the 'solution/model' evolved by eLCS)
        self.learnTrackOut = None       # Output file that will store tracking information during learning
        
        #-------------------------------------------------------
        # POPULATION REBOOT - Begin eLCS learning from an existing saved rule population
        #-------------------------------------------------------
        if cons.doPopulationReboot:    
            self.populationReboot()
            
        #-------------------------------------------------------
        # NORMAL eLCS - Run eLCS from scratch on given data
        #-------------------------------------------------------
        else:
            # Instantiate Population---------
            self.population = ClassifierSet()
            self.exploreIter = 0
            self.correct  = [0.0 for i in range(cons.trackingFrequency)]
            
        #Run the eLCS algorithm-------------------------------------------------------------------------------
        self.run_eLCS()


    def run_eLCS(self):
        """ Runs the initialized eLCS algorithm. """
        #--------------------------------------------------------------
        # MAJOR LEARNING LOOP
        #-------------------------------------------------------
        while self.exploreIter < cons.maxLearningIterations: 
            
            #-------------------------------------------------------
            # GET NEW INSTANCE AND RUN A LEARNING ITERATION
            #-------------------------------------------------------
            state_phenotype = cons.env.getTrainInstance() 
            self.runIteration(state_phenotype, self.exploreIter)
            
            #-------------------------------------------------------------------------------------------------------------------------------
            # EVALUATIONS OF ALGORITHM
            #-------------------------------------------------------------------------------------------------------------------------------
            cons.timer.startTimeEvaluation()
            
            #-------------------------------------------------------
            # TRACK LEARNING ESTIMATES
            #-------------------------------------------------------
            if (self.exploreIter%cons.trackingFrequency) == (cons.trackingFrequency - 1) and self.exploreIter > 0:
                self.population.runPopAveEval(self.exploreIter) 
                trackedAccuracy = sum(self.correct)/float(cons.trackingFrequency) #Accuracy over the last "trackingFrequency" number of iterations.
            cons.timer.stopTimeEvaluation()
            
            #-------------------------------------------------------
            # CHECKPOINT - COMPLETE EVALUTATION OF POPULATION - strategy different for discrete vs continuous phenotypes
            #-------------------------------------------------------
            if (self.exploreIter + 1) in cons.learningCheckpoints:
                if self.exploreIter + 1 != cons.maxLearningIterations:
                    cons.timer.startTimeEvaluation()
                else:
                    cons.timer.returnGlobalTimer()
                
                self.population.runPopAveEval(self.exploreIter)
                self.population.runAttGeneralitySum(True)
                cons.env.startEvaluationMode()  #Preserves learning position in training data
                if cons.testFile != 'None': #If a testing file is available.
                    if cons.env.formatData.discretePhenotype: 
                        trainEval = self.doPopEvaluation(True)
                        testEval = self.doPopEvaluation(False)
                    else: 
                        trainEval = self.doContPopEvaluation(True)
                        testEval = self.doContPopEvaluation(False)
                else:  #Only a training file is available
                    if cons.env.formatData.discretePhenotype: 
                        trainEval = self.doPopEvaluation(True)
                        testEval = None
                    else: 
                        trainEval = self.doContPopEvaluation(True)
                        testEval = None

                self.trainEval = trainEval
                self.testEval = testEval
                cons.env.stopEvaluationMode() #Returns to learning position in training data
                if self.exploreIter + 1 != cons.maxLearningIterations:
                    cons.timer.stopTimeEvaluation()
                    cons.timer.returnGlobalTimer()
                #----------------------------------------------------------------------------------------------------------------------------

            #-------------------------------------------------------
            # ADJUST MAJOR VALUES FOR NEXT ITERATION
            #-------------------------------------------------------
            self.exploreIter += 1       # Increment current learning iteration
            cons.env.newInstance(True)  # Step to next instance in training set

    def runIteration(self, state_phenotype, exploreIter):
        """ Run a single eLCS learning iteration. """
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # FORM A MATCH SET - includes covering
        #-----------------------------------------------------------------------------------------------------------------------------------------
        self.population.makeMatchSet(state_phenotype, exploreIter)
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # MAKE A PREDICTION - utilized here for tracking estimated learning progress.  Typically used in the explore phase of many LCS algorithms.
        #-----------------------------------------------------------------------------------------------------------------------------------------
        cons.timer.startTimeEvaluation() 
        prediction = Prediction(self.population)
        phenotypePrediction = prediction.getDecision()  
        #-------------------------------------------------------
        # PREDICTION NOT POSSIBLE
        #-------------------------------------------------------
        if phenotypePrediction == None or phenotypePrediction == 'Tie': 
            if cons.env.formatData.discretePhenotype:
                phenotypePrediction = random.choice(cons.env.formatData.phenotypeList)
            else:
                phenotypePrediction = random.randrange(cons.env.formatData.phenotypeList[0],cons.env.formatData.phenotypeList[1],(cons.env.formatData.phenotypeList[1]-cons.env.formatData.phenotypeList[0])/float(1000))
        else: #Prediction Successful
            #-------------------------------------------------------
            # DISCRETE PHENOTYPE PREDICTION
            #-------------------------------------------------------
            if cons.env.formatData.discretePhenotype:
                if phenotypePrediction == state_phenotype[1]:
                    self.correct[exploreIter%cons.trackingFrequency] = 1
                else:
                    self.correct[exploreIter%cons.trackingFrequency] = 0
            #-------------------------------------------------------
            # CONTINUOUS PHENOTYPE PREDICTION
            #-------------------------------------------------------
            else:
                predictionError = math.fabs(phenotypePrediction - float(state_phenotype[1]))
                phenotypeRange = cons.env.formatData.phenotypeList[1] - cons.env.formatData.phenotypeList[0]
                accuracyEstimate = 1.0 - (predictionError / float(phenotypeRange))
                self.correct[exploreIter%cons.trackingFrequency] = accuracyEstimate
        cons.timer.stopTimeEvaluation()
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # FORM A CORRECT SET
        #-----------------------------------------------------------------------------------------------------------------------------------------
        self.population.makeCorrectSet(state_phenotype[1])
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # UPDATE PARAMETERS
        #-----------------------------------------------------------------------------------------------------------------------------------------
        self.population.updateSets(exploreIter)
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # SUBSUMPTION - APPLIED TO CORRECT SET - A heuristic for addition additional generalization pressure to eLCS
        #-----------------------------------------------------------------------------------------------------------------------------------------
        if cons.doSubsumption:
            cons.timer.startTimeSubsumption()
            self.population.doCorrectSetSubsumption()
            cons.timer.stopTimeSubsumption()
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # RUN THE GENETIC ALGORITHM - Discover new offspring rules from a selected pair of parents
        #-----------------------------------------------------------------------------------------------------------------------------------------
        self.population.runGA(exploreIter, state_phenotype[0], state_phenotype[1])
        #-----------------------------------------------------------------------------------------------------------------------------------------
        # SELECT RULES FOR DELETION - This is done whenever there are more rules in the population than 'N', the maximum population size.
        #-----------------------------------------------------------------------------------------------------------------------------------------
        self.population.deletion(exploreIter)
        self.population.clearSets() #Clears the match and correct sets for the next learning iteration
        
        
    def doPopEvaluation(self, isTrain):
        """ Performs a complete evaluation of the current rule population.  The population is unchanged throughout this evaluation. Works on both training and testing data. """
        if isTrain:
            myType = "TRAINING"
        else:
            myType = "TESTING"
        noMatch = 0                     # How often does the population fail to have a classifier that matches an instance in the data.
        tie = 0                         # How often can the algorithm not make a decision between classes due to a tie.
        cons.env.resetDataRef(isTrain)  # Go to the first instance in dataset
        phenotypeList = cons.env.formatData.phenotypeList 
        #----------------------------------------------
        classAccDict = {}
        for each in phenotypeList:
            classAccDict[each] = ClassAccuracy()
        #----------------------------------------------
        if isTrain:
            instances = cons.env.formatData.numTrainInstances
        else:
            instances = cons.env.formatData.numTestInstances
        #----------------------------------------------------------------------------------------------
        for inst in range(instances):
            if isTrain:
                state_phenotype = cons.env.getTrainInstance()
            else:
                state_phenotype = cons.env.getTestInstance()
            #-----------------------------------------------------------------------------
            self.population.makeEvalMatchSet(state_phenotype[0])
            prediction = Prediction(self.population)
            phenotypeSelection = prediction.getDecision() 
            #-----------------------------------------------------------------------------
            
            if phenotypeSelection == None: 
                noMatch += 1
            elif phenotypeSelection == 'Tie':
                tie += 1
            else: #Instances which failed to be covered are excluded from the accuracy calculation 
                for each in phenotypeList:
                    thisIsMe = False
                    accuratePhenotype = False
                    truePhenotype = state_phenotype[1]
                    if each == truePhenotype:
                        thisIsMe = True 
                    if phenotypeSelection == truePhenotype:
                        accuratePhenotype = True
                    classAccDict[each].updateAccuracy(thisIsMe, accuratePhenotype)
                        
            cons.env.newInstance(isTrain) #next instance
            self.population.clearSets() 
        #----------------------------------------------------------------------------------------------
        #Calculate Standard Accuracy--------------------------------------------
        instancesCorrectlyClassified = classAccDict[phenotypeList[0]].T_myClass + classAccDict[phenotypeList[0]].T_otherClass  
        instancesIncorrectlyClassified = classAccDict[phenotypeList[0]].F_myClass + classAccDict[phenotypeList[0]].F_otherClass 
        standardAccuracy = float(instancesCorrectlyClassified) / float(instancesCorrectlyClassified + instancesIncorrectlyClassified)

        #Calculate Balanced Accuracy---------------------------------------------
        T_mySum = 0
        T_otherSum = 0
        F_mySum = 0
        F_otherSum = 0
        for each in phenotypeList: 
            T_mySum += classAccDict[each].T_myClass
            T_otherSum += classAccDict[each].T_otherClass
            F_mySum += classAccDict[each].F_myClass
            F_otherSum += classAccDict[each].F_otherClass
        balancedAccuracy = ((0.5*T_mySum / (float(T_mySum + F_otherSum)) + 0.5*T_otherSum / (float(T_otherSum + F_mySum)))) # BalancedAccuracy = (Specificity + Sensitivity)/2

        #Adjustment for uncovered instances - to avoid positive or negative bias we incorporate the probability of guessing a phenotype by chance (e.g. 50% if two phenotypes)
        predictionFail = float(noMatch)/float(instances)
        predictionTies = float(tie)/float(instances)
        instanceCoverage = 1.0 - predictionFail
        predictionMade = 1.0 - (predictionFail + predictionTies)
        
        adjustedStandardAccuracy = (standardAccuracy * predictionMade) + ((1.0 - predictionMade) * (1.0 / float(len(phenotypeList))))
        adjustedBalancedAccuracy = (balancedAccuracy * predictionMade) + ((1.0 - predictionMade) * (1.0 / float(len(phenotypeList))))
        resultList = [adjustedBalancedAccuracy, instanceCoverage]
        return resultList
        
    
    def doContPopEvaluation(self, isTrain):
        """ Performs evaluation of population via the copied environment. Specifically developed for continuous phenotype evaulation.  
        The population is maintained unchanging throughout the evaluation.  Works on both training and testing data. """
        if isTrain:
            myType = "TRAINING"
        else:
            myType = "TESTING"
        noMatch = 0 #How often does the population fail to have a classifier that matches an instance in the data.
        cons.env.resetDataRef(isTrain) #Go to first instance in data set
        accuracyEstimateSum = 0

        if isTrain:
            instances = cons.env.formatData.numTrainInstances
        else:
            instances = cons.env.formatData.numTestInstances
        #----------------------------------------------------------------------------------------------
        for inst in range(instances):
            if isTrain:
                state_phenotype = cons.env.getTrainInstance()
            else:
                state_phenotype = cons.env.getTestInstance()
            #-----------------------------------------------------------------------------
            self.population.makeEvalMatchSet(state_phenotype[0])
            prediction = Prediction(self.population)
            phenotypePrediction = prediction.getDecision() 
            #-----------------------------------------------------------------------------
            if phenotypePrediction == None: 
                noMatch += 1
            else: #Instances which failed to be covered are excluded from the initial accuracy calculation
                predictionError = math.fabs(float(phenotypePrediction) - float(state_phenotype[1]))
                phenotypeRange = cons.env.formatData.phenotypeList[1] - cons.env.formatData.phenotypeList[0]
                accuracyEstimateSum += 1.0 - (predictionError / float(phenotypeRange))
                           
            cons.env.newInstance(isTrain) #next instance
            self.population.clearSets() 
        #----------------------------------------------------------------------------------------------
        #Accuracy Estimate
        if instances == noMatch:
            accuracyEstimate = 0
        else:
            accuracyEstimate = accuracyEstimateSum / float(instances - noMatch)
        
        #Adjustment for uncovered instances - to avoid positive or negative bias we incorporate the probability of guessing a phenotype by chance (e.g. 50% if two phenotypes)
        instanceCoverage = 1.0 - (float(noMatch)/float(instances))
        adjustedAccuracyEstimate = accuracyEstimateSum / float(instances) #noMatchs are treated as incorrect predictions (can see no other fair way to do this)

        resultList = [adjustedAccuracyEstimate, instanceCoverage]
        return resultList
    
    
    def populationReboot(self):
        """ Manages the reformation of a previously saved eLCS classifier population. """
        cons.timer.setTimerRestart(cons.popRebootPath) #Rebuild timer objects
        #--------------------------------------------------------------------
        try: #Re-open track learning file for continued tracking of progress.
            self.learnTrackOut = open(cons.outFileName+'_LearnTrack.txt','a')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', cons.outFileName+'_LearnTrack.txt')
            raise      
        
        #Extract last iteration from file name---------------------------------------------
        temp = cons.popRebootPath.split('_')
        iterRef = len(temp)-1
        completedIterations = int(temp[iterRef])
        print("Rebooting rule population after " +str(completedIterations)+ " iterations.")
        self.exploreIter = completedIterations-1
        for i in range(len(cons.learningCheckpoints)):
            cons.learningCheckpoints[i] += completedIterations
        cons.maxLearningIterations += completedIterations

        #Rebuild existing population from text file.--------
        self.population = ClassifierSet(cons.popRebootPath)
        #---------------------------------------------------
        try: #Obtain correct track
            f = open(cons.popRebootPath+"_PopStats.txt", 'r')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', cons.popRebootPath+"_PopStats.txt")
            raise 
        else:
            correctRef = 26 #File reference position
            tempLine = None
            for i in range(correctRef):
                tempLine = f.readline()
            tempList = tempLine.strip().split('\t')
            self.correct = tempList
            if cons.env.formatData.discretePhenotype:
                for i in range(len(self.correct)):
                    self.correct[i] = int(self.correct[i])
            else:
                for i in range(len(self.correct)):
                    self.correct[i] = float(self.correct[i])
            f.close()
        
