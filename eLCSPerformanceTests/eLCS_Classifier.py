"""
Name:        eLCS_Classifier.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     November 1, 2013
Description: This module defines an individual classifier within the rule population, along with all respective parameters.
             Also included are classifier-level methods, including constructors(covering, copy, reboot) matching, subsumption, 
             crossover, and mutation.  Parameter update methods are also included.
             
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

#Import Required Modules---------------
from eLCS_Constants import *
import random
import copy
import math
#--------------------------------------

class Classifier:
    def __init__(self,a=None,b=None,c=None,d=None):
        #Major Parameters --------------------------------------------------
        self.specifiedAttList = []      # Attribute Specified in classifier: Similar to Bacardit 2009 - ALKR + GABIL, continuous and discrete rule representation
        self.condition = []             # States of Attributes Specified in classifier: Similar to Bacardit 2009 - ALKR + GABIL, continuous and discrete rule representation
        self.phenotype = None           # Class if the endpoint is discrete, and a continuous phenotype if the endpoint is continuous
        
        self.fitness = cons.init_fit    # Classifier fitness - initialized to a constant initial fitness value
        self.accuracy = 0.0             # Classifier accuracy - Accuracy calculated using only instances in the dataset which this rule matched.
        self.numerosity = 1             # The number of rule copies stored in the population.  (Indirectly stored as incremented numerosity)
        self.aveMatchSetSize = None     # A parameter used in deletion which reflects the size of match sets within this rule has been included.
        self.deletionVote = None        # The current deletion weight for this classifier.
        
        #Experience Management ---------------------------------------------
        self.timeStampGA = None         # Time since rule last in a correct set.
        self.initTimeStamp = None       # Iteration in which the rule first appeared.
        
        #Classifier Accuracy Tracking --------------------------------------
        self.matchCount = 0             # Known in many LCS implementations as experience i.e. the total number of times this classifier was in a match set
        self.correctCount = 0           # The total number of times this classifier was in a correct set
        
        if isinstance(c,list):
            self.classifierCovering(a,b,c,d)
        elif isinstance(a,Classifier):
            self.classifierCopy(a, b)
        elif isinstance(a,list) and b == None:
            self.rebootClassifier(a)
        else:
            print("Classifier: Error building classifier.")
            
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # CLASSIFIER CONSTRUCTION METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------       
    def classifierCovering(self, setSize, exploreIter, state, phenotype):
        """ Makes a new classifier when the covering mechanism is triggered.  The new classifier will match the current training instance. 
        Covering will NOT produce a default rule (i.e. a rule with a completely general condition). """
        #Initialize new classifier parameters----------
        self.timeStampGA = exploreIter
        self.initTimeStamp = exploreIter
        self.aveMatchSetSize = setSize
        dataInfo = cons.env.formatData
        #-------------------------------------------------------
        # DISCRETE PHENOTYPE
        #-------------------------------------------------------
        if dataInfo.discretePhenotype: 
            self.phenotype = phenotype
        #-------------------------------------------------------
        # CONTINUOUS PHENOTYPE
        #-------------------------------------------------------
        else:
            phenotypeRange = dataInfo.phenotypeList[1] - dataInfo.phenotypeList[0]
            rangeRadius = random.randint(25,75)*0.01*phenotypeRange / 2.0 #Continuous initialization domain radius.
            Low = float(phenotype) - rangeRadius
            High = float(phenotype) + rangeRadius
            self.phenotype = [Low,High] #ALKR Representation, Initialization centered around training instance  with a range between 25 and 75% of the domain size.      
        #-------------------------------------------------------
        # GENERATE MATCHING CONDITION
        #-------------------------------------------------------
        while len(self.specifiedAttList) < 1:
            for attRef in range(len(state)):
                if random.random() < cons.p_spec and state[attRef] != cons.labelMissingData:
                    self.specifiedAttList.append(attRef)
                    self.condition.append(self.buildMatch(attRef, state))

            
    def classifierCopy(self, clOld, exploreIter):
        """  Constructs an identical Classifier.  However, the experience of the copy is set to 0 and the numerosity 
        is set to 1 since this is indeed a new individual in a population. Used by the genetic algorithm to generate 
        offspring based on parent classifiers."""
        self.specifiedAttList = copy.deepcopy(clOld.specifiedAttList)
        self.condition = copy.deepcopy(clOld.condition) 
        self.phenotype = copy.deepcopy(clOld.phenotype)
        self.timeStampGA = exploreIter
        self.initTimeStamp = exploreIter
        self.aveMatchSetSize = copy.deepcopy(clOld.aveMatchSetSize)
        self.fitness = clOld.fitness
        self.accuracy = clOld.accuracy
        
        
    def rebootClassifier(self, classifierList): 
        """ Rebuilds a saved classifier as part of the population Reboot """
        numAttributes = cons.env.formatData.numAttributes
        attInfo = cons.env.formatData.attributeInfo
        for attRef in range(0,numAttributes):
            if classifierList[attRef] != '#':  #Attribute in rule is not wild
                if attInfo[attRef][0]: #Continuous Attribute
                    valueRange = classifierList[attRef].split(';')
                    self.condition.append(valueRange)
                    self.specifiedAttList.append(attRef)
                else:
                    self.condition.append(classifierList[attRef])
                    self.specifiedAttList.append(attRef)
        #-------------------------------------------------------
        # DISCRETE PHENOTYPE
        #-------------------------------------------------------
        if cons.env.formatData.discretePhenotype: 
            self.phenotype = str(classifierList[numAttributes])
        #-------------------------------------------------------
        # CONTINUOUS PHENOTYPE
        #-------------------------------------------------------
        else:
            self.phenotype = classifierList[numAttributes].split(';')
            for i in range(2): 
                self.phenotype[i] = float(self.phenotype[i])

        self.fitness = float(classifierList[numAttributes+1])
        self.accuracy = float(classifierList[numAttributes+2])
        self.numerosity = int(classifierList[numAttributes+3])
        self.aveMatchSetSize = float(classifierList[numAttributes+4])
        self.timeStampGA = int(classifierList[numAttributes+5])
        self.initTimeStamp = int(classifierList[numAttributes+6])
        
        self.deletionVote = float(classifierList[numAttributes+8])
        self.correctCount = int(classifierList[numAttributes+9])
        self.matchCount = int(classifierList[numAttributes+10])


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # MATCHING
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  
    def match(self, state):
        """ Returns if the classifier matches in the current situation. """ 
        for i in range(len(self.condition)):
            attributeInfo = cons.env.formatData.attributeInfo[self.specifiedAttList[i]]
            #-------------------------------------------------------
            # CONTINUOUS ATTRIBUTE
            #-------------------------------------------------------
            if attributeInfo[0]:
                instanceValue = state[self.specifiedAttList[i]]
                if self.condition[i][0] < instanceValue < self.condition[i][1] or instanceValue == cons.labelMissingData:
                    pass
                else:
                    return False  
            #-------------------------------------------------------
            # DISCRETE ATTRIBUTE
            #-------------------------------------------------------
            else:
                stateRep = state[self.specifiedAttList[i]]  
                if stateRep == self.condition[i] or stateRep == cons.labelMissingData:
                    pass
                else:
                    return False 
        return True
        

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # GENETIC ALGORITHM MECHANISMS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  
    def uniformCrossover(self, cl):
        """ Applies uniform crossover and returns if the classifiers changed. Handles both discrete and continuous attributes.  
        #SWARTZ: self. is where for the better attributes are more likely to be specified
        #DEVITO: cl. is where less useful attribute are more likely to be specified
        """
        if cons.env.formatData.discretePhenotype or random.random() < 0.5: #Always crossover condition if the phenotype is discrete (if continuous phenotype, half the time phenotype crossover is performed instead)
            p_self_specifiedAttList = copy.deepcopy(self.specifiedAttList)
            p_cl_specifiedAttList = copy.deepcopy(cl.specifiedAttList)
                
            #Make list of attribute references appearing in at least one of the parents.-----------------------------
            comboAttList = []
            for i in p_self_specifiedAttList:
                comboAttList.append(i)
            for i in p_cl_specifiedAttList:
                if i not in comboAttList:
                    comboAttList.append(i)
                elif not cons.env.formatData.attributeInfo[i][0]: #Attribute specified in both parents, and the attribute is discrete (then no reason to cross over)
                    comboAttList.remove(i)
            comboAttList.sort()
            #--------------------------------------------------------------------------------------------------------
            changed = False;   
            for attRef in comboAttList:  #Each condition specifies different attributes, so we need to go through all attributes in the dataset.
                attributeInfo = cons.env.formatData.attributeInfo[attRef]
                probability = 0.5  #Equal probability for attribute alleles to be exchanged.
                #-----------------------------
                ref = 0
                #if attRef in self.specifiedAttList:
                if attRef in p_self_specifiedAttList:
                    ref += 1
                #if attRef in cl.specifiedAttList:
                if attRef in p_cl_specifiedAttList:
                    ref += 1
                #-----------------------------

                if ref == 0:    #Attribute not specified in either condition (Attribute type makes no difference)
                    print("Error: UniformCrossover!")
                    pass
                
                elif ref == 1:  #Attribute specified in only one condition - do probabilistic switch of whole attribute state (Attribute type makes no difference)
                    if attRef in p_self_specifiedAttList and random.random() > probability: 
                        i = self.specifiedAttList.index(attRef) #reference to the position of the attribute in the rule representation
                        cl.condition.append(self.condition.pop(i)) #Take attribute from self and add to cl
                        cl.specifiedAttList.append(attRef)
                        self.specifiedAttList.remove(attRef)
                        changed = True #Remove att from self and add to cl

                        
                    if attRef in p_cl_specifiedAttList and random.random() < probability: 
                        i = cl.specifiedAttList.index(attRef) #reference to the position of the attribute in the rule representation
                        self.condition.append(cl.condition.pop(i)) #Take attribute from self and add to cl
                        self.specifiedAttList.append(attRef)
                        cl.specifiedAttList.remove(attRef)
                        changed = True #Remove att from cl and add to self.

    
                else: #Attribute specified in both conditions - do random crossover between state alleles.  The same attribute may be specified at different positions within either classifier
                    #-------------------------------------------------------
                    # CONTINUOUS ATTRIBUTE
                    #-------------------------------------------------------
                    if attributeInfo[0]: 
                        i_cl1 = self.specifiedAttList.index(attRef) #pairs with self (classifier 1)
                        i_cl2 = cl.specifiedAttList.index(attRef)   #pairs with cl (classifier 2)
                        tempKey = random.randint(0,3) #Make random choice between 4 scenarios, Swap minimums, Swap maximums, Self absorbs cl, or cl absorbs self.
                        if tempKey == 0:    #Swap minimum
                            temp = self.condition[i_cl1][0]
                            self.condition[i_cl1][0] = cl.condition[i_cl2][0]
                            cl.condition[i_cl2][0] = temp
                        elif tempKey == 1:  #Swap maximum
                            temp = self.condition[i_cl1][1]
                            self.condition[i_cl1][1] = cl.condition[i_cl2][1]
                            cl.condition[i_cl2][1] = temp
                        else: #absorb range
                            allList = self.condition[i_cl1] + cl.condition[i_cl2]
                            newMin = min(allList)
                            newMax = max(allList)
                            if tempKey == 2:  #self absorbs cl
                                self.condition[i_cl1] = [newMin,newMax]
                                #Remove cl
                                cl.condition.pop(i_cl2)
                                cl.specifiedAttList.remove(attRef)
                            else: #cl absorbs self
                                cl.condition[i_cl2] = [newMin,newMax]
                                #Remove self
                                self.condition.pop(i_cl1)
                                self.specifiedAttList.remove(attRef)
                    #-------------------------------------------------------
                    # DISCRETE ATTRIBUTE
                    #-------------------------------------------------------
                    else: 
                        pass

            tempList1 = copy.deepcopy(p_self_specifiedAttList)
            tempList2 = copy.deepcopy(cl.specifiedAttList)
            tempList1.sort()
            tempList2.sort()
            
            if changed and (tempList1 == tempList2):
                changed = False
                
            return changed
        #-------------------------------------------------------
        # CONTINUOUS PHENOTYPE CROSSOVER
        #-------------------------------------------------------
        else: 
            return self.phenotypeCrossover(cl)
        
        
    def phenotypeCrossover(self, cl):
        """ Crossover a continuous phenotype """
        changed = False
        if self.phenotype[0] == cl.phenotype[0] and self.phenotype[1] == cl.phenotype[1]:
            return changed
        else:
            tempKey = random.random() < 0.5 #Make random choice between 4 scenarios, Swap minimums, Swap maximums, Children preserve parent phenotypes.
            if tempKey: #Swap minimum
                temp = self.phenotype[0]
                self.phenotype[0] = cl.phenotype[0]
                cl.phenotype[0] = temp
                changed = True
            elif tempKey:  #Swap maximum
                temp = self.phenotype[1]
                self.phenotype[1] = cl.phenotype[1]
                cl.phenotype[1] = temp
                changed = True
            
        return changed
        
        
    def Mutation(self, state, phenotype):
        """ Mutates the condition of the classifier. Also handles phenotype mutation. This is a niche mutation, which means that the resulting classifier will still match the current instance.  """               
        changed = False;   
        #-------------------------------------------------------
        # MUTATE CONDITION
        #-------------------------------------------------------
        for attRef in range(cons.env.formatData.numAttributes):  #Each condition specifies different attributes, so we need to go through all attributes in the dataset.
            attributeInfo = cons.env.formatData.attributeInfo[attRef]
            if random.random() < cons.mu and state[attRef] != cons.labelMissingData:
                #MUTATION--------------------------------------------------------------------------------------------------------------
                if attRef not in self.specifiedAttList: #Attribute not yet specified
                    self.specifiedAttList.append(attRef)
                    self.condition.append(self.buildMatch(attRef, state)) #buildMatch handles both discrete and continuous attributes
                    changed = True
                    
                elif attRef in self.specifiedAttList: #Attribute already specified
                    i = self.specifiedAttList.index(attRef) #reference to the position of the attribute in the rule representation
                    #-------------------------------------------------------
                    # DISCRETE OR CONTINUOUS ATTRIBUTE - remove attribute specification with 50% chance if we have continuous attribute, or 100% if discrete attribute.
                    #-------------------------------------------------------
                    if not attributeInfo[0] or random.random() > 0.5: 
                        self.specifiedAttList.remove(attRef)
                        self.condition.pop(i) #buildMatch handles both discrete and continuous attributes
                        changed = True
                    #-------------------------------------------------------
                    # CONTINUOUS ATTRIBUTE - (mutate range with 50% probability vs. removing specification of this attribute all together)
                    #-------------------------------------------------------
                    else: 
                        #Mutate continuous range - based on Bacardit 2009 - Select one bound with uniform probability and add or subtract a randomly generated offset to bound, of size between 0 and 50% of att domain.
                        attRange = float(attributeInfo[1][1]) - float(attributeInfo[1][0])
                        mutateRange = random.random()*0.5*attRange
                        if random.random() > 0.5: #Mutate minimum 
                            if random.random() > 0.5: #Add
                                self.condition[i][0] += mutateRange
                            else: #Subtract
                                self.condition[i][0] -= mutateRange
                        else: #Mutate maximum
                            if random.random() > 0.5: #Add
                                self.condition[i][1] += mutateRange
                            else: #Subtract
                                self.condition[i][1] -= mutateRange
                                
                        #Repair range - such that min specified first, and max second.
                        self.condition[i].sort()
                        changed = True
                #-------------------------------------------------------
                # NO MUTATION OCCURS
                #-------------------------------------------------------
                else:
                    pass
        #-------------------------------------------------------
        # MUTATE PHENOTYPE
        #-------------------------------------------------------
        if cons.env.formatData.discretePhenotype:
            nowChanged = self.discretePhenotypeMutation()
        else:
            nowChanged = self.continuousPhenotypeMutation(phenotype)
        
        if changed or nowChanged:
            return True


    def discretePhenotypeMutation(self):
        """ Mutate this rule's discrete phenotype. """
        changed = False
        if random.random() < cons.mu:
            phenotypeList = copy.deepcopy(cons.env.formatData.phenotypeList)
            phenotypeList.remove(self.phenotype)
            newPhenotype = random.sample(phenotypeList,1)
            self.phenotype = newPhenotype[0]
            changed= True

        return changed
        
        
    def continuousPhenotypeMutation(self, phenotype):
        """ Mutate this rule's continuous phenotype. """
        changed = False
        if random.random() < cons.mu: #Mutate continuous phenotype
            phenRange = self.phenotype[1] - self.phenotype[0]
            mutateRange = random.random()*0.5*phenRange
            tempKey = random.randint(0,2) #Make random choice between 3 scenarios, mutate minimums, mutate maximums, mutate both
            if tempKey == 0: #Mutate minimum 
                if random.random() > 0.5 or self.phenotype[0] + mutateRange <= phenotype: #Checks that mutated range still contains current phenotype
                    self.phenotype[0] += mutateRange
                else: #Subtract
                    self.phenotype[0] -= mutateRange
                changed = True
            elif tempKey == 1: #Mutate maximum
                if random.random() > 0.5 or self.phenotype[1] - mutateRange >= phenotype: #Checks that mutated range still contains current phenotype
                    self.phenotype[1] -= mutateRange
                else: #Subtract
                    self.phenotype[1] += mutateRange
                changed = True
            else: #mutate both
                if random.random() > 0.5 or self.phenotype[0] + mutateRange <= phenotype: #Checks that mutated range still contains current phenotype
                    self.phenotype[0] += mutateRange
                else: #Subtract
                    self.phenotype[0] -= mutateRange
                if random.random() > 0.5 or self.phenotype[1] - mutateRange >= phenotype: #Checks that mutated range still contains current phenotype
                    self.phenotype[1] -= mutateRange
                else: #Subtract
                    self.phenotype[1] += mutateRange
                changed = True
            
            #Repair range - such that min specified first, and max second.
            self.phenotype.sort()
        #---------------------------------------------------------------------
        return changed    
        
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # SUBSUMPTION METHODS
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def subsumes(self, cl):
        """ Returns if the classifier (self) subsumes cl """
        #-------------------------------------------------------
        # DISCRETE PHENOTYPE
        #-------------------------------------------------------
        if cons.env.formatData.discretePhenotype: 
            if cl.phenotype == self.phenotype:
                if self.isSubsumer() and self.isMoreGeneral(cl):
                    return True
            return False
        #-------------------------------------------------------
        # CONTINUOUS PHENOTYPE -  NOTE: for continuous phenotypes, the subsumption intuition is reversed, i.e. While a subsuming rule condition is more general, a subsuming phenotype is more specific.
        #-------------------------------------------------------
        else:
            if self.phenotype[0] >= cl.phenotype[0] and self.phenotype[1] <= cl.phenotype[1]:
                if self.isSubsumer() and self.isMoreGeneral(cl):
                    return True
            return False  
        

    def isSubsumer(self):
        """ Returns if the classifier (self) is a possible subsumer. A classifier must be as or more accurate than the classifier it is trying to subsume.  """
        if self.matchCount > cons.theta_sub and self.accuracy > cons.acc_sub: 
            return True
        return False
    
    
    def isMoreGeneral(self,cl):
        """ Returns if the classifier (self) is more general than cl. Check that all attributes specified in self are also specified in cl. """ 
        if len(self.specifiedAttList) >= len(cl.specifiedAttList):
            return False
        for i in range(len(self.specifiedAttList)): #Check each attribute specified in self.condition
            attributeInfo = cons.env.formatData.attributeInfo[self.specifiedAttList[i]]
            if self.specifiedAttList[i] not in cl.specifiedAttList:
                return False
            #-------------------------------------------------------
            # CONTINUOUS ATTRIBUTE
            #-------------------------------------------------------
            if attributeInfo[0]:
                otherRef = cl.specifiedAttList.index(self.specifiedAttList[i])
                #If self has a narrower ranger of values than it is a subsumer
                if self.condition[i][0] < cl.condition[otherRef][0]:
                    return False
                if self.condition[i][1] > cl.condition[otherRef][1]:
                    return False
                
        return True
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # DELETION METHOD
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------      
    def getDelProp(self, meanFitness):
        """  Returns the vote for deletion of the classifier. """
        if self.fitness/self.numerosity >= cons.delta*meanFitness or self.matchCount < cons.theta_del:
            self.deletionVote = self.aveMatchSetSize*self.numerosity

        elif self.fitness == 0.0:
            self.deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (cons.init_fit/self.numerosity)
        else:
            self.deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (self.fitness/self.numerosity) 
        return self.deletionVote


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # OTHER METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  
    def buildMatch(self, attRef, state):
        """ Builds a matching condition for the classifierCovering method. """
        attributeInfo = cons.env.formatData.attributeInfo[attRef]
        #-------------------------------------------------------
        # CONTINUOUS ATTRIBUTE
        #-------------------------------------------------------
        if attributeInfo[0]:
            attRange = attributeInfo[1][1] - attributeInfo[1][0]
            rangeRadius = random.randint(25,75)*0.01*attRange / 2.0 #Continuous initialization domain radius.
            Low = state[attRef] - rangeRadius
            High = state[attRef] + rangeRadius
            condList = [Low,High] #ALKR Representation, Initialization centered around training instance  with a range between 25 and 75% of the domain size.
        #-------------------------------------------------------
        # DISCRETE ATTRIBUTE
        #-------------------------------------------------------
        else: 
            condList = state[attRef] #State already formatted like GABIL in DataManagement
            
        return condList
     

    def equals(self, cl):  
        """ Returns if the two classifiers are identical in condition and phenotype. This works for discrete or continuous attributes or phenotypes. """ 
        if cl.phenotype == self.phenotype and len(cl.specifiedAttList) == len(self.specifiedAttList): #Is phenotype the same and are the same number of attributes specified - quick equality check first.
            clRefs = sorted(cl.specifiedAttList)
            selfRefs = sorted(self.specifiedAttList)
            if clRefs == selfRefs:
                for i in range(len(cl.specifiedAttList)):
                    tempIndex = self.specifiedAttList.index(cl.specifiedAttList[i])
                    if cl.condition[i] == self.condition[tempIndex]:
                        pass
                    else:
                        return False
                return True
        return False


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # PARAMETER UPDATES
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------        
    def updateAccuracy(self):
        """ Update the accuracy tracker """
        self.accuracy = self.correctCount / float(self.matchCount)
        
        
    def updateFitness(self):
        """ Update the fitness parameter. """ 
        if cons.env.formatData.discretePhenotype or (self.phenotype[1]-self.phenotype[0])/cons.env.formatData.phenotypeRange < 0.5:
            self.fitness = pow(self.accuracy, cons.nu)
        else:
            if (self.phenotype[1]-self.phenotype[0]) >= cons.env.formatData.phenotypeRange:
                self.fitness = 0.0
            else:
                self.fitness = math.fabs(pow(self.accuracy, cons.nu) - (self.phenotype[1]-self.phenotype[0])/cons.env.formatData.phenotypeRange)


    def updateMatchSetSize(self, matchSetSize): 
        """  Updates the average match set size. """
        if self.matchCount < 1.0 / cons.beta:
            self.aveMatchSetSize = (self.aveMatchSetSize * (self.matchCount-1)+ matchSetSize) / float(self.matchCount)
        else:
            self.aveMatchSetSize = self.aveMatchSetSize + cons.beta * (matchSetSize - self.aveMatchSetSize)
    
        
    def updateExperience(self):
        """ Increases the experience of the classifier by one. Once an epoch has completed, rule accuracy can't change."""
        self.matchCount += 1 


    def updateCorrect(self):
        """ Increases the correct phenotype tracking by one. Once an epoch has completed, rule accuracy can't change."""
        self.correctCount += 1 


    def updateNumerosity(self, num):
        """ Updates the numberosity of the classifier.  Notice that 'num' can be negative! """
        self.numerosity += num
        

    def updateTimeStamp(self, ts):
        """ Sets the time stamp of the classifier. """
        self.timeStampGA = ts
        
        
    def setAccuracy(self,acc):
        """ Sets the accuracy of the classifier """
        self.accuracy = acc
        
        
    def setFitness(self, fit):
        """  Sets the fitness of the classifier. """
        self.fitness = fit
        
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # PRINT CLASSIFIER FOR POPULATION OUTPUT FILE
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------     
    def printClassifier(self):
        """ Formats and returns an output string describing this classifier. """ 
        classifierString = ""
        for attRef in range(cons.env.formatData.numAttributes):
            attributeInfo = cons.env.formatData.attributeInfo[attRef]
            if attRef in self.specifiedAttList:  #If the attribute was specified in the rule
                i = self.specifiedAttList.index(attRef)
                #-------------------------------------------------------
                # CONTINUOUS ATTRIBUTE 
                #-------------------------------------------------------
                if attributeInfo[0]:
                    classifierString += str(self.condition[i][0])+';'+str(self.condition[i][1]) + "\t"
                #-------------------------------------------------------
                # DISCRETE ATTRIBUTE 
                #-------------------------------------------------------
                else: 
                    classifierString += str(self.condition[i]) + "\t"
            else: # Attribute is wild.
                classifierString += '#' + "\t"
        #-------------------------------------------------------------------------------
        specificity = len(self.condition) / float(cons.env.formatData.numAttributes)
        
        if cons.env.formatData.discretePhenotype:
            classifierString += str(self.phenotype)+"\t"
        else:
            classifierString += str(self.phenotype[0])+';'+str(self.phenotype[1])+"\t"
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #classifierString += str(self.fitness)+"\t"+str(self.accuracy)+"\t"+str(self.numerosity)+"\t"+str(self.aveMatchSetSize)+"\t"+str(self.timeStampGA)+"\t"+str(self.initTimeStamp)+"\t"+str(specificity)+"\t"
        #classifierString += str(self.deletionVote)+"\t"+str(self.correctCount)+"\t"+str(self.matchCount)+"\n"
        classifierString += str(self.fitness)+"\n"

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        return classifierString
