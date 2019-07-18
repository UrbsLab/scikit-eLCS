from eLCS import *
import random
import copy
import math
import numpy as np

class Classifier():
    def __init__(self,elcs,a=None,b=None,c=None,d=None):
        #Major Parameters
        self.specifiedAttList = np.array([],dtype='int64')
        self.condition = np.array([]) #array of ClassifierConditionElements
        self.phenotype = None #Can be either np_array of min and max or a discrete phenotype

        self.fitness = elcs.init_fit
        self.accuracy = 0.0
        self.numerosity = 1
        self.aveMatchSetSize = None
        self.deletionVote = None

        #Experience Management
        self.timeStampGA = None
        self.initTimeStamp = 0

        # Classifier Accuracy Tracking --------------------------------------
        self.matchCount = 0  # Known in many LCS implementations as experience i.e. the total number of times this classifier was in a match set
        self.correctCount = 0  # The total number of times this classifier was in a correct set

        if isinstance(c,np.ndarray):
            self.classifierCovering(elcs,a,b,c,d)
        elif isinstance(a,Classifier):
            self.classifierCopy(a,b)

    #Classifier Construction Methods
    def classifierCovering(self, elcs,setSize, exploreIter, state, phenotype):
        # Initialize new classifier parameters----------
        self.timeStampGA = exploreIter
        self.initTimeStamp = exploreIter
        self.aveMatchSetSize = setSize
        dataInfo = elcs.env.formatData

        # -------------------------------------------------------
        # DISCRETE PHENOTYPE
        # -------------------------------------------------------
        if dataInfo.discretePhenotype:
            self.phenotype = phenotype
        # -------------------------------------------------------
        # CONTINUOUS PHENOTYPE
        # -------------------------------------------------------
        else:
            phenotypeRange = dataInfo.phenotypeList[1] - dataInfo.phenotypeList[0]
            rangeRadius = random.randint(25,75) * 0.01 * phenotypeRange / 2.0  # Continuous initialization domain radius.
            Low = float(phenotype) - rangeRadius
            High = float(phenotype) + rangeRadius
            self.phenotype = np.array([Low,High])

        while self.specifiedAttList.size < 1:
            for attRef in range(state.size):
                if random.random() < elcs.p_spec and state[attRef].value != elcs.labelMissingData:
                    self.specifiedAttList = np.append(self.specifiedAttList,attRef)
                    self.condition = np.append(self.condition,self.buildMatch(elcs,attRef,state))#Add classifierConditionElement

    def classifierCopy(self,toCopy, exploreIter):
        self.specifiedAttList = copy.deepcopy(toCopy.specifiedAttList)
        self.condition = ClassifierConditionElement(copy.deepcopy(toCopy.condition.type),copy.deepcopy(toCopy.condition.value),copy.deepcopy(toCopy.condition.list))
        self.phenotype = copy.deepcopy(toCopy.phenotype)
        self.timeStampGA = exploreIter
        self.initTimeStamp = exploreIter
        self.aveMatchSetSize = copy.deepcopy(toCopy.aveMatchSetSize)
        self.fitness = toCopy.fitness
        self.accuracy = toCopy.accuracy

    def buildMatch(self,elcs,attRef,state):
        attributeInfo = elcs.env.formatData.attributeInfo[attRef]

        #Continuous attribute
        if attributeInfo.type:
            attRange = attributeInfo.info[1]-attributeInfo.info[0]
            rangeRadius = random.randint(25, 75) * 0.01 * attRange / 2.0  # Continuous initialization domain radius.
            Low = state[attRef].value - rangeRadius
            High = state[attRef].value + rangeRadius
            condList = np.array([Low,High])
            return ClassifierConditionElement(1,range=condList)

        #Discrete attribute
        else:
            condList = state[attRef].value
            return ClassifierConditionElement(0, value=condList)

    #Matching
    def match(self,state,elcs):
        for i in range(self.condition.size):
            attributeInfo = elcs.env.formatData.attributeInfo[self.specifiedAttList[i]]

            #Continuous
            if attributeInfo.type:
                instanceValue = state[self.specifiedAttList[i]].value
                if self.condition[i].list[0] < instanceValue < self.condition[i].list[1] or instanceValue == elcs.labelMissingData:
                    pass
                else:
                    return False

            #Discrete
            else:
                stateRep = state[self.specifiedAttList[i]].value
                if stateRep == self.condition[i].value or stateRep == elcs.labelMissingData:
                    pass
                else:
                    return False
        return True

    def equals(self,cl):
        phenotypesMatch = False
        if isinstance(cl.phenotype,np.ndarray) and isinstance(self.phenotype,np.ndarray):
            if (cl.phenotype == self.phenotype).all():
                phenotypesMatch = True
        else:
            if cl.phenotype == self.phenotype:
                phenotypesMatch = True

        if phenotypesMatch and cl.specifiedAttList.size == self.specifiedAttList.size:
            clRefs = np.sort(cl.specifiedAttList)
            selfRefs = np.sort(self.specifiedAttList)
            if (clRefs == selfRefs).all():
                for i in range(cl.specifiedAttList.size):
                    tempIndex = np.where(self.specifiedAttList == cl.specifiedAttList[i])[0][0]
                    if (cl.condition[i].type == 1 and self.condition[i].type == 1 and cl.condition[i].list[0] == self.condition[i].list[0] and cl.condition[i].list[1] == self.condition[i].list[1]) or (cl.condition[i].type == 0 and self.condition[i].type == 0 and cl.condition[i].value == self.condition[i].value):
                        pass
                    else:
                        return False
                return True
        return False

    def updateNumerosity(self, num):
        """ Updates the numberosity of the classifier.  Notice that 'num' can be negative! """
        self.numerosity += num

    def updateExperience(self):
        """ Increases the experience of the classifier by one. Once an epoch has completed, rule accuracy can't change."""
        self.matchCount += 1

    def updateCorrect(self):
        """ Increases the correct phenotype tracking by one. Once an epoch has completed, rule accuracy can't change."""
        self.correctCount += 1

    def updateMatchSetSize(self, elcs,matchSetSize):
        """  Updates the average match set size. """
        if self.matchCount < 1.0 / elcs.beta:
            self.aveMatchSetSize = (self.aveMatchSetSize * (self.matchCount - 1) + matchSetSize) / float(
                self.matchCount)
        else:
            self.aveMatchSetSize = self.aveMatchSetSize + elcs.beta * (matchSetSize - self.aveMatchSetSize)

    def updateAccuracy(self):
        """ Update the accuracy tracker """
        self.accuracy = self.correctCount / float(self.matchCount)

    def updateFitness(self,elcs):
        """ Update the fitness parameter. """
        if elcs.env.formatData.discretePhenotype or (self.phenotype[1] - self.phenotype[0]) / elcs.env.formatData.phenotypeRange < 0.5:
            self.fitness = pow(self.accuracy, elcs.nu)
        else:
            if (self.phenotype[1] - self.phenotype[0]) >= elcs.env.formatData.phenotypeRange:
                self.fitness = 0.0
            else:
                self.fitness = math.fabs(pow(self.accuracy, elcs.nu) - (self.phenotype[1] - self.phenotype[0]) / elcs.env.formatData.phenotypeRange)

    def isSubsumer(self,elcs):
        if self.matchCount > elcs.theta_sub and self.accuracy > elcs.acc_sub:
            return True
        return False

    def isMoreGeneral(self,cl,elcs):
        if self.specifiedAttList.size >= cl.specifiedAttList.size:
            return False
        for i in range(self.specifiedAttList.size):
            attributeInfo = elcs.env.formatData.attributeInfo[self.specifiedAttList[i]]
            if self.specifiedAttList[i] not in cl.specifiedAttList:
                return False

            #Continuous
            if attributeInfo.type:
                otherRef = np.where(cl.specifiedAttList == self.specifiedAttList[i])[0][0]
                if self.condition[i].list[0] < cl.condition[otherRef].list[0]:
                    return False
                if self.condition[i].list[1] < cl.condition[i].list[1]:
                    return False
        return True

    def updateTimeStamp(self, ts):
        """ Sets the time stamp of the classifier. """
        self.timeStampGA = ts

    def uniformCrossover(self,elcs,cl):
        if elcs.env.formatData.discretePhenotype or random.random() < 0.5:
            p_self_specifiedAttList = copy.deepcopy(self.specifiedAttList)
            p_cl_specifiedAttList = copy.deepcopy(cl.specifiedAttList)

            # Make list of attribute references appearing in at least one of the parents.-----------------------------
            comboAttList = np.unique(np.concatenate((p_self_specifiedAttList,p_cl_specifiedAttList)))
            for i in comboAttList:
                if not elcs.env.formatData.attributeInfo[i].type:
                    index = np.where(comboAttList==i)[0][0]
                    comboAttList = np.delete(comboAttList,index)

            changed = False
            for attRef in comboAttList:
                attributeInfo = elcs.env.formatData.attributeInfo[attRef]
                probability = 0.5
                ref = 0
                p_self_specifiedAttList+=1
                p_cl_specifiedAttList+=1

                if ref == 0:
                    pass
                elif ref == 1:
                    if attRef in p_self_specifiedAttList and random.random() > probability:
                        i = np.where(self.specifiedAttList == attRef)[0][0]
                        cl.condition = np.append(cl.condition,self.condition[i])
                        self.condition = np.delete(self.condition,i)

                        cl.specifiedAttList = np.append(cl.specifiedAttList,attRef)
                        self.specifiedAttList = np.delete(self.specifiedAttList,i)
                        changed = True

                    if attRef in p_cl_specifiedAttList and random.random() < probability:
                        i = np.where(cl.specifiedAttList == attRef)[0][0]
                        self.condition = np.append(self.condition, cl.condition[i])
                        cl.condition = np.delete(cl.condition, i)

                        self.specifiedAttList = np.append(self.specifiedAttList, attRef)
                        cl.specifiedAttList = np.delete(cl.specifiedAttList, i)
                        changed = True
                else:
                    #Continuous Attribute
                    if attributeInfo.type:
                        i_cl1 = np.where(self.specifiedAttList == attRef)
                        i_cl2 = np.where(cl.specifiedAttList == attRef)
                        tempKey = random.randint(0,3)
                        if tempKey == 0:
                            temp = self.condition[i_cl1].list[0]
                            self.condition[i_cl1].list[0] = cl.condition[i_cl2].list[0]
                            cl.condition[i_cl2].list[0] = temp
                        elif tempKey == 1:
                            temp = self.condition[i_cl1].list[1]
                            self.condition[i_cl1].list[1] = cl.condition[i_cl2].list[1]
                            cl.condition[i_cl2].list[1] = temp
                        else:
                            allList = np.concatenate((self.condition[i_cl1],cl.condition[i_cl2]))
                            newMin = np.amin(allList)
                            newMax = np.amax(allList)
                            if tempKey == 2:
                                self.condition[i_cl1] = ClassifierConditionElement(1,np.array([newMin,newMax]))
                                cl.condition = np.delete(cl.condition,i_cl2)
                                a = np.where(cl.specifiedAttList == attRef)[0][0]
                                cl.specifiedAttList = np.delete(cl.specifiedAttList,a)
                            else:
                                cl.condition[i_cl2] = ClassifierConditionElement(1, np.array([newMin, newMax]))
                                self.condition = np.delete(self.condition, i_cl1)
                                a = np.where(self.specifiedAttList == attRef)[0][0]
                                self.specifiedAttList = np.delete(self.specifiedAttList, a)

                    #Discrete Attribute
                    else:
                        pass

            tempList1 = copy.deepcopy(p_self_specifiedAttList)
            tempList2 = copy.deepcopy(cl.specifiedAttList)
            tempList1 = np.sort(tempList1)
            tempList2 = np.sort(tempList2)

            if changed and (tempList1 == tempList2):
                changed = False

            return changed
        else:
            return self.phenotypeCrossover(cl)


    def phenotypeCrossover(self,cl):
        changed = False
        if (self.phenotype[0] == cl.phenotype[0] and self.phenotype[1] == cl.phenotype[1]):
            return changed
        else:
            tempKey = random.random() < 0.5  # Make random choice between 4 scenarios, Swap minimums, Swap maximums, Children preserve parent phenotypes.
            if tempKey:  # Swap minimum
                temp = self.phenotype[0]
                self.phenotype[0] = cl.phenotype[0]
                cl.phenotype[0] = temp
                changed = True
            elif tempKey:  # Swap maximum
                temp = self.phenotype[1]
                self.phenotype[1] = cl.phenotype[1]
                cl.phenotype[1] = temp
                changed = True

        return changed

    def setAccuracy(self, acc):
        """ Sets the accuracy of the classifier """
        self.accuracy = acc

    def setFitness(self, fit):
        """  Sets the fitness of the classifier. """
        self.fitness = fit

    def Mutation(self,elcs,state,phenotype):
        changed = False
        #Mutate Condition
        for attRef in range(elcs.env.formatData.numAttributes):
            attributeInfo = elcs.env.formatData.attributeInfo[attRef]
            if random.random() < elcs.upsilon and state[attRef].value != elcs.labelMissingData:
                #Mutation
                if attRef not in self.specifiedAttList:
                    self.specifiedAttList = np.append(self.specifiedAttList,attRef)
                    self.condition = np.append(self.condition,self.buildMatch(attRef,state))
                    changed = True
                elif attRef in self.specifiedAttList:
                    i = np.where(self.specifiedAttList == attRef)
                    if not attributeInfo.type or random.random > 0.5:
                        self.specifiedAttList = np.delete(self.specifiedAttList,i)
                        self.condition = np.delete(self.condition,i)
                        changed = True
                    else:
                        attRange = float(attributeInfo.info[1]) - float(attributeInfo.info[0])
                        mutateRange = random.random() * 0.5 * attRange
                        if random.random() > 0.5:
                            if random.random() > 0.5:
                                self.condition[i].list[0] += mutateRange
                            else:
                                self.condition[i].list[0] -= mutateRange
                        else:
                            if random.random() > 0.5:
                                self.condition[i].list[1] += mutateRange
                            else:
                                self.condition[i].list[1] -= mutateRange
                        self.condition[i].list = np.sort(self.condition[i].list)
                        changed = True

                else:
                    pass

        #Mutate Phenotype
        if elcs.env.formatData.discretePhenotype:
            nowChanged = self.discretePhenotypeMutation(elcs)
        else:
            nowChanged = self.continuousPhenotypeMutation(elcs,phenotype)

        if changed or nowChanged:
            return True

    def discretePhenotypeMutation(self,elcs):
        changed = False
        if random.random() < elcs.upsilon:
            phenotypeList = copy.deepcopy(elcs.env.formatData.phenotypeList)
            index = np.where(phenotypeList == self.phenotype)
            phenotypeList = np.delete(phenotypeList,index)
            newPhenotype = np.random.choice(phenotypeList)
            self.phenotype = newPhenotype
            changed = True
        return changed

    def continuousPhenotypeMutation(self,elcs,phenotype):
        changed = False
        if random.random() < elcs.upsilon:
            phenRange = self.phenotype[1] - self.phenotype[0]
            mutateRange = random.random() * 0.5 * phenRange
            tempKey = random.randint(0,2)  # Make random choice between 3 scenarios, mutate minimums, mutate maximums, mutate both
            if tempKey == 0:  # Mutate minimum
                if random.random() > 0.5 or self.phenotype[
                    0] + mutateRange <= phenotype:  # Checks that mutated range still contains current phenotype
                    self.phenotype[0] += mutateRange
                else:  # Subtract
                    self.phenotype[0] -= mutateRange
                changed = True
            elif tempKey == 1:  # Mutate maximum
                if random.random() > 0.5 or self.phenotype[
                    1] - mutateRange >= phenotype:  # Checks that mutated range still contains current phenotype
                    self.phenotype[1] -= mutateRange
                else:  # Subtract
                    self.phenotype[1] += mutateRange
                changed = True
            else:  # mutate both
                if random.random() > 0.5 or self.phenotype[
                    0] + mutateRange <= phenotype:  # Checks that mutated range still contains current phenotype
                    self.phenotype[0] += mutateRange
                else:  # Subtract
                    self.phenotype[0] -= mutateRange
                if random.random() > 0.5 or self.phenotype[
                    1] - mutateRange >= phenotype:  # Checks that mutated range still contains current phenotype
                    self.phenotype[1] -= mutateRange
                else:  # Subtract
                    self.phenotype[1] += mutateRange
                changed = True
            self.phenotype = np.sort(self.phenotype)
        return changed

    def subsumes(self,elcs,cl):
        #Discrete Phenotype
        if elcs.env.formatData.discretePhenotype:
            if cl.phenotype == self.phenotype:
                if self.isSubsumer(elcs) and self.isMoreGeneral(cl,elcs):
                    return True
            return False

        #Continuous Phenotype
        else:
            if self.phenotype[0] >= cl.phenotype[0] and self.phenotype[1] <= cl.phenotype[1]:
                if self.isSubsumer(elcs) and self.isMoreGeneral(cl,elcs):
                    return True
                return False

    def getDelProp(self, elcs,meanFitness):
        """  Returns the vote for deletion of the classifier. """
        if self.fitness / self.numerosity >= elcs.delta * meanFitness or self.matchCount < elcs.theta_del:
            self.deletionVote = self.aveMatchSetSize * self.numerosity

        elif self.fitness == 0.0:
            self.deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (elcs.init_fit / self.numerosity)
        else:
            self.deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (self.fitness / self.numerosity)
        return self.deletionVote

class ClassifierConditionElement():
    def __init__(self,type,value=0,range=np.array([])):
        if type == 0:
            self.type = 0
            self.value = value
        else:
            self.type = 1
            self.list = range