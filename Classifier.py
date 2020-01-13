from eLCS import *
import random
import copy
import math
import numpy as np
from DynamicNPArray import TupleArray

class Classifier():
    def __init__(self,elcs,a=None,b=None,c=None,d=None):
        #Major Parameters
        self.specifiedAttList = TupleArray(k=1)
        self.conditionType = TupleArray(k=1) #0 for discrete, 1 for continuous
        self.conditionDiscrete = TupleArray(k=1) #discrete values
        self.conditionContinuous = TupleArray(k=2) #continouous values
        self.phenotype = None #arbitrary

        self.fitness = elcs.init_fit
        self.accuracy = 0.0
        self.numerosity = 1
        self.aveMatchSetSize = None
        self.deletionVote = None

        # Experience Management
        self.timeStampGA = None
        self.initTimeStamp = None

        # Classifier Accuracy Tracking --------------------------------------
        self.matchCount = 0  # Known in many LCS implementations as experience i.e. the total number of times this classifier was in a match set
        self.correctCount = 0  # The total number of times this classifier was in a correct set

        if isinstance(c, TupleArray):
            self.classifierCovering(elcs, a, b, c, d)
        elif isinstance(a, Classifier):
            self.classifierCopy(a, b)

    # Classifier Construction Methods
    def classifierCovering(self, elcs, setSize, exploreIter, state, phenotype):
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
            phenotypeRange = dataInfo.phenotypeList.a[1,0] - dataInfo.phenotypeList.a[0,0]
            rangeRadius = random.randint(25,75) * 0.01 * phenotypeRange / 2.0  # Continuous initialization domain radius.
            Low = float(phenotype) - rangeRadius
            High = float(phenotype) + rangeRadius
            self.phenotype = TupleArray(np.array([Low, High]))

        while self.specifiedAttList.size() < 1:
            for attRef in range(state.size()):
                if random.random() < elcs.p_spec and not(np.isnan(state.a[attRef,0])):
                    # print("B",end="")
                    self.specifiedAttList.append(attRef)
                    self.buildMatch(elcs, attRef, state)  # Add classifierConditionElement

    def classifierCopy(self, toCopy, exploreIter):
        self.specifiedAttList = copy.deepcopy(toCopy.specifiedAttList)
        self.conditionType = copy.deepcopy(toCopy.conditionType)
        self.conditionDiscrete = copy.deepcopy(toCopy.conditionDiscrete)
        self.conditionContinuous = copy.deepcopy(toCopy.conditionContinuous)

        self.phenotype = copy.deepcopy(toCopy.phenotype)
        self.timeStampGA = exploreIter
        self.initTimeStamp = exploreIter
        self.aveMatchSetSize = copy.deepcopy(toCopy.aveMatchSetSize)
        self.fitness = toCopy.fitness
        self.accuracy = toCopy.accuracy

    def buildMatch(self, elcs, attRef, state):
        attributeInfoType = elcs.env.formatData.attributeInfoType.a[attRef,0]
        if not(attributeInfoType): #Discrete
            attributeInfoValue = elcs.env.formatData.attributeInfoDiscrete.a[attRef,0]
        else:
            attributeInfoValue = elcs.env.formatData.attributeInfoContinuous.a[attRef,0]

        # Continuous attribute
        if attributeInfoType:
            attRange = attributeInfoValue[1] - attributeInfoValue[0]
            rangeRadius = random.randint(25, 75) * 0.01 * attRange / 2.0  # Continuous initialization domain radius.
            ar = state.a[attRef,0]
            Low = ar - rangeRadius
            High = ar + rangeRadius
            condList = np.array([Low, High])

            self.conditionContinuous.append(condList)
            self.conditionDiscrete.append(np.nan)
            self.conditionType.append(1)

        # Discrete attribute
        else:
            condList = state.a[attRef,0]

            self.conditionContinuous.append(np.array([np.nan,np.nan]))
            self.conditionDiscrete.append(condList)
            self.conditionType.append(0)

    # Matching
    def match(self, state, elcs):
        for i in range(self.conditionDiscrete.size()):
            specifiedIndex = self.specifiedAttList.a[i,0]
            attributeInfoType = elcs.env.formatData.attributeInfoType.a[specifiedIndex,0]
            if not (attributeInfoType):  # Discrete
                attributeInfoValue = elcs.env.formatData.attributeInfoDiscrete.a[specifiedIndex,0]
            else:
                attributeInfoValue = elcs.env.formatData.attributeInfoContinuous.a[specifiedIndex,0]

            # Continuous
            if attributeInfoType:
                instanceValue = state.a[specifiedIndex,0]
                if np.isnan(instanceValue):
                    pass
                elif self.conditionContinuous.a[i,0] < instanceValue < self.conditionContinuous.a[i,1]:
                    pass
                else:
                    return False

            # Discrete
            else:
                stateRep = state.a[specifiedIndex,0]
                if stateRep == self.conditionDiscrete.a[i,0] or np.isnan(stateRep):
                    pass
                else:
                    return False
        return True

    def equals(self, elcs, cl):
        phenotypesMatch = False
        if not elcs.env.formatData.discretePhenotype:
            if (cl.phenotype.getArray() == self.phenotype.getArray()).all():
                phenotypesMatch = True
        else:
            if cl.phenotype == self.phenotype:
                phenotypesMatch = True

        if phenotypesMatch and cl.specifiedAttList.size() == self.specifiedAttList.size():
            clRefs = np.sort(cl.specifiedAttList.getArray())
            selfRefs = np.sort(self.specifiedAttList.getArray())
            if (clRefs == selfRefs).all():
                for i in range(cl.specifiedAttList.size()):
                    tempIndex = np.where(self.specifiedAttList.getArray() == cl.specifiedAttList.getArray[i])[0][0]
                    if not ((cl.conditionType.a[i,0] == 1 and self.conditionType.a[tempIndex,0] == 1 and cl.conditionContinuous.a[i,0] == self.conditionContinuous.a[tempIndex,0] and cl.conditionContinuous.a[i,1] == self.conditionContinuous.a[tempIndex,1]) or
                            (cl.conditionType.a[i,0] == 0 and self.conditionType.a[tempIndex,0] == 0 and cl.conditionDiscrete.a[i,0] == self.conditionDiscrete.a[tempIndex,0])):
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

    def updateMatchSetSize(self, elcs, matchSetSize):
        """  Updates the average match set size. """
        if self.matchCount < 1.0 / elcs.beta:
            self.aveMatchSetSize = (self.aveMatchSetSize * (self.matchCount - 1) + matchSetSize) / float(
                self.matchCount)
        else:
            self.aveMatchSetSize = self.aveMatchSetSize + elcs.beta * (matchSetSize - self.aveMatchSetSize)

    def updateAccuracy(self):
        """ Update the accuracy tracker """
        self.accuracy = self.correctCount / float(self.matchCount)

    def updateFitness(self, elcs):
        """ Update the fitness parameter. """
        if elcs.env.formatData.discretePhenotype or (
                self.phenotype.a[1,0] - self.phenotype.a[0,0]) / elcs.env.formatData.phenotypeRange < 0.5:
            self.fitness = pow(self.accuracy, elcs.nu)
        else:
            if (self.phenotype.a[1,0] - self.phenotype.a[0,0]) >= elcs.env.formatData.phenotypeRange:
                self.fitness = 0.0
            else:
                self.fitness = math.fabs(pow(self.accuracy, elcs.nu) - (
                            self.phenotype.a[1,0] - self.phenotype.a[0,0]) / elcs.env.formatData.phenotypeRange)

    def isSubsumer(self, elcs):
        if self.matchCount > elcs.theta_sub and self.accuracy > elcs.acc_sub:
            return True
        return False

    def isMoreGeneral(self, cl, elcs):
        if self.specifiedAttList.size() >= cl.specifiedAttList.size():
            return False
        for i in range(self.specifiedAttList.size()):
            attributeInfoType = elcs.env.formatData.attributeInfoType.a[self.specifiedAttList.a[i,0],0]
            if self.specifiedAttList.a[i,0] not in cl.specifiedAttList.getArray():
                return False

            # Continuous
            if attributeInfoType:
                otherRef = np.where(cl.specifiedAttList.getArray() == self.specifiedAttList.getArray()[i])[0][0]
                if self.conditionContinuous.a[i,0] < cl.conditionContinuous.a[otherRef,0]:
                    return False
                if self.conditionContinuous.a[i,1] > cl.conditionContinuous.a[otherRef,1]:
                    return False
        return True

    def uniformCrossover(self, elcs, cl):
        if elcs.env.formatData.discretePhenotype or random.random() < 0.5:
            p_self_specifiedAttList = copy.deepcopy(self.specifiedAttList)
            p_cl_specifiedAttList = copy.deepcopy(cl.specifiedAttList)
            pSelfList = p_self_specifiedAttList.getArray()
            pClList = p_cl_specifiedAttList.getArray()

            # Make list of attribute references appearing in at least one of the parents.-----------------------------
            comboAttList = TupleArray(k=1)
            for i in pSelfList:
                comboAttList.append(i)
            for i in pClList:
                if i not in comboAttList.getArray():
                    comboAttList.append(i)
                elif not elcs.env.formatData.attributeInfoType.a[i,0]:
                    index = np.where(comboAttList.getArray() == i)[0][0]
                    comboAttList.removeAtIndex(index)
            comboAttList = TupleArray(np.sort(comboAttList.getArray()))

            changed = False
            for attRef in comboAttList.getArray():
                attributeInfoType = elcs.env.formatData.attributeInfoType.a[attRef,0]
                probability = 0.5
                ref = 0
                if attRef in pSelfList:
                    ref += 1
                if attRef in pClList:
                    ref += 1

                if ref == 0:
                    pass
                elif ref == 1:
                    if attRef in pSelfList and random.random() > probability:
                        i = np.where(self.specifiedAttList.getArray() == attRef)[0][0]
                        cl.conditionType.append(self.conditionType.a[i,0])
                        cl.conditionDiscrete.append(self.conditionDiscrete.a[i,0])
                        cl.conditionContinuous.append(self.conditionContinuous.a[i,0])
                        self.conditionType.removeAtIndex(i)
                        self.conditionDiscrete.removeAtIndex(i)
                        self.conditionContinuous.removeAtIndex(i)

                        cl.specifiedAttList.append(attRef)
                        self.specifiedAttList.removeAtIndex(i)
                        changed = True

                    if attRef in pClList and random.random() < probability:
                        i = np.where(cl.specifiedAttList.getArray() == attRef)[0][0]
                        self.conditionType.append(cl.conditionType.a[i,0])
                        self.conditionDiscrete.append(cl.conditionDiscrete.a[i,0])
                        self.conditionContinuous.append(cl.conditionContinuous.a[i,0])
                        cl.conditionType.removeAtIndex(i)
                        cl.conditionDiscrete.removeAtIndex(i)
                        cl.conditionContinuous.removeAtIndex(i)

                        self.specifiedAttList.append(attRef)
                        cl.specifiedAttList.removeAtIndex(i)
                        changed = True
                else:
                    # Continuous Attribute
                    if attributeInfoType:
                        i_cl1 = np.where(self.specifiedAttList.getArray() == attRef)[0][0]
                        i_cl2 = np.where(cl.specifiedAttList.getArray() == attRef)[0][0]
                        tempKey = random.randint(0, 3)
                        if tempKey == 0:
                            temp = self.conditionContinuous.a[i_cl1,0]
                            self.conditionContinuous.a[i_cl1,0] = cl.conditionContinuous.a[i_cl2,0]
                            cl.conditionContinuous.a[i_cl2,0] = temp
                        elif tempKey == 1:
                            temp = self.conditionContinuous.a[i_cl1,1]
                            self.conditionContinuous.a[i_cl1,1] = cl.conditionContinuous.a[i_cl2,1]
                            cl.conditionContinuous.a[i_cl2,1] = temp
                        else:
                            allList = np.concatenate((self.conditionContinuous.a[i_cl1], cl.conditionContinuous.a[i_cl2]))
                            newMin = np.amin(allList)
                            newMax = np.amax(allList)
                            if tempKey == 2:
                                self.conditionContinuous.a[i_cl1] = np.array([newMin, newMax])
                                cl.conditionType.removeAtIndex(i_cl2)
                                cl.conditionContinuous.removeAtIndex(i_cl2)
                                cl.conditionDiscrete.removeAtIndex(i_cl2)

                                cl.specifiedAttList.removeFirstElementWithValue(attRef)
                            else:
                                cl.conditionContinuous.a[i_cl2] = np.array([newMin, newMax])
                                self.conditionType.removeAtIndex(i_cl1)
                                self.conditionContinuous.removeAtIndex(i_cl1)
                                self.conditionDiscrete.removeAtIndex(i_cl1)

                                self.specifiedAttList.removeFirstElementWithValue(attRef)

                    # Discrete Attribute
                    else:
                        pass

            tempList1 = copy.deepcopy(p_self_specifiedAttList.getArray())
            tempList2 = copy.deepcopy(cl.specifiedAttList.getArray())
            tempList1 = np.sort(tempList1)
            tempList2 = np.sort(tempList2)

            # if changed:
            # print("CHANGED")
            # print(tempList1)
            # print(tempList2)

            if changed and len(set(tempList1) & set(tempList2)) == tempList2.size:
                # print("PASS")
                changed = False

            return changed
        else:
            return self.phenotypeCrossover(cl)

    def phenotypeCrossover(self, cl):
        changed = False
        if (self.phenotype.a[0,0] == cl.phenotype.a[0,0] and self.phenotype.a[1,0] == cl.phenotype.a[1,0]):
            return changed
        else:
            tempKey = random.random() < 0.5  # Make random choice between 4 scenarios, Swap minimums, Swap maximums, Children preserve parent phenotypes.
            if tempKey:  # Swap minimum
                temp = self.phenotype.a[0,0]
                self.phenotype.a[0,0] = cl.phenotype.a[0,0]
                cl.phenotype.a[0,0] = temp
                changed = True
            elif tempKey:  # Swap maximum
                temp = self.phenotype.a[1,0]
                self.phenotype.a[1,0] = cl.phenotype.a[1,0]
                cl.phenotype[1,0] = temp
                changed = True

        return changed

    def Mutation(self, elcs, state, phenotype):
        changed = False
        # Mutate Condition
        for attRef in range(elcs.env.formatData.numAttributes):
            attributeInfoType = elcs.env.formatData.attributeInfoType.a[attRef,0]
            if not (attributeInfoType):  # Discrete
                attributeInfoValue = elcs.env.formatData.attributeInfoDiscrete.a[attRef,0]
            else:
                attributeInfoValue = elcs.env.formatData.attributeInfoContinuous.a[attRef,0]

            if random.random() < elcs.upsilon and not(np.isnan(state.a[attRef,0])):
                # Mutation
                if attRef not in self.specifiedAttList.getArray():
                    self.specifiedAttList.append(attRef)
                    self.buildMatch(elcs, attRef, state)
                    changed = True
                elif attRef in self.specifiedAttList.getArray():
                    i = np.where(self.specifiedAttList.getArray() == attRef)[0][0]

                    if not attributeInfoType or random.random() > 0.5:
                        self.specifiedAttList.removeAtIndex(i)
                        self.conditionType.removeAtIndex(i)
                        self.conditionDiscrete.removeAtIndex(i)
                        self.conditionContinuous.removeAtIndex(i)
                        changed = True
                    else:
                        attRange = float(attributeInfoValue[1]) - float(attributeInfoValue[0])
                        mutateRange = random.random() * 0.5 * attRange
                        if random.random() > 0.5:
                            if random.random() > 0.5:
                                self.conditionContinuous.a[i,0] += mutateRange
                            else:
                                self.conditionContinuous.a[i,0] -= mutateRange
                        else:
                            if random.random() > 0.5:
                                self.conditionContinuous.a[i,1] += mutateRange
                            else:
                                self.conditionContinuous.a[i,1] -= mutateRange
                        self.conditionContinuous.a[i] = np.sort(self.conditionContinuous.a[i])
                        changed = True

                else:
                    pass

        # Mutate Phenotype
        if elcs.env.formatData.discretePhenotype:
            nowChanged = self.discretePhenotypeMutation(elcs)
        else:
            nowChanged = self.continuousPhenotypeMutation(elcs, phenotype)

        if changed or nowChanged:
            return True

    def discretePhenotypeMutation(self, elcs):
        changed = False
        if random.random() < elcs.upsilon:
            phenotypeList = copy.deepcopy(elcs.env.formatData.phenotypeList)
            index = np.where(phenotypeList.getArray() == self.phenotype)
            phenotypeList.removeAtIndex(index)
            newPhenotype = np.random.choice(phenotypeList.getArray())
            self.phenotype = TupleArray(newPhenotype)
            changed = True
        return changed

    def continuousPhenotypeMutation(self, elcs, phenotype):
        changed = False
        if random.random() < elcs.upsilon:
            phenRange = self.phenotype.a[1,0] - self.phenotype.a[0,0]
            mutateRange = random.random() * 0.5 * phenRange
            tempKey = random.randint(0,2)  # Make random choice between 3 scenarios, mutate minimums, mutate maximums, mutate both
            if tempKey == 0:  # Mutate minimum
                if random.random() > 0.5 or self.phenotype.a[
                    0,0] + mutateRange <= phenotype:  # Checks that mutated range still contains current phenotype
                    self.phenotype.a[0,0] += mutateRange
                else:  # Subtract
                    self.phenotype.a[0,0] -= mutateRange
                changed = True
            elif tempKey == 1:  # Mutate maximum
                if random.random() > 0.5 or self.phenotype.a[
                    1,0] - mutateRange >= phenotype:  # Checks that mutated range still contains current phenotype
                    self.phenotype.a[1,0] -= mutateRange
                else:  # Subtract
                    self.phenotype.a[1,0] += mutateRange
                changed = True
            else:  # mutate both
                if random.random() > 0.5 or self.phenotype.a[
                    0,0] + mutateRange <= phenotype:  # Checks that mutated range still contains current phenotype
                    self.phenotype.a[0,0] += mutateRange
                else:  # Subtract
                    self.phenotype.a[0,0] -= mutateRange
                if random.random() > 0.5 or self.phenotype.a[
                    1,0] - mutateRange >= phenotype:  # Checks that mutated range still contains current phenotype
                    self.phenotype.a[1,0] -= mutateRange
                else:  # Subtract
                    self.phenotype.a[1,0] += mutateRange
                changed = True
            self.phenotype = TupleArray(np.sort(self.phenotype.getArray()))
        return changed

    def updateTimeStamp(self, ts):
        """ Sets the time stamp of the classifier. """
        self.timeStampGA = ts

    def setAccuracy(self, acc):
        """ Sets the accuracy of the classifier """
        self.accuracy = acc

    def setFitness(self, fit):
        """  Sets the fitness of the classifier. """
        self.fitness = fit

    def subsumes(self, elcs, cl):
        # Discrete Phenotype
        if elcs.env.formatData.discretePhenotype:
            if cl.phenotype == self.phenotype:
                if self.isSubsumer(elcs) and self.isMoreGeneral(cl, elcs):
                    return True
            return False

        # Continuous Phenotype
        else:
            if self.phenotype.a[0,0] >= cl.phenotype.a[0,0] and self.phenotype.a[1,0] <= cl.phenotype.a[1,0]:
                if self.isSubsumer(elcs) and self.isMoreGeneral(cl, elcs):
                    return True
                return False

    def getDelProp(self, elcs, meanFitness):
        """  Returns the vote for deletion of the classifier. """
        if self.fitness / self.numerosity >= elcs.delta * meanFitness or self.matchCount < elcs.theta_del:
            self.deletionVote = self.aveMatchSetSize * self.numerosity

        elif self.fitness == 0.0:
            self.deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (
                        elcs.init_fit / self.numerosity)
        else:
            self.deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (
                        self.fitness / self.numerosity)
        return self.deletionVote