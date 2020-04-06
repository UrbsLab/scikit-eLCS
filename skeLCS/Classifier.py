import random
import copy
import math
import numpy as np
from skeLCS.DynamicNPArray import ArrayFactory, GenericArray

class Classifier:
    def __init__(self,elcs,a=None,b=None,c=None,d=None):
        #Major Parameters
        self.specifiedAttList = []
        self.condition = []
        self.phenotype = None #arbitrary

        self.fitness = elcs.init_fit
        self.accuracy = 0.0
        self.numerosity = 1
        self.aveMatchSetSize = None
        self.deletionVote = None
        self.deletionProb = None

        # Experience Management
        self.timeStampGA = None
        self.initTimeStamp = None

        # Classifier Accuracy Tracking --------------------------------------
        self.matchCount = 0  # Known in many LCS implementations as experience i.e. the total number of times this classifier was in a match set
        self.correctCount = 0  # The total number of times this classifier was in a correct set

        if isinstance(c, list):
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
            phenotypeRange = dataInfo.phenotypeList[1] - dataInfo.phenotypeList[0]
            rangeRadius = random.randint(25,75) * 0.01 * phenotypeRange / 2.0  # Continuous initialization domain radius.
            Low = float(phenotype) - rangeRadius
            High = float(phenotype) + rangeRadius
            self.phenotype = [Low, High]

        while len(self.specifiedAttList) < 1:
            for attRef in range(len(state)):
                if random.random() < elcs.p_spec and not(np.isnan(state[attRef])):
                    self.specifiedAttList.append(attRef)
                    self.buildMatch(elcs, attRef, state)  # Add classifierConditionElement

    def classifierCopy(self, toCopy, exploreIter):
        self.specifiedAttList = copy.deepcopy(toCopy.specifiedAttList)
        self.condition = copy.deepcopy(toCopy.condition)

        self.phenotype = copy.deepcopy(toCopy.phenotype)
        self.timeStampGA = exploreIter
        self.initTimeStamp = exploreIter
        self.aveMatchSetSize = copy.deepcopy(toCopy.aveMatchSetSize)
        self.fitness = toCopy.fitness
        self.accuracy = toCopy.accuracy

    def buildMatch(self, elcs, attRef, state):
        attributeInfoType = elcs.env.formatData.attributeInfoType[attRef]
        if not(attributeInfoType): #Discrete
            attributeInfoValue = elcs.env.formatData.attributeInfoDiscrete[attRef]
        else:
            attributeInfoValue = elcs.env.formatData.attributeInfoContinuous[attRef]

        # Continuous attribute
        if attributeInfoType:
            attRange = attributeInfoValue[1] - attributeInfoValue[0]
            rangeRadius = random.randint(25, 75) * 0.01 * attRange / 2.0  # Continuous initialization domain radius.
            ar = state[attRef]
            Low = ar - rangeRadius
            High = ar + rangeRadius
            condList = [Low, High]
            self.condition.append(condList)

        # Discrete attribute
        else:
            condList = state[attRef]
            self.condition.append(condList)

    # Matching
    def match(self, state, elcs):
        for i in range(len(self.condition)):
            specifiedIndex = self.specifiedAttList[i]
            attributeInfoType = elcs.env.formatData.attributeInfoType[specifiedIndex]

            # Continuous
            if attributeInfoType:
                instanceValue = state[specifiedIndex]
                if elcs.matchForMissingness:
                    if np.isnan(instanceValue):
                        pass
                    elif self.condition[i][0] < instanceValue < self.condition[i][1]:
                        pass
                    else:
                        return False
                else:
                    if np.isnan(instanceValue):
                        return False
                    elif self.condition[i][0] < instanceValue < self.condition[i][1]:
                        pass
                    else:
                        return False

            # Discrete
            else:
                stateRep = state[specifiedIndex]
                if elcs.matchForMissingness:
                    if stateRep == self.condition[i] or np.isnan(stateRep):
                        pass
                    else:
                        return False
                else:
                    if stateRep == self.condition[i]:
                        pass
                    elif np.isnan(stateRep):
                        return False
                    else:
                        return False
        return True

    def equals(self, elcs, cl):
        if cl.phenotype == self.phenotype and len(cl.specifiedAttList) == len(self.specifiedAttList):
            clRefs = sorted(cl.specifiedAttList)
            selfRefs = sorted(self.specifiedAttList)
            if clRefs == selfRefs:
                for i in range(len(cl.specifiedAttList)):
                    tempIndex = self.specifiedAttList.index(cl.specifiedAttList[i])
                    if not (cl.condition[i] == self.condition[tempIndex]):
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
                self.phenotype[1] - self.phenotype[0]) / elcs.env.formatData.phenotypeRange < 0.5:
            self.fitness = pow(self.accuracy, elcs.nu)
        else:
            if (self.phenotype[1] - self.phenotype[0]) >= elcs.env.formatData.phenotypeRange:
                self.fitness = 0.0
            else:
                self.fitness = math.fabs(pow(self.accuracy, elcs.nu) - (
                            self.phenotype[1] - self.phenotype[0]) / elcs.env.formatData.phenotypeRange)

    def isSubsumer(self, elcs):
        if self.matchCount > elcs.theta_sub and self.accuracy > elcs.acc_sub:
            return True
        return False

    def isMoreGeneral(self, cl, elcs):
        if len(self.specifiedAttList) >= len(cl.specifiedAttList):
            return False
        for i in range(len(self.specifiedAttList)):
            attributeInfoType = elcs.env.formatData.attributeInfoType[self.specifiedAttList[i]]
            if self.specifiedAttList[i] not in cl.specifiedAttList:
                return False

            # Continuous
            if attributeInfoType:
                otherRef = cl.specifiedAttList.index(self.specifiedAttList[i])
                if self.condition[i][0] < cl.condition[otherRef][0]:
                    return False
                if self.condition[i][1] > cl.condition[otherRef][1]:
                    return False
        return True

    def uniformCrossover(self, elcs, cl):
        if elcs.env.formatData.discretePhenotype or random.random() < 0.5:
            p_self_specifiedAttList = copy.deepcopy(self.specifiedAttList)
            p_cl_specifiedAttList = copy.deepcopy(cl.specifiedAttList)

            # Make list of attribute references appearing in at least one of the parents.-----------------------------
            comboAttList = []
            for i in p_self_specifiedAttList:
                comboAttList.append(i)
            for i in p_cl_specifiedAttList:
                if i not in comboAttList:
                    comboAttList.append(i)
                elif not elcs.env.formatData.attributeInfoType[i]:
                    comboAttList.remove(i)
            comboAttList.sort()

            changed = False
            for attRef in comboAttList:
                attributeInfoType = elcs.env.formatData.attributeInfoType[attRef]
                probability = 0.5
                ref = 0
                if attRef in p_self_specifiedAttList:
                    ref += 1
                if attRef in p_cl_specifiedAttList:
                    ref += 1

                if ref == 0:
                    pass
                elif ref == 1:
                    if attRef in p_self_specifiedAttList and random.random() > probability:
                        i = self.specifiedAttList.index(attRef)
                        cl.condition.append(self.condition.pop(i))

                        cl.specifiedAttList.append(attRef)
                        self.specifiedAttList.remove(attRef)
                        changed = True

                    if attRef in p_cl_specifiedAttList and random.random() < probability:
                        i = cl.specifiedAttList.index(attRef)
                        self.condition.append(cl.condition.pop(i))

                        self.specifiedAttList.append(attRef)
                        cl.specifiedAttList.remove(attRef)
                        changed = True
                else:
                    # Continuous Attribute
                    if attributeInfoType:
                        i_cl1 = self.specifiedAttList.index(attRef)
                        i_cl2 = cl.specifiedAttList.index(attRef)
                        tempKey = random.randint(0, 3)
                        if tempKey == 0:
                            temp = self.condition[i_cl1][0]
                            self.condition[i_cl1][0] = cl.condition[i_cl2][0]
                            cl.condition[i_cl2][0] = temp
                        elif tempKey == 1:
                            temp = self.condition[i_cl1][1]
                            self.condition[i_cl1][1] = cl.condition[i_cl2][1]
                            cl.condition[i_cl2][1] = temp
                        else:
                            allList = self.condition[i_cl1] + cl.condition[i_cl2]
                            newMin = min(allList)
                            newMax = max(allList)
                            if tempKey == 2:
                                self.condition[i_cl1] = [newMin, newMax]
                                cl.condition.pop(i_cl2)

                                cl.specifiedAttList.remove(attRef)
                            else:
                                cl.condition[i_cl2] = [newMin, newMax]
                                self.condition.pop(i_cl1)

                                self.specifiedAttList.remove(attRef)

                    # Discrete Attribute
                    else:
                        pass

            tempList1 = copy.deepcopy(p_self_specifiedAttList)
            tempList2 = copy.deepcopy(cl.specifiedAttList)
            tempList1.sort()
            tempList2.sort()

            if changed and len(set(tempList1) & set(tempList2)) == len(tempList2):
                changed = False

            return changed
        else:
            return self.phenotypeCrossover(cl)

    def phenotypeCrossover(self, cl):
        changed = False
        if self.phenotype == cl.phenotype:
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

    def Mutation(self, elcs, state, phenotype):
        changed = False
        # Mutate Condition
        for attRef in range(elcs.env.formatData.numAttributes):
            attributeInfoType = elcs.env.formatData.attributeInfoType[attRef]
            if not (attributeInfoType):  # Discrete
                attributeInfoValue = elcs.env.formatData.attributeInfoDiscrete[attRef]
            else:
                attributeInfoValue = elcs.env.formatData.attributeInfoContinuous[attRef]

            if random.random() < elcs.upsilon and not(np.isnan(state[attRef])):
                # Mutation
                if attRef not in self.specifiedAttList:
                    self.specifiedAttList.append(attRef)
                    self.buildMatch(elcs, attRef, state)
                    changed = True
                elif attRef in self.specifiedAttList:
                    i = self.specifiedAttList.index(attRef)

                    if not attributeInfoType or random.random() > 0.5:
                        del self.specifiedAttList[i]
                        del self.condition[i]
                        changed = True
                    else:
                        attRange = float(attributeInfoValue[1]) - float(attributeInfoValue[0])
                        mutateRange = random.random() * 0.5 * attRange
                        if random.random() > 0.5:
                            if random.random() > 0.5:
                                self.condition[i][0] += mutateRange
                            else:
                                self.condition[i][0] -= mutateRange
                        else:
                            if random.random() > 0.5:
                                self.condition[i][1] += mutateRange
                            else:
                                self.condition[i][1] -= mutateRange
                        self.condition[i] = sorted(self.condition[i])
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
            phenotypeList.remove(self.phenotype)
            newPhenotype = random.choice(phenotypeList)
            self.phenotype = newPhenotype
            changed = True
        return changed

    def continuousPhenotypeMutation(self, elcs, phenotype):
        changed = False
        if random.random() < elcs.upsilon:
            phenRange = self.phenotype[1] - self.phenotype[0]
            mutateRange = random.random() * 0.5 * phenRange
            tempKey = random.randint(0,2)  # Make random choice between 3 scenarios, mutate minimums, mutate maximums, mutate both
            if tempKey == 0:  # Mutate minimum
                if random.random() > 0.5 or self.phenotype[0] + mutateRange <= phenotype:  # Checks that mutated range still contains current phenotype
                    self.phenotype[0] += mutateRange
                else:  # Subtract
                    self.phenotype[0] -= mutateRange
                changed = True
            elif tempKey == 1:  # Mutate maximum
                if random.random() > 0.5 or self.phenotype[1] - mutateRange >= phenotype:  # Checks that mutated range still contains current phenotype
                    self.phenotype[1] -= mutateRange
                else:  # Subtract
                    self.phenotype[1] += mutateRange
                changed = True
            else:  # mutate both
                if random.random() > 0.5 or self.phenotype[0] + mutateRange <= phenotype:  # Checks that mutated range still contains current phenotype
                    self.phenotype[0] += mutateRange
                else:  # Subtract
                    self.phenotype[0] -= mutateRange
                if random.random() > 0.5 or self.phenotype[1] - mutateRange >= phenotype:  # Checks that mutated range still contains current phenotype
                    self.phenotype[1] -= mutateRange
                else:  # Subtract
                    self.phenotype[1] += mutateRange
                changed = True
            self.phenotype.sort()
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
            if self.phenotype[0] >= cl.phenotype[0] and self.phenotype[1] <= cl.phenotype[1]:
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