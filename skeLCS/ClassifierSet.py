from skeLCS.Classifier import Classifier
import random
import copy

class ClassifierSet:
    def __init__(self,elcs):
        #Major Parameters
        self.popSet = []
        self.matchSet = []
        self.correctSet = []
        self.microPopSize = 0

        #Evaluation Parameters
        self.aveGenerality = 0.0
        self.expRules = 0.0
        self.attributeSpecList = None
        self.attributeAccList = None
        self.avePhenotypeRange = 0.0

    def makeMatchSet(self,state_phenotype,exploreIter,elcs):
        state = state_phenotype[0]
        phenotype = state_phenotype[1]
        doCovering = True
        setNumerositySum = 0

        #Matching
        if not elcs.hasTrained:
            elcs.timer.startTimeMatching()
        for i in range(len(self.popSet)):
            cl = self.popSet[i]
            if cl.match(state,elcs):
                self.matchSet.append(i)
                setNumerositySum += cl.numerosity

                #Covering Check
                if elcs.env.formatData.discretePhenotype:
                    if cl.phenotype == phenotype:
                        doCovering = False
                else:
                    if float(cl.phenotype[0]) <= float(phenotype) <= float(cl.phenotype[1]):
                        doCovering = False
        if not elcs.hasTrained:
            elcs.timer.stopTimeMatching()

        #Covering
        while doCovering:
            newCl = Classifier(elcs,setNumerositySum+1,exploreIter,state,phenotype)
            self.addClassifierToPopulation(elcs,newCl,True)
            self.matchSet.append(len(self.popSet) - 1)
            elcs.trackingObj.coveringCount+=1
            doCovering = False

    def getIdenticalClassifier(self,elcs,newCl):
        for cl in self.popSet:
            if newCl.equals(elcs,cl):
                return cl
        return None

    def addClassifierToPopulation(self,elcs,cl,covering):
        oldCl = None
        if not covering:
            oldCl = self.getIdenticalClassifier(elcs,cl)
        if oldCl != None:
            oldCl.updateNumerosity(1)
            self.microPopSize += 1
        else:
            self.popSet.append(cl)
            self.microPopSize += 1

    def makeCorrectSet(self,elcs,phenotype):
        for i in range(len(self.matchSet)):
            ref = self.matchSet[i]
            #Discrete Phenotype
            if elcs.env.formatData.discretePhenotype:
                if self.popSet[ref].phenotype == phenotype:
                    self.correctSet.append(ref)

            #Continuous Phenotype
            else:
                if float(phenotype) <= float(self.popSet[ref].phenotype[1]) and float(phenotype) >= float(self.popSet[ref].phenotype[0]):
                    self.correctSet.append(ref)

    def updateSets(self,elcs,exploreIter):
        matchSetNumerosity = 0
        for ref in self.matchSet:
            matchSetNumerosity += self.popSet[ref].numerosity

        for ref in self.matchSet:
            self.popSet[ref].updateExperience()
            self.popSet[ref].updateMatchSetSize(elcs,matchSetNumerosity)
            if ref in self.correctSet:
                self.popSet[ref].updateCorrect()

            self.popSet[ref].updateAccuracy()
            self.popSet[ref].updateFitness(elcs)

    def doCorrectSetSubsumption(self,elcs):
        subsumer = None
        for ref in self.correctSet:
            cl = self.popSet[ref]
            if cl.isSubsumer(elcs):
                if subsumer == None or cl.isMoreGeneral(subsumer,elcs):
                    subsumer = cl

        if subsumer != None:
            i = 0
            while i < len(self.correctSet):
                ref = self.correctSet[i]
                if subsumer.isMoreGeneral(self.popSet[ref],elcs):
                    elcs.trackingObj.subsumptionCount += 1
                    subsumer.updateNumerosity(self.popSet[ref].numerosity)
                    self.removeMacroClassifier(ref)
                    self.deleteFromMatchSet(ref)
                    self.deleteFromCorrectSet(ref)
                    i -= 1
                i+=1

    def removeMacroClassifier(self,ref):
        del self.popSet[ref]

    def deleteFromMatchSet(self,deleteRef):
        if deleteRef in self.matchSet:
            self.matchSet.remove(deleteRef)

        for j in range(len(self.matchSet)):
            ref = self.matchSet[j]
            if ref > deleteRef:
                self.matchSet[j] -=1

    def deleteFromCorrectSet(self,deleteRef):
        if deleteRef in self.correctSet:
            self.correctSet.remove(deleteRef)

        for j in range(len(self.correctSet)):
            ref = self.correctSet[j]
            if ref > deleteRef:
                self.correctSet[j] -= 1

    def runGA(self,elcs,exploreIter,state,phenotype):
        #GA Run Requirement
        if (exploreIter - self.getIterStampAverage()) < elcs.theta_GA:
            return
        if not elcs.hasTrained:
            elcs.timer.startTimeSelection()

        self.setIterStamps(exploreIter)
        changed = False

        #Select Parents
        if elcs.selectionMethod == "roulette":
            selectList = self.selectClassifierRW()
            clP1 = selectList[0]
            clP2 = selectList[1]
        elif elcs.selectionMethod == "tournament":
            selectList = self.selectClassifierT(elcs)
            clP1 = selectList[0]
            clP2 = selectList[1]

        if not elcs.hasTrained:
            elcs.timer.stopTimeSelection()

        #Initialize Offspring
        cl1 = Classifier(elcs,clP1,exploreIter)
        if clP2 == None:
            cl2 = Classifier(elcs,clP1, exploreIter)
        else:
            cl2 = Classifier(elcs,clP2, exploreIter)

        #Crossover Operator (uniform crossover)
        if not cl1.equals(elcs,cl2) and random.random() < elcs.chi:
            changed = cl1.uniformCrossover(elcs,cl2)

        #Initialize Key Offspring Parameters
        if changed:
            cl1.setAccuracy((cl1.accuracy + cl2.accuracy) / 2.0)
            cl1.setFitness(elcs.fitnessReduction * (cl1.fitness + cl2.fitness) / 2.0)
            cl2.setAccuracy(cl1.accuracy)
            cl2.setFitness(cl1.fitness)
        else:
            cl1.setFitness(elcs.fitnessReduction * cl1.fitness)
            cl2.setFitness(elcs.fitnessReduction * cl2.fitness)

        #Mutation Operator
        nowchanged = cl1.Mutation(elcs,state,phenotype)
        howaboutnow = cl2.Mutation(elcs,state,phenotype)

        #Add offspring to population
        if changed or nowchanged or howaboutnow:
            if nowchanged:
                elcs.trackingObj.mutationCount += 1
            if howaboutnow:
                elcs.trackingObj.mutationCount += 1
            if changed:
                elcs.trackingObj.crossOverCount += 1

            self.insertDiscoveredClassifiers(elcs,cl1, cl2, clP1, clP2, exploreIter)  # Subsumption

    def getIterStampAverage(self):
        sumCl = 0.0
        numSum = 0.0
        for i in range(len(self.correctSet)):
            ref = self.correctSet[i]
            sumCl += self.popSet[ref].timeStampGA * self.popSet[ref].numerosity
            numSum += self.popSet[ref].numerosity
        if numSum != 0:
            return sumCl/float(numSum)
        else:
            return 0

    def setIterStamps(self,exploreIter):
        for i in range(len(self.correctSet)):
            ref = self.correctSet[i]
            self.popSet[ref].updateTimeStamp(exploreIter)

    def selectClassifierRW(self):
        setList = copy.deepcopy(self.correctSet)

        if len(setList) > 2:
            selectList = [None,None]
            currentCount = 0

            while currentCount < 2:
                fitSum = self.getFitnessSum(setList)

                choiceP = random.random() * fitSum
                i = 0
                sumCl = self.popSet[setList[i]].fitness
                while choiceP > sumCl:
                    i = i + 1
                    sumCl += self.popSet[setList[i]].fitness

                selectList[currentCount] = self.popSet[setList[i]]
                setList.remove(setList[i])
                currentCount += 1

        elif len(setList) == 2:
            selectList = [self.popSet[setList[0]], self.popSet[setList[1]]]
        elif len(setList) == 1:
            selectList = [self.popSet[setList[0]], self.popSet[setList[0]]]

        return selectList

    def getFitnessSum(self, setList):
        """ Returns the sum of the fitnesses of all classifiers in the set. """
        sumCl = 0.0
        for i in range(len(setList)):
            ref = setList[i]
            sumCl += self.popSet[ref].fitness
        return sumCl

    def selectClassifierT(self,elcs):
        selectList = [None, None]
        currentCount = 0
        setList = self.correctSet

        while currentCount < 2:
            tSize = int(len(setList) * elcs.theta_sel)

            #Select tSize elements from correctSet
            posList = random.sample(setList,tSize)

            bestF = 0
            bestC = self.correctSet[0]
            for j in posList:
                if self.popSet[j].fitness > bestF:
                    bestF = self.popSet[j].fitness
                    bestC = j

            selectList[currentCount] = self.popSet[bestC]
            currentCount += 1

        return selectList

    def insertDiscoveredClassifiers(self,elcs,cl1,cl2,clP1,clP2,exploreIter):
        if elcs.doSubsumption:
            if not elcs.hasTrained:
                elcs.timer.startTimeSubsumption()
            if len(cl1.specifiedAttList) > 0:
                self.subsumeClassifier(elcs,cl1,clP1,clP2)
            if len(cl2.specifiedAttList) > 0:
                self.subsumeClassifier(elcs,cl2, clP1, clP2)
            if not elcs.hasTrained:
                elcs.timer.stopTimeSubsumption()
        else:
            if len(cl1.specifiedAttList) > 0:
                self.addClassifierToPopulation(elcs,cl1,False)
            if len(cl2.specifiedAttList) > 0:
                self.addClassifierToPopulation(elcs,cl2, False)

    def subsumeClassifier(self,elcs,cl=None,cl1P=None,cl2P=None):
        if cl1P != None and cl1P.subsumes(elcs,cl):
            self.microPopSize += 1
            cl1P.updateNumerosity(1)
            elcs.trackingObj.subsumptionCount+=1
        elif cl2P != None and cl2P.subsumes(elcs,cl):
            self.microPopSize += 1
            cl2P.updateNumerosity(1)
            elcs.trackingObj.subsumptionCount += 1
        else:
            self.subsumeClassifier2(elcs,cl)  # Try to subsume in the correct set.

    def subsumeClassifier2(self,elcs,cl):
        choices = []
        for ref in self.correctSet:
            if self.popSet[ref].subsumes(elcs,cl):
                choices.append(ref)

        if len(choices) > 0:
            choice = int(random.random()*len(choices))
            self.popSet[int(choices[choice])].updateNumerosity(1)
            self.microPopSize += 1
            elcs.trackingObj.subsumptionCount += 1
            return
        self.addClassifierToPopulation(elcs,cl,False)

    def deletion(self,elcs,exploreIter):
        while (self.microPopSize > elcs.N):
            self.deleteFromPopulation(elcs)

    def deleteFromPopulation(self,elcs):
        meanFitness = self.getPopFitnessSum() / float(self.microPopSize)

        sumCl = 0.0
        voteList = []
        for cl in self.popSet:
            vote = cl.getDelProp(elcs,meanFitness)
            sumCl += vote
            voteList.append(vote)
        for cl in self.popSet:
            cl.deletionProb = cl.deletionVote/sumCl
        choicePoint = sumCl * random.random()  # Determine the choice point

        newSum = 0.0
        for i in range(len(voteList)):
            cl = self.popSet[i]
            newSum = newSum + voteList[i]
            if newSum > choicePoint:  # Select classifier for deletion
                # Delete classifier----------------------------------
                cl.updateNumerosity(-1)
                self.microPopSize -= 1
                if cl.numerosity < 1:  # When all micro-classifiers for a given classifier have been depleted.
                    self.removeMacroClassifier(i)
                    self.deleteFromMatchSet(i)
                    self.deleteFromCorrectSet(i)
                    elcs.trackingObj.deletionCount += 1
                return
        return

    def getPopFitnessSum(self):
        """ Returns the sum of the fitnesses of all classifiers in the set. """
        sumCl = 0.0
        for cl in self.popSet:
            sumCl += cl.fitness * cl.numerosity
        return sumCl

    def clearSets(self,elcs):
        """ Clears out references in the match and correct sets for the next learning iteration. """
        self.matchSet = []
        self.correctSet = []

    def runPopAveEval(self,exploreIter,elcs):
        genSum = 0
        agedCount = 0
        for cl in self.popSet:
            genSum += ((elcs.env.formatData.numAttributes - len(cl.condition))/float(elcs.env.formatData.numAttributes))*cl.numerosity
        if self.microPopSize == 0:
            self.aveGenerality = 'NA'
        else:
            self.aveGenerality = genSum/float(self.microPopSize)

        if not elcs.env.formatData.discretePhenotype:
            sumRuleRange = 0
            for cl in self.popSet:
                sumRuleRange += (cl.phenotype[1]-cl.phenotype[0])*cl.numerosity
            phenotypeRange = elcs.env.formatData.phenotypeList[1]-elcs.env.formatData.phenotypeList[0]
            self.avePhenotypeRange = (sumRuleRange / float(self.microPopSize)) / float(phenotypeRange)

    def runAttGeneralitySum(self,isEvaluationSummary,elcs):
        if isEvaluationSummary:
            self.attributeSpecList = []
            self.attributeAccList = []
            for i in range(elcs.env.formatData.numAttributes):
                self.attributeSpecList.append(0)
                self.attributeAccList.append(0.0)
            for cl in self.popSet:
                for ref in cl.specifiedAttList:
                    self.attributeSpecList[ref] += cl.numerosity
                    self.attributeAccList[ref] += cl.numerosity * cl.accuracy

    def makeEvalMatchSet(self,state,elcs):
        for i in range(len(self.popSet)):
            cl = self.popSet[i]
            if cl.match(state,elcs):
                self.matchSet.append(i)