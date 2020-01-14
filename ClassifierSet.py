from Classifier import *
import numpy as np
from DynamicNPArray import ArrayFactory

class ClassifierSet:
    def __init__(self,elcs):
        #Major Parameters
        self.popSet = ArrayFactory.createArray(k=1,dtype=Classifier,minSize=elcs.N)
        self.matchSet = ArrayFactory.createArray(k=1,minSize=elcs.N)
        self.correctSet = ArrayFactory.createArray(k=1,minSize=elcs.N)
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
        for i in range(self.popSet.size()):
            cl = self.popSet.getI(i)
            if cl.match(state,elcs):
                self.matchSet.append(i)
                setNumerositySum += cl.numerosity

                #Covering Check
                if elcs.env.formatData.discretePhenotype:
                    if cl.phenotype == phenotype:
                        doCovering = False
                else:
                    if float(cl.phenotype.getI(0)) <= float(phenotype) <= float(cl.phenotype.getI(1)):
                        doCovering = False

        #Covering
        while doCovering:
            #print("Covering")
            newCl = Classifier(elcs,setNumerositySum+1,exploreIter,state,phenotype)
            self.addClassifierToPopulation(elcs,newCl,True)
            self.matchSet.append(self.popSet.size() - 1)
            elcs.coveringCounter+=1
            doCovering = False

    def getIdenticalClassifier(self,elcs,newCl):
        for cl in self.popSet.getArray():
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
        for i in range(self.matchSet.size()):
            ref = self.matchSet.getI(i)
            #Discrete Phenotype
            if elcs.env.formatData.discretePhenotype:
                if self.popSet.getI(ref).phenotype == phenotype:
                    self.correctSet.append(ref)

            #Continuous Phenotype
            else:
                if float(phenotype.getI(1)) <= float(self.popSet.getI(ref).phenotype.getI(1)) and float(phenotype.getI(0)) >= float(self.popSet.getI(ref).phenotype.getI(0)):
                    self.correctSet.append(ref)

    def updateSets(self,elcs,exploreIter):
        matchSetNumerosity = 0
        for ref in self.matchSet.getArray():
            matchSetNumerosity += self.popSet.getI(ref).numerosity

        for ref in self.matchSet.getArray():
            self.popSet.getI(ref).updateExperience()
            self.popSet.getI(ref).updateMatchSetSize(elcs,matchSetNumerosity)
            if ref in self.correctSet.getArray():
                self.popSet.getI(ref).updateCorrect()

            self.popSet.getI(ref).updateAccuracy()
            self.popSet.getI(ref).updateFitness(elcs)

    def doCorrectSetSubsumption(self,elcs):
        subsumer = None
        for ref in self.correctSet.getArray():
            cl = self.popSet.getI(ref)
            if cl.isSubsumer(elcs):
                if subsumer == None or cl.isMoreGeneral(subsumer,elcs):
                    subsumer = cl

        if subsumer != None:
            i = 0
            while i < self.correctSet.size():
                ref = self.correctSet.getI(i)
                if subsumer.isMoreGeneral(self.popSet.getI(ref),elcs):
                    if elcs.printGAMech:
                        print("Subsumption Done:")
                        elcs.printClassifier(self.popSet.getI(ref))
                        print("Subsumed by")
                        elcs.printClassifier(subsumer)
                    elcs.subsumptionCounter += 1
                    subsumer.updateNumerosity(self.popSet.getI(ref).numerosity)
                    self.removeMacroClassifier(ref)
                    self.deleteFromMatchSet(ref)
                    self.deleteFromCorrectSet(ref)
                    i -= 1
                i+=1

    def removeMacroClassifier(self,ref):
        self.popSet.removeAtIndex(ref)

    def deleteFromMatchSet(self,deleteRef):
        if deleteRef in self.matchSet.getArray():
            self.matchSet.removeFirstElementWithValue(deleteRef)

        for j in range(self.matchSet.size()):
            ref = self.matchSet.getI(j)
            if ref > deleteRef:
                self.matchSet.setI(j,value=self.matchSet.getI(j)-1)

    def deleteFromCorrectSet(self,deleteRef):
        if deleteRef in self.correctSet.getArray():
            self.correctSet.removeFirstElementWithValue(deleteRef)

        for j in range(self.correctSet.size()):
            ref = self.correctSet.getI(j)
            if ref > deleteRef:
                self.correctSet.setI(j,value=self.correctSet.getI(j)-1)

    def runGA(self,elcs,exploreIter,state,phenotype):
        #GA Run Requirement
        if (exploreIter - self.getIterStampAverage()) < elcs.theta_GA:
            return
        elcs.timer.startTimeSelection()

        elcs.gaCounter += 1
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

        if elcs.printGAMech:
            print("First Chosen Parent:")
            elcs.printClassifier(clP1)
            print("Second Chosen Parent")
            elcs.printClassifier(clP2)

        #Initialize Offspring
        cl1 = Classifier(elcs,clP1,exploreIter)
        if clP2 == None:
            cl2 = Classifier(elcs,clP1, exploreIter)
        else:
            cl2 = Classifier(elcs,clP2, exploreIter)

        #Crossover Operator (uniform crossover)
        if not cl1.equals(elcs,cl2) and random.random() < elcs.chi:
            if elcs.printGAMech:
                print("Crossover Invoked")
            changed = cl1.uniformCrossover(elcs,cl2)
            if elcs.printGAMech:
                elcs.printClassifier(cl1)
                elcs.printClassifier(cl2)

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
        elcs.timer.stopTimeSelection()

        #Add offspring to population
        if changed or nowchanged or howaboutnow:
            if elcs.printMutation or elcs.printCrossOver or elcs.printGAMech:
                if nowchanged:
                    if elcs.printGAMech:
                        print("Mutation Cl1")
                    elcs.mutationCounter += 1
                if howaboutnow:
                    if elcs.printGAMech:
                        print("Mutation Cl2")
                    elcs.mutationCounter += 1
                if changed:
                    if elcs.printGAMech:
                        print("Crossover")
                    elcs.crossOverCounter += 1
            elcs.timer.startTimeSubsumption()
            self.insertDiscoveredClassifiers(elcs,cl1, cl2, clP1, clP2, exploreIter)  # Subsumption
            elcs.timer.stopTimeSubsumption()

    def getIterStampAverage(self):
        sumCl = 0.0
        numSum = 0.0
        for i in range(self.correctSet.size()):
            ref = self.correctSet.getI(i)
            sumCl += self.popSet.getI(ref).timeStampGA * self.popSet.getI(ref).numerosity
            numSum += self.popSet.getI(ref).numerosity
        #print("ITERSTAMP AVG: ",sumCl/float(numSum))
        return sumCl/float(numSum)

    def setIterStamps(self,exploreIter):
        for i in range(self.correctSet.size()):
            ref = self.correctSet.getI(i)
            self.popSet.getI(ref).updateTimeStamp(exploreIter)

    def selectClassifierRW(self):
        setList = copy.deepcopy(self.correctSet)

        if setList.size() > 2:
            selectList = np.array([None,None])
            currentCount = 0

            while currentCount < 2:
                fitSum = self.getFitnessSum(setList)

                choiceP = random.random() * fitSum
                i = 0
                sumCl = self.popSet.getI(setList[i]).fitness
                while choiceP > sumCl:
                    i = i + 1
                    sumCl += self.popSet.getI(setList[i]).fitness

                selectList[currentCount] = self.popSet.getI(setList.getI(i))
                index = np.where(setList.getArray() == setList.getI(i))[0][0]
                setList.removeAtIndex(index)
                currentCount += 1

        elif setList.size() == 2:
            selectList = np.array([self.popSet.getI(setList.getI(0)), self.popSet.getI(setList.getI(1))])
        elif setList.size() == 1:
            selectList = np.array([self.popSet.getI(setList.getI(0)), self.popSet.getI(setList.getI(0))])

        return selectList

    def getFitnessSum(self, setList):
        """ Returns the sum of the fitnesses of all classifiers in the set. """
        sumCl = 0.0
        for i in range(setList.size()):
            ref = setList.getI(i)
            sumCl += self.popSet.getI(ref).fitness
        return sumCl

    def selectClassifierT(self,elcs):
        selectList = np.array([None, None])
        currentCount = 0
        setList = self.correctSet

        while currentCount < 2:
            tSize = int(setList.size() * elcs.theta_sel)

            #Select tSize elements from correctSet
            copyList = copy.deepcopy(self.correctSet)
            posList = ArrayFactory.createArray(k=1,minSize=max(8,tSize))
            for i in range(tSize):
                choice = np.random.choice(copyList.getArray())
                index = np.where(copyList.getArray()==choice)[0][0]
                posList.append(choice)
                copyList.removeAtIndex(index)

            bestF = 0
            bestC = self.correctSet.getI(0)
            for j in posList.getArray():
                if self.popSet.getI(j).fitness > bestF:
                    bestF = self.popSet.getI(j).fitness
                    bestC = j

            selectList[currentCount] = self.popSet.getI(bestC)
            currentCount += 1

        return selectList

    def insertDiscoveredClassifiers(self,elcs,cl1,cl2,clP1,clP2,exploreIter):
        if elcs.doSubsumption:
            elcs.timer.startTimeSubsumption()
            if cl1.specifiedAttList.size() > 0:
                self.subsumeClassifier(elcs,cl1,clP1,clP2)
            if cl2.specifiedAttList.size() > 0:
                self.subsumeClassifier(elcs,cl2, clP1, clP2)
            elcs.timer.stopTimeSubsumption()
        else:
            if cl1.specifiedAttList.size() > 0:
                self.addClassifierToPopulation(elcs,cl1,False)
            if cl2.specifiedAttList.size() > 0:
                self.addClassifierToPopulation(elcs,cl2, False)
            if elcs.printGAMech:
                print("Offspring Classifier:")
                elcs.printClassifier(cl1)
                print("Offspring Classifier:")
                elcs.printClassifier(cl2)

    def subsumeClassifier(self,elcs,cl=None,cl1P=None,cl2P=None):
        if cl1P != None and cl1P.subsumes(elcs,cl):
            self.microPopSize += 1
            cl1P.updateNumerosity(1)
            if elcs.printGAMech:
                print("Parent 1 Subsumes:")
                elcs.printClassifier(cl)
            elcs.subsumptionCounter+=1
        elif cl2P != None and cl2P.subsumes(elcs,cl):
            self.microPopSize += 1
            cl2P.updateNumerosity(1)
            if elcs.printGAMech:
                print("Parent 2 Subsumes:")
                elcs.printClassifier(cl)
            elcs.subsumptionCounter += 1
        else:
            self.subsumeClassifier2(elcs,cl)  # Try to subsume in the correct set.

    def subsumeClassifier2(self,elcs,cl):
        choices = ArrayFactory.createArray(k=1,minSize=self.correctSet.size())
        for ref in self.correctSet.getArray():
            if self.popSet.getI(ref).subsumes(elcs,cl):
                choices.append(ref)

        if choices.size() > 0:
            choice = int(random.random()*choices.size())
            self.popSet.getI(int(choices.getI(choice))).updateNumerosity(1)
            self.microPopSize += 1
            if elcs.printGAMech:
                print("Subsumption Done in Correct Set:")
                elcs.printClassifier(cl)
                print("Subsumed by")
                elcs.printClassifier(self.popSet.getI(int(choices.getI(choice))))
            elcs.subsumptionCounter += 1
            return
        self.addClassifierToPopulation(elcs,cl,False)
        if elcs.printGAMech:
            print("Offspring Classifier:")
            elcs.printClassifier(cl)

    def deletion(self,elcs,exploreIter):
        while (self.microPopSize > elcs.N):
            self.deleteFromPopulation(elcs)

    def deleteFromPopulation(self,elcs):
        meanFitness = self.getPopFitnessSum() / float(self.microPopSize)

        sumCl = 0.0
        voteList = ArrayFactory.createArray(k=1,minSize=elcs.N)
        for cl in self.popSet.getArray():
            vote = cl.getDelProp(elcs,meanFitness)
            sumCl += vote
            voteList.append(vote)

        choicePoint = sumCl * random.random()  # Determine the choice point

        newSum = 0.0
        for i in range(voteList.size()):
            cl = self.popSet.getI(i)
            newSum = newSum + voteList.getI(i)
            if newSum > choicePoint:  # Select classifier for deletion
                # Delete classifier----------------------------------
                cl.updateNumerosity(-1)
                self.microPopSize -= 1
                if cl.numerosity < 1:  # When all micro-classifiers for a given classifier have been depleted.
                    self.removeMacroClassifier(i)
                    self.deleteFromMatchSet(i)
                    self.deleteFromCorrectSet(i)
                return
        return

    def getPopFitnessSum(self):
        """ Returns the sum of the fitnesses of all classifiers in the set. """
        sumCl = 0.0
        for cl in self.popSet.getArray():
            sumCl += cl.fitness * cl.numerosity
        return sumCl

    def clearSets(self,elcs):
        """ Clears out references in the match and correct sets for the next learning iteration. """
        self.matchSet = ArrayFactory.createArray(k=1,minSize=elcs.N)
        self.correctSet = ArrayFactory.createArray(k=1,minSize=elcs.N)

    def runPopAveEval(self,exploreIter,elcs):
        genSum = 0
        agedCount = 0
        for cl in self.popSet.getArray():
            genSum += ((elcs.env.formatData.numAttributes - cl.conditionType.size())/float(elcs.env.formatData.numAttributes))*cl.numerosity
        if self.microPopSize == 0:
            self.aveGenerality = 'NA'
        else:
            self.aveGenerality = genSum/float(self.microPopSize)

        if not elcs.env.formatData.discretePhenotype:
            sumRuleRange = 0
            for cl in self.popSet.getArray():
                sumRuleRange += (cl.phenotype.getI(1)-cl.phenotype.getI(0))*cl.numerosity
            phenotypeRange = elcs.env.formatData.phenotypeList.getI(1)-elcs.env.formatData.phenotypeList.getI(0)
            self.avePhenotypeRange = (sumRuleRange / float(self.microPopSize)) / float(phenotypeRange)

    def runAttGeneralitySum(self,isEvaluationSummary,elcs):
        if isEvaluationSummary:
            self.attributeSpecList = ArrayFactory.createArray(k=1,minSize=elcs.env.formatData.numAttributes)
            self.attributeAccList = ArrayFactory.createArray(k=1,minSize=elcs.env.formatData.numAttributes)
            for i in range(elcs.env.formatData.numAttributes):
                self.attributeSpecList.append(0)
                self.attributeAccList.append(0.0)
            for cl in self.popSet.getArray():
                for ref in cl.specifiedAttList.getArray():
                    self.attributeSpecList.setI(ref,value=self.attributeSpecList.getI(ref)+cl.numerosity)
                    self.attributeAccList.setI(ref,value=self.attributeAccList.getI(ref)+cl.numerosity * cl.accuracy)

    def makeEvalMatchSet(self,state,elcs):
        for i in range(self.popSet.size()):
            cl = self.popSet.getI(i)
            if cl.match(state,elcs):
                self.matchSet.append(i)