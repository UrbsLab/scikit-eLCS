import random
import sys
import numpy as np
from eLCS import *

class DataManagement:
    def __init__(self,dataFeatures,dataPhenotypes,elcs):
        #About Attributes
        self.numAttributes = dataFeatures.shape[1]  # The number of attributes in the input file.
        self.attributeInfo = np.array([])  # Stores AttributeInfo type element for each attribute with continuous/discrete

        #About Phenotypes
        self.discretePhenotype = True  # Is the Class/Phenotype Discrete? (False = Continuous)
        self.phenotypeList = np.array([])  # Stores all possible discrete phenotype states/classes or maximum and minimum values for a continuous phenotype
        self.phenotypeRange = None  # Stores the difference between the maximum and minimum values for a continuous phenotype

        #About Dataset
        self.numTrainInstances = dataFeatures.shape[0]  # The number of instances in the training data
        self.discriminatePhenotype(dataPhenotypes,elcs)
        if (self.discretePhenotype == True):
            self.discriminateClasses(dataPhenotypes)
        else:
            self.characterizePhenotype(dataPhenotypes,elcs)

        self.discriminateAttributes(dataFeatures,elcs)
        self.characterizeAttributes(dataFeatures,elcs)

        self.trainFormatted = self.formatData(dataFeatures,dataPhenotypes,elcs)

    def discriminatePhenotype(self,phenotypes,elcs):#Determine if phenotype is discrete or continuous
        if (elcs.explicitPhenotype == "n"):
            currentPhenotypeIndex = 0
            classDict = {}
            while (self.discretePhenotype and len(list(classDict.keys()))<=elcs.discreteAttributeLimit and currentPhenotypeIndex < self.numTrainInstances):
                target = phenotypes[currentPhenotypeIndex]
                if (target in list(classDict.keys())):
                    classDict[target]+=1
                elif (target == elcs.labelMissingData):
                    pass
                else:
                    classDict[target] = 1
                currentPhenotypeIndex+=1

            if (len(list(classDict.keys())) > elcs.discreteAttributeLimit):
                self.discretePhenotype = False
                self.phenotypeList = np.array([float(target),float(target)])
        elif elcs.explicitPhenotype == "c":
            self.discretePhenotype = False
            self.phenotypeList = np.array([float(phenotypes[0]),float(phenotypes[0])])

    def discriminateClasses(self,phenotypes):
        currentPhenotypeIndex = 0
        classCount = {}
        while (currentPhenotypeIndex < self.numTrainInstances):
            target = phenotypes[currentPhenotypeIndex]
            if (target in self.phenotypeList):
                classCount[target]+=1
            else:
                self.phenotypeList = np.append(self.phenotypeList,target)
                classCount[target] = 1
            currentPhenotypeIndex+=1

    def characterizePhenotype(self,phenotypes,elcs):
        for target in phenotypes:
            if target == elcs.labelMissingData:
                pass
            elif float(target) > self.phenotypeList[1]:
                self.phenotypeList[1] = float(target)
            elif float(target) < self.phenotypeList[0]:
                self.phenotypeList[0] = float(target)
            else:
                pass
        self.phenotypeRange = self.phenotypeList[1] - self.phenotypeList[0]

    def discriminateAttributes(self,features,elcs):
        self.discreteCount = 0
        self.continuousCount = 0
        for att in range(self.numAttributes):
            attIsDiscrete = True
            currentInstanceIndex = 0
            stateDict = {}
            while attIsDiscrete and len(list(stateDict.keys())) <= elcs.discreteAttributeLimit and currentInstanceIndex < self.numTrainInstances:
                target = features[currentInstanceIndex,att]
                if target in list(stateDict.keys()):
                    stateDict[target] += 1
                elif target == elcs.labelMissingData:
                    pass
                else:
                    stateDict[target] = 1
                currentInstanceIndex+=1

            if len(list(stateDict.keys())) > elcs.discreteAttributeLimit:
                attIsDiscrete = False

            if att in elcs.explicitlyDiscreteAttributeIndexes:
                attIsDiscrete = True
            if att in elcs.explicitlyContinuousAttributeIndexes:
                attIsDiscrete = False

            if attIsDiscrete:
                self.attributeInfo = np.append(self.attributeInfo,AttributeInfoElement('discrete'))
                self.discreteCount += 1
            else:
                self.attributeInfo = np.append(self.attributeInfo,AttributeInfoElement('continuous',target))
                self.continuousCount += 1

    def characterizeAttributes(self,features,elcs):
        for currentFeatureIndexInAttributeInfo in range(self.numAttributes):
            for currentInstanceIndex in range(self.numTrainInstances):
                target = features[currentInstanceIndex,currentFeatureIndexInAttributeInfo]
                if not self.attributeInfo[currentFeatureIndexInAttributeInfo].type:#if attribute is discrete
                    if target in self.attributeInfo[currentFeatureIndexInAttributeInfo].info or target == elcs.labelMissingData:
                        pass
                    else:
                        self.attributeInfo[currentFeatureIndexInAttributeInfo].info = np.append(self.attributeInfo[currentFeatureIndexInAttributeInfo].info,target)
                else: #if attribute is continuous
                    if target == elcs.labelMissingData:
                        pass
                    elif float(target) > self.attributeInfo[currentFeatureIndexInAttributeInfo].info[1]:
                        self.attributeInfo[currentFeatureIndexInAttributeInfo].info[1] = float(target)
                    elif float(target) < self.attributeInfo[currentFeatureIndexInAttributeInfo].info[0]:
                        self.attributeInfo[currentFeatureIndexInAttributeInfo].info[0] = float(target)
                    else:
                        pass

    def formatData(self,features,phenotypes,elcs):
        formatted = np.array([])
        for i in range(self.numTrainInstances):
            formatted = np.append(formatted,DataInstance())

        for instance in range(self.numTrainInstances):
            for attribute in range(self.numAttributes):
                target = features[instance][attribute]

                if self.attributeInfo[attribute].type:#If attribute is continuous
                    if target == elcs.labelMissingData:
                        formatted[instance].attributeList = np.append(formatted[instance].attributeList,MultiStateValue(target))
                    else:
                        formatted[instance].attributeList = np.append(formatted[instance].attributeList,MultiStateValue(float(target)))
                else:
                    formatted[instance].attributeList = np.append(formatted[instance].attributeList,MultiStateValue(target))

            if self.discretePhenotype:
                formatted[instance].phenotype = phenotypes[instance]
            else:
                formatted[instance].phenotype = float(phenotypes[instance])

        np.random.shuffle(formatted) #Disable shuffling for now
        return formatted


class AttributeInfoElement():
    def __init__(self,type,target=None):
        if (type == 'discrete'):#is Discrete
            self.type = 0
            self.info = np.array([])
        elif (type == 'continuous'):#is Continuous
            self.type = 1
            self.info = np.array([float(target),float(target)])

class MultiStateValue():#Allows floats and strings to be stored within the same numpy array
    def __init__(self,value = 0):
        self.value = value

class DataInstance():
    def __init__(self,attributeList = np.array([]),phenotype = 0):
        self.attributeList = attributeList#np array of multistate values
        self.phenotype = phenotype
