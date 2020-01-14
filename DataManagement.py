from eLCS import *
import numpy as np
from DynamicNPArray import ArrayFactory

class DataManagement:
    def __init__(self, dataFeatures, dataPhenotypes, elcs):
        # About Attributes
        self.numAttributes = dataFeatures.shape[1]  # The number of attributes in the input file.
        self.attributeInfoType = ArrayFactory.createArray(np.empty(self.numAttributes,dtype=bool)) #stores false (d) or true (c) depending on its type, which points to parallel reference in one of the below 2 arrays
        self.attributeInfoContinuous = ArrayFactory.createArray(np.empty((self.numAttributes,2))) #stores continuous ranges and NaN otherwise
        self.attributeInfoDiscrete = ArrayFactory.createArray(np.empty(self.numAttributes,dtype=object)) #stores arrays of discrete values or NaN otherwise.
        for i in range(0,self.numAttributes):
            self.attributeInfoDiscrete.setI(i,value=AttributeInfoDiscreteElement())

        # About Phenotypes
        self.discretePhenotype = True  # Is the Class/Phenotype Discrete? (False = Continuous)
        self.phenotypeList = ArrayFactory.createArray(k=1)  # Stores all possible discrete phenotype states/classes or maximum and minimum values for a continuous phenotype
        self.phenotypeRange = None  # Stores the difference between the maximum and minimum values for a continuous phenotype
        self.isDefault = True #Is discrete attribute limit an int or string

        #About Dataset
        self.numTrainInstances = dataFeatures.shape[0]  # The number of instances in the training data
        self.discriminatePhenotype(dataPhenotypes, elcs)
        if (self.discretePhenotype):
            self.discriminateClasses(dataPhenotypes)
        else:
            self.characterizePhenotype(dataPhenotypes,elcs)

        self.discriminateAttributes(dataFeatures, elcs)
        self.characterizeAttributes(dataFeatures, elcs)
        self.trainFormatted = self.formatData(dataFeatures,dataPhenotypes,elcs)

    def discriminatePhenotype(self,phenotypes,elcs):#Determine if phenotype is discrete or continuous
        try:
            int(elcs.discretePhenotypeLimit)
        except:
            self.isDefault = False

        if (self.isDefault):
            currentPhenotypeIndex = 0
            classDict = {}
            while (self.discretePhenotype and len(list(classDict.keys()))<=elcs.discreteAttributeLimit and currentPhenotypeIndex < self.numTrainInstances):
                target = phenotypes[currentPhenotypeIndex]
                if (target in list(classDict.keys())):
                    classDict[target]+=1
                elif np.isnan(target):
                    pass
                else:
                    classDict[target] = 1
                currentPhenotypeIndex+=1

            if (len(list(classDict.keys())) > elcs.discretePhenotypeLimit):
                self.discretePhenotype = False
                self.phenotypeList.append(float(target))
                self.phenotypeList.append(float(target))
        elif elcs.discretePhenotypeLimit == "c":
            self.discretePhenotype = False
            self.phenotypeList.append(float(phenotypes[0]))
            self.phenotypeList.append(float(phenotypes[0]))
        elif elcs.discretePhenotypeLimit == "d":
            self.discretePhenotype = True
            self.phenotypeList = ArrayFactory.createArray(k=1)

    def discriminateClasses(self,phenotypes):
        currentPhenotypeIndex = 0
        classCount = {}
        while (currentPhenotypeIndex < self.numTrainInstances):
            target = phenotypes[currentPhenotypeIndex]
            if (target in self.phenotypeList.getArray()):
                classCount[target]+=1
            else:
                self.phenotypeList.append(target)
                classCount[target] = 1
            currentPhenotypeIndex+=1

    def characterizePhenotype(self,phenotypes,elcs):
        for target in phenotypes:
            if np.isnan(target):
                pass
            elif float(target) > self.phenotypeList.getI(1):
                self.phenotypeList.setI(1,0,value=float(target))
            elif float(target) < self.phenotypeList.getI(0):
                self.phenotypeList.setI(1,0,value=float(target))
            else:
                pass
        self.phenotypeRange = self.phenotypeList.getI(1) - self.phenotypeList.getI(0)

    def discriminateAttributes(self,features,elcs):
        self.discreteCount = 0
        self.continuousCount = 0
        for att in range(self.numAttributes):
            attIsDiscrete = True
            if self.isDefault:
                currentInstanceIndex = 0
                stateDict = {}
                while attIsDiscrete and len(list(stateDict.keys())) <= elcs.discreteAttributeLimit and currentInstanceIndex < self.numTrainInstances:
                    target = features[currentInstanceIndex,att]
                    if target in list(stateDict.keys()):
                        stateDict[target] += 1
                    elif np.isnan(target):
                        pass
                    else:
                        stateDict[target] = 1
                    currentInstanceIndex+=1

                if len(list(stateDict.keys())) > elcs.discreteAttributeLimit:
                    attIsDiscrete = False
            elif elcs.discreteAttributeLimit == "c":
                if att in elcs.specifiedAttributes:
                    attIsDiscrete = False
                else:
                    attIsDiscrete = True
            elif elcs.discreteAttributeLimit == "d":
                if att in elcs.specifiedAttributes:
                    attIsDiscrete = True
                else:
                    attIsDiscrete = False

            if attIsDiscrete:
                self.attributeInfoType.setI(att,False)
                self.discreteCount += 1
            else:
                self.attributeInfoType.setI(att,True)
                self.continuousCount += 1

    def characterizeAttributes(self,features,elcs):
        for currentFeatureIndexInAttributeInfo in range(self.numAttributes):
            for currentInstanceIndex in range(self.numTrainInstances):
                target = features[currentInstanceIndex,currentFeatureIndexInAttributeInfo]
                if not self.attributeInfoType.getI(currentFeatureIndexInAttributeInfo):#if attribute is discrete
                    if target in self.attributeInfoDiscrete.getI(currentFeatureIndexInAttributeInfo).distinctValues.getArray() or np.isnan(target):
                        pass
                    else:
                        self.attributeInfoDiscrete.getI(currentFeatureIndexInAttributeInfo).distinctValues.append(target)
                else: #if attribute is continuous
                    if np.isnan(target):
                        pass
                    elif float(target) > self.attributeInfoContinuous.getI(currentFeatureIndexInAttributeInfo,1):
                        self.attributeInfoContinuous.setI(currentFeatureIndexInAttributeInfo,1,float(target))
                    elif float(target) < self.attributeInfoContinuous.getI(currentFeatureIndexInAttributeInfo,0):
                        self.attributeInfoContinuous.setI(currentFeatureIndexInAttributeInfo, 0, float(target))
                    else:
                        pass

    def formatData(self,features,phenotypes,elcs):
        formatted = np.insert(features,self.numAttributes,phenotypes,1) #Combines features and phenotypes into one array
        np.random.shuffle(formatted)
        return ArrayFactory.createArray(formatted)

class AttributeInfoDiscreteElement():
    def __init__(self):
        self.distinctValues = ArrayFactory.createArray(k=1)