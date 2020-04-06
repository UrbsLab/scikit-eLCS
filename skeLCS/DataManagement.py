import numpy as np

class DataManagement:
    def __init__(self, dataFeatures, dataPhenotypes, elcs):
        # About Attributes
        self.numAttributes = dataFeatures.shape[1]  # The number of attributes in the input file.
        self.attributeInfoType = [0]*self.numAttributes #stores false (d) or true (c) depending on its type, which points to parallel reference in one of the below 2 arrays
        self.attributeInfoContinuous = [[0,0]]*self.numAttributes #stores continuous ranges and NaN otherwise
        self.attributeInfoDiscrete = [0]*self.numAttributes #stores arrays of discrete values or NaN otherwise.
        for i in range(0,self.numAttributes):
            self.attributeInfoDiscrete[i] = AttributeInfoDiscreteElement()

        # About Phenotypes
        self.discretePhenotype = True  # Is the Class/Phenotype Discrete? (False = Continuous)
        self.phenotypeList = []  # Stores all possible discrete phenotype states/classes or maximum and minimum values for a continuous phenotype
        self.phenotypeRange = None  # Stores the difference between the maximum and minimum values for a continuous phenotype
        self.isDefault = True #Is discrete attribute limit an int or string
        try:
            int(elcs.discreteAttributeLimit)
        except:
            self.isDefault = False

        #About Dataset
        self.numTrainInstances = dataFeatures.shape[0]  # The number of instances in the training data
        self.discriminatePhenotype(dataPhenotypes, elcs)
        if (self.discretePhenotype):
            self.discriminateClasses(dataPhenotypes)
        else:
            self.characterizePhenotype(dataPhenotypes,elcs)

        self.discriminateAttributes(dataFeatures, elcs)
        self.characterizeAttributes(dataFeatures, elcs)
        self.trainFormatted = self.formatData(dataFeatures,dataPhenotypes,elcs) #The only np array

    def discriminatePhenotype(self,phenotypes,elcs):#Determine if phenotype is discrete or continuous
        try:
            int(elcs.discretePhenotypeLimit)
            self.isPhenotypeDefault = True
        except:
            self.isPhenotypeDefault = False

        if (self.isPhenotypeDefault):
            currentPhenotypeIndex = 0
            classDict = {}
            while (self.discretePhenotype and len(list(classDict.keys()))<=elcs.discretePhenotypeLimit and currentPhenotypeIndex < self.numTrainInstances):
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
                self.phenotypeList = [float(target),float(target)]
        elif elcs.discretePhenotypeLimit == "c":
            self.discretePhenotype = False
            self.phenotypeList = [float(phenotypes[0]), float(phenotypes[0])]
        elif elcs.discretePhenotypeLimit == "d":
            self.discretePhenotype = True
            self.phenotypeList = []

    def discriminateClasses(self,phenotypes):
        currentPhenotypeIndex = 0
        classCount = {}
        while (currentPhenotypeIndex < self.numTrainInstances):
            target = phenotypes[currentPhenotypeIndex]
            if target in self.phenotypeList:
                classCount[target]+=1
            else:
                self.phenotypeList.append(target)
                classCount[target] = 1
            currentPhenotypeIndex+=1

    def characterizePhenotype(self,phenotypes,elcs):
        for target in phenotypes:
            if np.isnan(target):
                pass
            elif float(target) > self.phenotypeList[1]:
                self.phenotypeList[1] = float(target)
            elif float(target) < self.phenotypeList[0]:
                self.phenotypeList[0] = float(target)
            else:
                pass
        self.phenotypeRange = self.phenotypeList[1] - self.phenotypeList[0]

    def discriminateAttributes(self,features,elcs):
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
                self.attributeInfoType[att] = False
            else:
                self.attributeInfoType[att] = True

    def characterizeAttributes(self,features,elcs):
        for currentFeatureIndexInAttributeInfo in range(self.numAttributes):
            for currentInstanceIndex in range(self.numTrainInstances):
                target = features[currentInstanceIndex,currentFeatureIndexInAttributeInfo]
                if not self.attributeInfoType[currentFeatureIndexInAttributeInfo]:#if attribute is discrete
                    if target in self.attributeInfoDiscrete[currentFeatureIndexInAttributeInfo].distinctValues or np.isnan(target):
                        pass
                    else:
                        self.attributeInfoDiscrete[currentFeatureIndexInAttributeInfo].distinctValues.append(target)
                else: #if attribute is continuous
                    if np.isnan(target):
                        pass
                    elif float(target) > self.attributeInfoContinuous[currentFeatureIndexInAttributeInfo][1]:
                        self.attributeInfoContinuous[currentFeatureIndexInAttributeInfo][1] = float(target)
                    elif float(target) < self.attributeInfoContinuous[currentFeatureIndexInAttributeInfo][0]:
                        self.attributeInfoContinuous[currentFeatureIndexInAttributeInfo][0] = float(target)
                    else:
                        pass

    def formatData(self,features,phenotypes,elcs):
        formatted = np.insert(features,self.numAttributes,phenotypes,1) #Combines features and phenotypes into one array
        np.random.shuffle(formatted)
        shuffledFeatures = formatted[:,:-1].tolist()
        shuffledLabels = formatted[:,self.numAttributes].tolist()
        return [shuffledFeatures,shuffledLabels]

class AttributeInfoDiscreteElement():
    def __init__(self):
        self.distinctValues = []