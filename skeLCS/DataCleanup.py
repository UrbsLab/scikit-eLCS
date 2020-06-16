

import numpy as np
import pandas as pd
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

class StringEnumerator:
    def __init__(self, inputFile, classLabel):
        self.classLabel = classLabel
        self.map = {} #Dictionary of header names: Attribute dictionaries
        data = pd.read_csv(inputFile, sep=',')  # Puts data from csv into indexable np arrays
        data = data.fillna("NA")
        self.dataFeatures = data.drop(classLabel, axis=1).values #splits into an array of instances
        self.dataPhenotypes = data[classLabel].values
        self.dataHeaders = data.drop(classLabel, axis=1).columns.values

        tempPhenoArray = np.empty(len(self.dataPhenotypes),dtype=object)
        for instanceIndex in range(len(self.dataPhenotypes)):
            tempPhenoArray[instanceIndex] = str(self.dataPhenotypes[instanceIndex])
        self.dataPhenotypes = tempPhenoArray

        tempFeatureArray = np.empty((len(self.dataPhenotypes),len(self.dataHeaders)),dtype=object)
        for instanceIndex in range(len(self.dataFeatures)):
            for attrInst in range(len(self.dataHeaders)):
                tempFeatureArray[instanceIndex][attrInst] = str(self.dataFeatures[instanceIndex][attrInst])
        self.dataFeatures = tempFeatureArray

        self.delete_all_instances_without_phenotype()

    def print_invalid_attributes(self):
        print("ALL INVALID ATTRIBUTES & THEIR DISTINCT VALUES")
        for attr in range(len(self.dataHeaders)):
            distinctValues = []
            isInvalid = False
            for instIndex in range(len(self.dataFeatures)):
                val = self.dataFeatures[instIndex,attr]
                if not val in distinctValues and val != "NA":
                    distinctValues.append(self.dataFeatures[instIndex,attr])
                if val != "NA":
                    try:
                        float(val)
                    except:
                        isInvalid = True
            if isInvalid:
                print(str(self.dataHeaders[attr])+": ",end="")
                for i in distinctValues:
                    print(str(i)+"\t",end="")
                print()

        distinctValues = []
        isInvalid = False
        for instIndex in range(len(self.dataPhenotypes)):
            val = self.dataPhenotypes[instIndex]
            if not val in distinctValues and val != "NA":
                distinctValues.append(self.dataPhenotypes[instIndex])
            if val != "NA":
                try:
                    float(val)
                except:
                    isInvalid = True
        if isInvalid:
            print(str(self.classLabel)+" (the phenotype): ",end="")
            for i in distinctValues:
                print(str(i)+"\t",end="")
            print()

    def change_class_name(self,newName):
        if newName in self.dataHeaders:
            raise Exception("New Class Name Cannot Be An Already Existing Data Header Name")
        if self.classLabel in self.map.keys():
            self.map[self.newName] = self.map.pop(self.classLabel)
        self.classLabel = newName

    def change_header_name(self,currentName,newName):
        if newName in self.dataHeaders or newName == self.classLabel:
            raise Exception("New Class Name Cannot Be An Already Existing Data Header or Phenotype Name")
        if currentName in self.dataHeaders:
            headerIndex = np.where(self.dataHeaders == currentName)[0][0]
            self.dataHeaders[headerIndex] = newName
            if currentName in self.map.keys():
                self.map[newName] = self.map.pop(currentName)
        else:
            raise Exception("Current Header Doesn't Exist")

    def add_attribute_converter(self,headerName,array):#map is an array of strings, ordered by how it is to be enumerated enumeration
        if headerName in self.dataHeaders and not (headerName in self.map):
            newAttributeConverter = {}
            for index in range(len(array)):
                if str(array[index]) != "NA" and str(array[index]) != "" and str(array[index]) != "NaN":
                    newAttributeConverter[str(array[index])] = str(index)
            self.map[headerName] = newAttributeConverter

    def add_attribute_converter_map(self,headerName,map):
        if headerName in self.dataHeaders and not (headerName in self.map) and not("" in map) and not("NA" in map) and not("NaN" in map):
            self.map[headerName] = map
        else:
            raise Exception("Invalid Map")

    def add_attribute_converter_random(self,headerName):
        if headerName in self.dataHeaders and not (headerName in self.map):
            headerIndex = np.where(self.dataHeaders == headerName)[0][0]
            uniqueItems = []
            for instance in self.dataFeatures:
                if not(instance[headerIndex] in uniqueItems) and instance[headerIndex] != "NA":
                    uniqueItems.append(instance[headerIndex])
            self.add_attribute_converter(headerName,np.array(uniqueItems))

    def add_class_converter(self,array):
        if not (self.classLabel in self.map.keys()):
            newAttributeConverter = {}
            for index in range(len(array)):
                newAttributeConverter[str(array[index])] = str(index)
            self.map[self.classLabel] = newAttributeConverter

    def add_class_converter_random(self):
        if not (self.classLabel in self.map.keys()):
            uniqueItems = []
            for instance in self.dataPhenotypes:
                if not (instance in uniqueItems) and instance != "NA":
                    uniqueItems.append(instance)
            self.add_class_converter(np.array(uniqueItems))

    def convert_all_attributes(self):
        for attribute in self.dataHeaders:
            if attribute in self.map.keys():
                i = np.where(self.dataHeaders == attribute)[0][0]
                for state in self.dataFeatures:#goes through each instance's state
                    if (state[i] in self.map[attribute].keys()):
                        state[i] = self.map[attribute][state[i]]

        if self.classLabel in self.map.keys():
            for state in self.dataPhenotypes:
                if (state in self.map[self.classLabel].keys()):
                    i = np.where(self.dataPhenotypes == state)
                    self.dataPhenotypes[i] = self.map[self.classLabel][state]

    def delete_attribute(self,headerName):
        if headerName in self.dataHeaders:
            i = np.where(headerName == self.dataHeaders)[0][0]
            self.dataHeaders = np.delete(self.dataHeaders,i)
            if headerName in self.map.keys():
                del self.map[headerName]

            newFeatures = []
            for instanceIndex in range(len(self.dataFeatures)):
                instance = np.delete(self.dataFeatures[instanceIndex],i)
                newFeatures.append(instance)
            self.dataFeatures = np.array(newFeatures)
        else:
            raise Exception("Header Doesn't Exist")

    def delete_all_instances_without_header_data(self,headerName):
        newFeatures = []
        newPhenotypes = []
        attributeIndex = np.where(self.dataHeaders == headerName)[0][0]

        for instanceIndex in range(len(self.dataFeatures)):
            instance = self.dataFeatures[instanceIndex]
            if instance[attributeIndex] != "NA":
                newFeatures.append(instance)
                newPhenotypes.append(self.dataPhenotypes[instanceIndex])

        self.dataFeatures = np.array(newFeatures)
        self.dataPhenotypes = np.array(newPhenotypes)

    def delete_all_instances_without_phenotype(self):
        newFeatures = []
        newPhenotypes = []
        for instanceIndex in range(len(self.dataFeatures)):
            instance = self.dataPhenotypes[instanceIndex]
            if instance != "NA":
                newFeatures.append(self.dataFeatures[instanceIndex])
                newPhenotypes.append(instance)

        self.dataFeatures = np.array(newFeatures)
        self.dataPhenotypes = np.array(newPhenotypes)

    def print(self):
        isFullNumber = self.check_is_full_numeric()
        print("Converted Data Features and Phenotypes")
        for header in self.dataHeaders:
            print(header,end="\t")
        print()
        for instanceIndex in range(len(self.dataFeatures)):
            for attribute in self.dataFeatures[instanceIndex]:
                if attribute != "NA":
                    if (isFullNumber):
                        print(float(attribute), end="\t")
                    else:
                        print(attribute, end="\t\t")
                else:
                    print("NA", end = "\t")
            if self.dataPhenotypes[instanceIndex] != "NA":
                if (isFullNumber):
                    print(float(self.dataPhenotypes[instanceIndex]))
                else:
                    print(self.dataPhenotypes[instanceIndex])
            else:
                print("NA")
        print()

    def print_attribute_conversions(self):
        print("Changed Attribute Conversions")
        for headerName,conversions in self.map:
            print(headerName + " conversions:")
            for original,numberVal in conversions:
                print("\tOriginal: "+original+" Converted: "+numberVal)
            print()
        print()

    def check_is_full_numeric(self):
        try:
            for instance in self.dataFeatures:
                for value in instance:
                    if value != "NA":
                        float(value)
            for value in self.dataPhenotypes:
                if value != "NA":
                    float(value)

        except:
            return False

        return True

    def get_params(self):
        if not(self.check_is_full_numeric()):
            raise Exception("Features and Phenotypes must be fully numeric")

        newFeatures = []
        newPhenotypes = []
        for instanceIndex in range(len(self.dataFeatures)):
            newInstance = []
            for attribute in self.dataFeatures[instanceIndex]:
                if attribute == "NA":
                    newInstance.append(np.nan)
                else:
                    newInstance.append(float(attribute))

            newFeatures.append(np.array(newInstance,dtype=float))
            if self.dataPhenotypes[instanceIndex] == "NA": #Should never happen. All NaN phenotypes should be removed automatically at init. Just a safety mechanism.
                newPhenotypes.append(np.nan)
            else:
                newPhenotypes.append(float(self.dataPhenotypes[instanceIndex]))

        return self.dataHeaders,self.classLabel,np.array(newFeatures,dtype=float),np.array(newPhenotypes,dtype=float)