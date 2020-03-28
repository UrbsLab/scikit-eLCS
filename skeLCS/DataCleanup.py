'''
Name: eLCS.py
Authors: Robert Zhang in association with Ryan Urbanowicz
Contact: robertzh@wharton.upenn.edu
Description: This module creates a class that takes in data, and cleans it up to be used by another machine learning module
'''

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

        self.deleteAllInstancesWithoutPhenotype()

    def printInvalidAttributes(self):
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

    def changeClassName(self,newName):
        if newName in self.dataHeaders:
            raise Exception("New Class Name Cannot Be An Already Existing Data Header Name")
        if self.classLabel in self.map.keys():
            self.map[self.newName] = self.map.pop(self.classLabel)
        self.classLabel = newName

    def changeHeaderName(self,currentName,newName):
        if newName in self.dataHeaders or newName == self.classLabel:
            raise Exception("New Class Name Cannot Be An Already Existing Data Header or Phenotype Name")
        if currentName in self.dataHeaders:
            headerIndex = np.where(self.dataHeaders == currentName)[0][0]
            self.dataHeaders[headerIndex] = newName
            if currentName in self.map.keys():
                self.map[newName] = self.map.pop(currentName)
        else:
            raise Exception("Current Header Doesn't Exist")

    def addAttributeConverter(self,headerName,array):#map is an array of strings, ordered by how it is to be enumerated enumeration
        if headerName in self.dataHeaders and not (headerName in self.map):
            newAttributeConverter = {}
            for index in range(len(array)):
                if str(array[index]) != "NA" and str(array[index]) != "" and str(array[index]) != "NaN":
                    newAttributeConverter[str(array[index])] = str(index)
            self.map[headerName] = newAttributeConverter

    def addAttributeConverterMap(self,headerName,map):
        if headerName in self.dataHeaders and not (headerName in self.map) and not("" in map) and not("NA" in map) and not("NaN" in map):
            self.map[headerName] = map
        else:
            raise Exception("Invalid Map")

    def addAttributeConverterRandom(self,headerName):
        if headerName in self.dataHeaders and not (headerName in self.map):
            headerIndex = np.where(self.dataHeaders == headerName)[0][0]
            uniqueItems = np.array([])
            for instance in self.dataFeatures:
                if not(instance[headerIndex] in uniqueItems) and instance[headerIndex] != "NA":
                    uniqueItems = np.append(uniqueItems,instance[headerIndex])
            self.addAttributeConverter(headerName,uniqueItems)

    def addClassConverter(self,array):
        if not (self.classLabel in self.map.keys()):
            newAttributeConverter = {}
            for index in range(len(array)):
                newAttributeConverter[str(array[index])] = str(index)
            self.map[self.classLabel] = newAttributeConverter

    def addClassConverterRandom(self):
        if not (self.classLabel in self.map.keys()):
            uniqueItems = np.array([])
            for instance in self.dataPhenotypes:
                if not (instance in uniqueItems) and instance != "NA":
                    uniqueItems = np.append(uniqueItems, instance)
            self.addClassConverter(uniqueItems)

    def convertAllAttributes(self):
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

    def deleteAttribute(self,headerName):
        if headerName in self.dataHeaders:
            i = np.where(headerName == self.dataHeaders)[0][0]
            newFeatures = np.array([[2,3]])
            self.dataHeaders = np.delete(self.dataHeaders,i)
            if headerName in self.map.keys():
                del self.map[headerName]

            for instanceIndex in range(len(self.dataFeatures)):
                instance = np.delete(self.dataFeatures[instanceIndex],i)
                if (instanceIndex == 0):
                    newFeatures = np.array([instance])
                else:
                    newFeatures = np.concatenate((newFeatures,[instance]),axis=0)
            self.dataFeatures = newFeatures
        else:
            raise Exception("Header Doesn't Exist")

    def deleteAllInstancesWithoutHeaderData(self,headerName):
        newFeatures = np.array([[2,3]])
        newPhenotypes = np.array([])
        attributeIndex = np.where(self.dataHeaders == headerName)[0][0]

        firstTime = True
        for instanceIndex in range(len(self.dataFeatures)):
            instance = self.dataFeatures[instanceIndex]
            if instance[attributeIndex] != "NA":
                if firstTime:
                    firstTime = False
                    newFeatures = np.array([instance])
                else:
                    newFeatures = np.concatenate((newFeatures,[instance]),axis = 0)
                newPhenotypes = np.append(newPhenotypes,self.dataPhenotypes[instanceIndex])

        self.dataFeatures = newFeatures
        self.dataPhenotypes = newPhenotypes

    def deleteAllInstancesWithoutPhenotype(self):
        newFeatures = np.array([[2,3]])
        newPhenotypes = np.array([])
        firstTime = True
        for instanceIndex in range(len(self.dataFeatures)):
            instance = self.dataPhenotypes[instanceIndex]
            if instance != "NA":
                if firstTime:
                    firstTime = False
                    newFeatures = np.array([self.dataFeatures[instanceIndex]])
                else:
                    newFeatures = np.concatenate((newFeatures,[self.dataFeatures[instanceIndex]]),axis = 0)
                newPhenotypes = np.append(newPhenotypes,instance)

        self.dataFeatures = newFeatures
        self.dataPhenotypes = newPhenotypes

    def print(self):
        isFullNumber = self.checkIsFullNumeric()
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

    def printAttributeConversions(self):
        print("Changed Attribute Conversions")
        for headerName,conversions in self.map:
            print(headerName + " conversions:")
            for original,numberVal in conversions:
                print("\tOriginal: "+original+" Converted: "+numberVal)
            print()
        print()

    def checkIsFullNumeric(self):
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

    def getParams(self):
        if not(self.checkIsFullNumeric()):
            raise Exception("Features and Phenotypes must be fully numeric")

        newFeatures = np.array([[2,3]],dtype=float)
        newPhenotypes = np.array([],dtype=float)
        firstTime = True
        for instanceIndex in range(len(self.dataFeatures)):
            newInstance = np.array([],dtype=float)
            for attribute in self.dataFeatures[instanceIndex]:
                if attribute == "NA":
                    newInstance = np.append(newInstance, np.nan)
                else:
                    newInstance = np.append(newInstance, float(attribute))

            if firstTime:
                firstTime = False
                newFeatures = np.array([newInstance])
            else:
                newFeatures = np.concatenate((newFeatures,[newInstance]),axis = 0)

            if self.dataPhenotypes[instanceIndex] == "NA": #Should never happen. All NaN phenotypes should be removed automatically at init. Just a safety mechanism.
                newPhenotypes = np.append(newPhenotypes, np.nan)
            else:
                newPhenotypes = np.append(newPhenotypes, float(self.dataPhenotypes[instanceIndex]))

        return self.dataHeaders,self.classLabel,newFeatures,newPhenotypes