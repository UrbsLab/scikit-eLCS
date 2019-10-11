'''
Name: eLCS.py
Authors: Robert Zhang in association with Ryan Urbanowicz
Contact: robertzh@wharton.upenn.edu
Description: This module creates a class that takes in data, and cleans it up to be used by another machine learning module
'''

import numpy as np
import pandas as pd

class StringEnumerator:
    def __init__(self, inputFile, classLabel):
        self.classLabel = classLabel
        self.map = {} #Dictionary of header names: Attribute dictionaries
        data = pd.read_csv(inputFile, sep=',')  # Puts data from csv into indexable np arrays
        data = data.fillna("NA")
        self.dataFeatures = data.drop(classLabel, axis=1).values
        self.dataPhenotypes = data[classLabel].values
        self.dataHeaders = data.drop(classLabel, axis=1).columns.values

    def addAttributeConverter(self,headerName,array):#map is an array of strings, ordered by how it is to be enumerated enumeration
        newAttributeConverter = {}
        for index in range(len(array)):
            newAttributeConverter[array[index]] = index
        self.map[headerName] = newAttributeConverter

    def addClassConverter(self,array):#assumes no other
        newAttributeConverter = {}
        for index in range(len(array)):
            newAttributeConverter[array[index]] = index
        self.map[self.classLabel] = newAttributeConverter

    def convertAllAttributes(self):
        for attribute in self.dataHeaders:
            if attribute in self.map.keys():
                i = np.where(self.dataHeaders == attribute)[0][0]
                for state in self.dataFeatures:#goes through each instance's state
                    if (state[i] in self.map[attribute].keys()):
                        state[i] = self.map[attribute][state[i]]

        for self.classLabel in self.map.keys():
            for state in self.dataPhenotypes:
                if (state in self.map[self.classLabel].keys()):
                    state = self.map[self.classLabel][state]

    def deleteAttribute(self,headerName):
        i = np.where(headerName == self.dataHeaders)[0][0]
        newFeatures = np.array([[2,3]])
        self.dataHeaders = np.delete(self.dataHeaders,i)

        for instanceIndex in range(len(self.dataFeatures)):
            instance = np.delete(self.dataFeatures[instanceIndex],i)
            if (instanceIndex == 0):
                newFeatures = np.array([instance])
            else:
                newFeatures = np.concatenate((newFeatures,[instance]),axis=0)
        self.dataFeatures = newFeatures

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