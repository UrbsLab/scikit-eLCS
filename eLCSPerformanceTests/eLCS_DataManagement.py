"""
Name:        eLCS_DataManagement.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     November 1, 2013
Description: Able to manage both training and testing data.  This module loads the dataset, detects and characterizes all attributes in the dataset, 
             handles missing data, and finally formats the data so that it may be conveniently utilized by eLCS.
             
---------------------------------------------------------------------------------------------------------------------------------------------------------
eLCS: Educational Learning Classifier System - A basic LCS coded for educational purposes.  This LCS algorithm uses supervised learning, and thus is most 
similar to "UCS", an LCS algorithm published by Ester Bernado-Mansilla and Josep Garrell-Guiu (2003) which in turn is based heavily on "XCS", an LCS 
algorithm published by Stewart Wilson (1995).  

Copyright (C) 2013 Ryan Urbanowicz 
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the 
Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABLILITY 
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, 
Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#Import Required Modules---------------
from eLCS_Constants import * 
import random
#--------------------------------------

class DataManagement:
    def __init__(self, rawTrainData, rawTestData, infoList = None):
        #Initialize global variables-------------------------------------------------
        self.numAttributes = None       # The number of attributes in the input file. 
        self.areInstanceIDs = False     # Does the dataset contain a column of Instance IDs? (If so, it will not be included as an attribute)
        self.instanceIDRef = None       # The column reference for Instance IDs
        self.phenotypeRef = None        # The column reference for the Class/Phenotype column
        self.discretePhenotype = True   # Is the Class/Phenotype Discrete? (False = Continuous)
        self.attributeInfo = []         # Stores Discrete (0) or Continuous (1) for each attribute
        self.phenotypeList = []         # Stores all possible discrete phenotype states/classes or maximum and minimum values for a continuous phenotype
        self.phenotypeRange = None      # Stores the difference between the maximum and minimum values for a continuous phenotype
        
        #Train/Test Specific-----------------------------------------------------------------------------
        self.trainHeaderList = cons.trainHeaderList       # The dataset column headers for the training data
        self.testHeaderList = cons.testHeaderList        # The dataset column headers for the testing data
        self.numTrainInstances = None   # The number of instances in the training data
        self.numTestInstances = None    # The number of instances in the testing data
        
        #Detect Features of training data--------------------------------------------------------------------------
        self.characterizeDataset(rawTrainData,rawTestData)  #Detect number of attributes, instances, and reference locations.
        
        if cons.testFile == 'None': #If no testing data is available, formatting relies solely on training data.
            data4Formating = rawTrainData
        else:
            data4Formating = rawTrainData + rawTestData #Merge Training and Testing datasets

        self.discriminatePhenotype(data4Formating) #Determine if endpoint/phenotype is discrete or continuous.
        if self.discretePhenotype:
            self.discriminateClasses(data4Formating) #Detect number of unique phenotype identifiers.
        else:
            self.characterizePhenotype(data4Formating)
            
        self.discriminateAttributes(data4Formating) #Detect whether attributes are discrete or continuous.
        self.characterizeAttributes(data4Formating) #Determine potential attribute states or ranges.
        
        #Format and Shuffle Datasets----------------------------------------------------------------------------------------
        if cons.testFile != 'None':
            self.testFormatted = self.formatData(rawTestData) #Stores the formatted testing data set used throughout the algorithm.

        self.trainFormatted = self.formatData(rawTrainData) #Stores the formatted training data set used throughout the algorithm.

    def characterizeDataset(self, rawTrainData,rawTestData):
        " Detect basic dataset parameters " 
        #Detect Instance ID's and save location if they occur.  Then save number of attributes in data.
        if cons.labelInstanceID in self.trainHeaderList:
            self.areInstanceIDs = True
            self.instanceIDRef = self.trainHeaderList.index(cons.labelInstanceID)
            print("DataManagement: Instance ID Column location = "+str(self.instanceIDRef))
            self.numAttributes = len(self.trainHeaderList)-2 #one column for InstanceID and another for the phenotype.
        else:
            self.numAttributes = len(self.trainHeaderList)-1
        
        #Identify location of phenotype column
        if cons.labelPhenotype in self.trainHeaderList:
            self.phenotypeRef = self.trainHeaderList.index(cons.labelPhenotype)
        else:
            print("DataManagement: Error - Phenotype column not found!  Check data set to ensure correct phenotype column label, or inclusion in the data.")

        #Adjust training header list to just include attributes labels
        if self.areInstanceIDs:
            if self.phenotypeRef > self.instanceIDRef:
                self.trainHeaderList.pop(self.phenotypeRef)
                self.trainHeaderList.pop(self.instanceIDRef)
                self.testHeaderList.pop(self.phenotypeRef)
                self.testHeaderList.pop(self.instanceIDRef)
            else:
                self.trainHeaderList.pop(self.instanceIDRef)
                self.trainHeaderList.pop(self.phenotypeRef)
                self.testHeaderList.pop(self.instanceIDRef)
                self.testHeaderList.pop(self.phenotypeRef)
        else:
            self.trainHeaderList.pop(self.phenotypeRef)
            self.testHeaderList.pop(self.phenotypeRef)


        
        #Store number of instances in training data
        self.numTrainInstances = len(rawTrainData)
        self.numTestInstances = len(rawTestData)

    def discriminatePhenotype(self, rawData):
        """ Determine whether the phenotype is Discrete(class-based) or Continuous """
        inst = 0
        classDict = {}
        while self.discretePhenotype and len(list(classDict.keys())) <= cons.discreteAttributeLimit and inst < self.numTrainInstances:  #Checks which discriminate between discrete and continuous attribute
            target = rawData[inst][self.phenotypeRef]
            if target in list(classDict.keys()):  #Check if we've seen this attribute state yet.
                classDict[target] += 1
            elif target == cons.labelMissingData: #Ignore missing data
                pass
            else: #New state observed
                classDict[target] = 1
            inst += 1

        if len(list(classDict.keys())) > cons.discreteAttributeLimit:
            self.discretePhenotype = False
            self.phenotypeList = [float(target),float(target)]

    def discriminateClasses(self, rawData):
        """ Determines number of classes and their identifiers. Only used if phenotype is discrete. """
        inst = 0
        classCount = {}
        while inst < self.numTrainInstances:
            target = rawData[inst][self.phenotypeRef]
            if target in self.phenotypeList:
                classCount[target] += 1 
            else:
                self.phenotypeList.append(target)
                classCount[target] = 1
            inst += 1

    def discriminateAttributes(self, rawData):
        """ Determine whether attributes in dataset are discrete or continuous and saves this information. """
        self.discreteCount = 0
        self.continuousCount = 0
        for att in range(len(rawData[0])):
            if att != self.instanceIDRef and att != self.phenotypeRef:  #Get just the attribute columns (ignores phenotype and instanceID columns)
                attIsDiscrete = True
                inst = 0
                stateDict = {}
                while attIsDiscrete and len(list(stateDict.keys())) <= cons.discreteAttributeLimit and inst < self.numTrainInstances:  #Checks which discriminate between discrete and continuous attribute
                    target = rawData[inst][att]
                    if target in list(stateDict.keys()):  #Check if we've seen this attribute state yet.
                        stateDict[target] += 1
                    elif target == cons.labelMissingData: #Ignore missing data
                        pass
                    else: #New state observed
                        stateDict[target] = 1
                    inst += 1

                if len(list(stateDict.keys())) > cons.discreteAttributeLimit:
                    attIsDiscrete = False
                if attIsDiscrete:
                    self.attributeInfo.append([0,[]])    
                    self.discreteCount += 1
                else:
                    self.attributeInfo.append([1,[float(target),float(target)]])   #[min,max]
                    self.continuousCount += 1
            
    def characterizeAttributes(self, rawData):
        """ Determine range (if continuous) or states (if discrete) for each attribute and saves this information"""
        attributeID = 0
        for att in range(len(rawData[0])):
            if att != self.instanceIDRef and att != self.phenotypeRef:  #Get just the attribute columns (ignores phenotype and instanceID columns)
                for inst in range(len(rawData)):
                    target = rawData[inst][att]
                    if not self.attributeInfo[attributeID][0]: #If attribute is discrete
                        if target in self.attributeInfo[attributeID][1] or target == cons.labelMissingData:
                            pass  #NOTE: Could potentially store state frequency information to guide learning.
                        else:
                            self.attributeInfo[attributeID][1].append(target)
                    else: #If attribute is continuous
                        
                        #Find Minimum and Maximum values for the continuous attribute so we know the range.
                        if target == cons.labelMissingData:
                            pass
                        elif float(target) > self.attributeInfo[attributeID][1][1]:  #error
                            self.attributeInfo[attributeID][1][1] = float(target)
                        elif float(target) < self.attributeInfo[attributeID][1][0]:
                            self.attributeInfo[attributeID][1][0] = float(target)
                        else:
                            pass
                attributeID += 1
                

    def characterizePhenotype(self, rawData):
        """ Determine range of phenotype values. """
        for inst in range(len(rawData)):
            target = rawData[inst][self.phenotypeRef]
            
            #Find Minimum and Maximum values for the continuous phenotype so we know the range.
            if target == cons.labelMissingData:
                pass
            elif float(target) > self.phenotypeList[1]:  
                self.phenotypeList[1] = float(target)
            elif float(target) < self.phenotypeList[0]:
                self.phenotypeList[0] = float(target)
            else:
                pass
        self.phenotypeRange = self.phenotypeList[1] - self.phenotypeList[0]
                
            
    def formatData(self,rawData):
        """ Get the data into a format convenient for the algorithm to interact with. Specifically each instance is stored in a list as follows; [Attribute States, Phenotype, InstanceID] """
        formatted = []
        #Initialize data format---------------------------------------------------------
        for i in range(len(rawData)):  
            formatted.append([None,None,None]) #[Attribute States, Phenotype, InstanceID]

        for inst in range(len(rawData)):
            stateList = []
            attributeID = 0
            for att in range(len(rawData[0])):
                if att != self.instanceIDRef and att != self.phenotypeRef:  #Get just the attribute columns (ignores phenotype and instanceID columns)
                    target = rawData[inst][att]
                    
                    if self.attributeInfo[attributeID][0]: #If the attribute is continuous
                        if target == cons.labelMissingData:
                            stateList.append(target) #Missing data saved as text label
                        else:
                            stateList.append(float(target)) #Save continuous data as floats. 
                    else: #If the attribute is discrete - Format the data to correspond to the GABIL (DeJong 1991)
                        stateList.append(target) #missing data, and discrete variables, all stored as string objects   
                    attributeID += 1
            
            #Final Format-----------------------------------------------
            formatted[inst][0] = stateList                           #Attribute states stored here
            if self.discretePhenotype:
                formatted[inst][1] = rawData[inst][self.phenotypeRef]        #phenotype stored here
            else:
                formatted[inst][1] = float(rawData[inst][self.phenotypeRef])
            if self.areInstanceIDs:
                formatted[inst][2] = rawData[inst][self.instanceIDRef]   #Instance ID stored here
            else:
                pass    #instance ID neither given nor required.
            #-----------------------------------------------------------
        #random.shuffle(formatted) #One time randomization of the order the of the instances in the data, so that if the data was ordered by phenotype, this potential learning bias (based on instance ordering) is eliminated.
        return formatted
    