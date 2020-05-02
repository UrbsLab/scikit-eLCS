"""
Name:        eLCS_Constants.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     November 1, 2013
Description: Stores and manages all algorithm run parameters, making them accessible anywhere in the rest of the algorithm code by (cons.) .
             
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
import os
import pandas as pd
import numpy as np
import random
import copy

class Constants:
    def setConstants(self,par):
        """ Takes the parameters parsed as a dictionary from eLCS_ConfigParser and saves them as global constants. """
        
        # Major Run Parameters -----------------------------------------------------------------------------------------
        self.cv = par['cv']                                                                         #If False, no cv, if int, do that many cvs
        self.dataFile = par['dataFile']
        if self.cv == False:
            self.testFile = "None"
        else:
            self.testFile = "Existant"

        if str(par['randomSeed']) == 'False':
            self.useSeed = False                                                #Saved as Boolean
        else:
            self.useSeed = True                                                 #Saved as Boolean
            self.randomSeed = int(par['randomSeed'])                            #Saved as integer

        # Set random seed if specified.-----------------------------------------------
        if self.useSeed:
            random.seed(self.randomSeed)
        else:
            random.seed(None)

        d = pd.read_csv(self.dataFile)
        data = d.values.tolist()
        random.shuffle(data)
        self.trainHeaderList1 = d.columns.values.tolist()
        self.testHeaderList1 = d.columns.values.tolist()
        self.trainHeaderList = copy.deepcopy(self.trainHeaderList1)
        self.testHeaderList = copy.deepcopy(self.testHeaderList1)

        if self.cv == False:
            self.train = data
            self.test = []
        else:
            self.cvCount = 0
            self.split = np.array_split(data, self.cv)
            for i in range(len(self.split)):
                self.split[i] = self.split[i].tolist()

        self.learningIterations = par['learningIterations']                     #Saved as text
        self.N = int(par['N'])                                                  #Saved as integer
        self.p_spec = float(par['p_spec'])                                      #Saved as float
        
        # Logistical Run Parameters ------------------------------------------------------------------------------------
        self.labelInstanceID = par['labelInstanceID']                           #Saved as text
        self.labelPhenotype = par['labelPhenotype']                             #Saved as text
        self.labelMissingData = par['labelMissingData']                         #Saved as text
        self.discreteAttributeLimit = int(par['discreteAttributeLimit'])        #Saved as integer
        self.trackingFrequency = int(par['trackingFrequency'])                  #Saved as integer
        
        # Supervised Learning Parameters -------------------------------------------------------------------------------
        self.nu = int(par['nu'])                                                #Saved as integer
        self.chi = float(par['chi'])                                            #Saved as float
        self.upsilon = float(par['upsilon'])                                    #Saved as float
        self.theta_GA = int(par['theta_GA'])                                    #Saved as integer
        self.theta_del = int(par['theta_del'])                                  #Saved as integer
        self.theta_sub = int(par['theta_sub'])                                  #Saved as integer
        self.acc_sub = float(par['acc_sub'])                                    #Saved as float
        self.beta = float(par['beta'])                                          #Saved as float
        self.delta = float(par['delta'])                                        #Saved as float
        self.init_fit = float(par['init_fit'])                                  #Saved as float
        self.fitnessReduction = float(par['fitnessReduction'])                  #Saved as float
        
        # Algorithm Heuristic Options -------------------------------------------------------------------------------
        self.doSubsumption = par['doSubsumption']                  #Saved as Boolean
        self.selectionMethod = par['selectionMethod']                           #Saved as text
        self.theta_sel = float(par['theta_sel'])                                #Saved as float
        
        # PopulationReboot -------------------------------------------------------------------------------
        self.doPopulationReboot = par['doPopulationReboot']          #Saved as Boolean
        self.popRebootPath = par['popRebootPath']                               #Saved as text

    def setCV(self):
        if self.cv == False:
            pass
        else:
            self.trainHeaderList = copy.deepcopy(self.trainHeaderList1)
            self.testHeaderList = copy.deepcopy(self.testHeaderList1)
            train = copy.deepcopy(self.split)
            train.pop(self.cvCount)
            t = []
            for i in range(len(self.split) - 1):
                t += train[i]
            self.train = t
            self.test = self.split[self.cvCount]
            self.cvCount+=1
        
    def referenceTimer(self, timer):
        """ Store reference to the timer object. """
        self.timer = timer
        
        
    def referenceEnv(self, e):
        """ Store reference to environment object. """
        self.env = e

    def parseIterations(self):
        """ Parse the 'learningIterations' string to identify the maximum number of learning iterations as well as evaluation checkpoints. """
        checkpoints = self.learningIterations.split('.') 
        for i in range(len(checkpoints)): 
            checkpoints[i] = int(checkpoints[i])
            
        self.learningCheckpoints = checkpoints
        self.maxLearningIterations = self.learningCheckpoints[(len(self.learningCheckpoints)-1)] 
        
        if self.trackingFrequency == 0:
            self.trackingFrequency = self.env.formatData.numTrainInstances  #Adjust tracking frequency to match the training data size - learning tracking occurs once every epoch

#To access one of the above constant values from another module, import GHCS_Constants * and use "cons.something"
cons = Constants() 