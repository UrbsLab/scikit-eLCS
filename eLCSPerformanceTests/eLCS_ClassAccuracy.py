"""
Name:        eLCS_ClassAccuracy.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     November 1, 2013
Description: Manages the logistical aspects of balance accuracy calculations which can handle unbalanced datasets, and/or datasets with multiple discrete classes.
             
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

class ClassAccuracy:
    def __init__(self):
        """ Initialize the accuracy calculation for a single class """
        self.T_myClass = 0      #For binary class problems this would include true positives
        self.T_otherClass = 0   #For binary class problems this would include true negatives 
        self.F_myClass = 0      #For binary class problems this would include false positives 
        self.F_otherClass = 0   #For binary class problems this would include false negatives 


    def updateAccuracy(self, thisIsMe, accurateClass):
        """ Increment the appropriate cell of the confusion matrix """
        if thisIsMe and accurateClass:
            self.T_myClass += 1
        elif accurateClass:
            self.T_otherClass += 1
        elif thisIsMe:
            self.F_myClass += 1
        else:
            self.F_otherClass += 1
        
        
    def reportClassAccuracy(self):
        """ Print to standard out, summary on the class accuracy. """
        print("-----------------------------------------------")
        print("TP = "+str(self.T_myClass))
        print("TN = "+str(self.T_otherClass))
        print("FP = "+str(self.F_myClass))
        print("FN = "+str(self.F_otherClass))
        print("-----------------------------------------------")