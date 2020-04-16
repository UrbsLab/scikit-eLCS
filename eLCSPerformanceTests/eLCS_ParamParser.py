from eLCS_Constants import *
import os
import copy


# ---------------------------------

class ParamParser:
    def __init__(self, dataFile, cv=False,learningIterations="10000", trackingFrequency=100000, N=1000,
                 p_spec=0.5, discreteAttributeLimit=10,nu=5, chi=0.8, upsilon=0.04, theta_GA=25,
                 theta_del=20, theta_sub=20, acc_sub=0.99, beta=0.2, delta=0.1, init_fit=0.01, fitnessReduction=0.1,
                 doSubsumption=True, selectionMethod='tournament', theta_sel=0.5,randomSeed = False,labelInstanceID='InstanceID',labelPhenotype="class",
                 labelMissingData="NA",doPopulationReboot=False,popRebootPath='ExampleRun_eLCS_50000'):

        self.parameters = {}
        self.parameters['learningIterations'] = learningIterations
        self.parameters['trackingFrequency'] = trackingFrequency
        self.parameters['N'] = N
        self.parameters['p_spec'] = p_spec
        self.parameters['discreteAttributeLimit'] = discreteAttributeLimit
        self.parameters['nu'] = nu
        self.parameters['chi'] = chi
        self.parameters['upsilon'] = upsilon
        self.parameters['theta_GA'] = theta_GA
        self.parameters['theta_del'] = theta_del
        self.parameters['theta_sub'] = theta_sub
        self.parameters['acc_sub'] = acc_sub
        self.parameters['beta'] = beta
        self.parameters['delta'] = delta
        self.parameters['init_fit'] = init_fit
        self.parameters['fitnessReduction'] = fitnessReduction
        self.parameters['doSubsumption'] = doSubsumption
        self.parameters['selectionMethod'] = selectionMethod
        self.parameters['theta_sel'] = theta_sel
        self.parameters['randomSeed'] = randomSeed
        self.parameters['labelInstanceID'] = labelInstanceID
        self.parameters['labelPhenotype'] = labelPhenotype
        self.parameters['labelMissingData'] = labelMissingData
        self.parameters['doPopulationReboot'] = doPopulationReboot
        self.parameters['popRebootPath'] = popRebootPath
        self.parameters['cv'] = cv
        self.parameters['dataFile'] = dataFile
        cons.setConstants(self.parameters)  # Store run parameters in the 'Constants' module.