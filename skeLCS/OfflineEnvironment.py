from skeLCS.DataManagement import DataManagement

class OfflineEnvironment:
    def __init__(self,features,phenotypes,eLCS):
        """Initialize Offline Environment"""
        self.dataRef = 0
        self.storeDataRef = 0
        self.formatData = DataManagement(features,phenotypes,eLCS)

        self.currentTrainState = self.formatData.trainFormatted[0][self.dataRef]
        self.currentTrainPhenotype = self.formatData.trainFormatted[1][self.dataRef]

    def getTrainInstance(self):
        return (self.currentTrainState,self.currentTrainPhenotype)

    def newInstance(self):
        if self.dataRef < self.formatData.numTrainInstances-1:
            self.dataRef+=1
            self.currentTrainState = self.formatData.trainFormatted[0][self.dataRef]
            self.currentTrainPhenotype = self.formatData.trainFormatted[1][self.dataRef]
        else:
            self.resetDataRef()

    def resetDataRef(self):
        self.dataRef = 0
        self.currentTrainState = self.formatData.trainFormatted[0][self.dataRef]
        self.currentTrainPhenotype = self.formatData.trainFormatted[1][self.dataRef]

    def startEvaluationMode(self):
        """ Turns on evaluation mode.  Saves the instance we left off in the training data. """
        self.storeDataRef = self.dataRef

    def stopEvaluationMode(self):
        """ Turns off evaluation mode.  Re-establishes place in dataset."""
        self.dataRef = self.storeDataRef
