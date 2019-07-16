class Constants():
    def setConstants(self,parameterNames,parameterValues):
        self.parameterDictionary = dict(zip(parameterNames,parameterValues)) #zip parameter names and constants together

    def referenceEnv(self,env):
        self.parameterDictionary['env'] = env