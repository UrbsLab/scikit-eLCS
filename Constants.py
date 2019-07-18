class Constants():
    def setConstants(self,parameterNames,parameterValues):
        self.parameterDictionary = dict(zip(parameterNames,parameterValues)) #zip parameter names and constants together

        for key in list(self.parameterDictionary.keys()):
            try:
                if '.' in self.parameterDictionary[key]:
                    self.parameterDictionary[key] = float(self.parameterDictionary[key])
                else:
                    self.parameterDictionary[key] = int(self.parameterDictionary[key])
            except:
                pass

    def referenceEnv(self,env):
        self.parameterDictionary['env'] = env