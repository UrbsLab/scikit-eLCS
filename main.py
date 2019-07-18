import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from eLCS import eLCS

'''Separates out features into np_array of shape [number of items, number of features per item] 
and labels into np_array of shape [number of items]'''
data = pd.read_csv('FakeDataCSV.csv',sep=',') #Puts data from csv into indexable np arrays
dataFeatures, dataPhenotypes = data.drop('class', axis=1).values, data['class'].values

'''Separates out parameters from parameters file'''
parameters = pd.read_csv('ConfigFile.csv',sep=',')
parameterNames, parameterValues = parameters["ParameterNames"].values, parameters["ParameterValues"].values
clf = eLCS(parameterNames,parameterValues)
clf = clf.fit(dataFeatures,dataPhenotypes)

#print(np.mean(cross_val_score(clf, features, labels)))
print("hi")