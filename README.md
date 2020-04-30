Master Status: [![Build Status](https://travis-ci.com/UrbsLab/scikit-eLCS.svg?branch=master)](https://travis-ci.com/UrbsLab/scikit-eLCS)

# scikit-eLCS

The scikit-eLCS package includes a sklearn-compatible Python implementation of eLCS, a supervised learning variant of the Learning Classifier System, based off of UCS. In general, Learning Classifier Systems (LCSs) are a classification of Rule Based Machine Learning Algorithms that have been shown to perform well on problems involving high amounts of heterogeneity and epistasis. Well designed LCSs are also highly human interpretable. LCS variants have been shown to adeptly handle supervised and reinforced, classification and regression, online and offline learning problems, as well as missing or unbalanced data. These characteristics of versatility and interpretability give LCSs a wide range of potential applications, notably those in biomedicine. This package is **still under active development** and we encourage you to check back on this repository for updates.

eLCS, or Educational Learning Classifier System, implements the core components of a Michigan-Style Learning Classifier System (where the system's genetic algorithm operates on a rule level, evolving a population of rules with each their own parameters) in an easy to understand way, while still being highly functional in solving ML problems.

While Learning Classifier Systems are commonly applied to genetic analyses, where epistatis (i.e. feature interactions) is common, the eLCS algorithm implemented in this package can be applied to almost any supervised classification data set and supports:

* Feature sets that are discrete/categorical, continuous-valued or a mix of both
* Data with missing values
* Binary endpoints (i.e., classification)
* Multi-class endpoints (i.e., classification)
* eLCS does not currently support regression problems. We have built out the infrastructure for it do so, but have disabled its functionality for this version.

Built into this code, is a strategy to 'automatically' detect from the loaded data, these relevant above characteristics so that they don't need to be parameterized at initialization.

The core Scikit package only supports numeric data. However, an additional StringEnumerator Class is provided within the DataCleanup file that allows quick data conversion from any type of data into pure numeric data, making it possible for natively string/non-numeric data to be run by eLCS.

In addition, powerful data tracking collection methods are built into the scikit package, that continuously tracks features every iteration such as:

* Approximate Accuracy
* Average Population Generality
* Macropopulation Size
* Micropopulation Size
* Match Set, Correct Set Sizes
* Number of classifiers subsumed/deleted/covered
* Number of crossover/mutation operations performed
* Times for matching, deletion, subsumption, selection, evaluation

These values can then be exported as a csv after training is complete for analysis using the built in "exportIterationTrackingData" method.

In addition, the package includes functionality that allows the final rule population to be exported as a csv after training.

## Usage
For more information on the eLCS algorithm and how to use it, please refer to the ["eLCS User Guide"](https://github.com/UrbsLab/scikit-eLCS/blob/master/eLCS%20User%20Guide.ipynb) Jupyter Notebook inside this repository.

## Usage TLDR
```python
#Import Necessary Packages/Modules
from skeLCS.eLCS import eLCS
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

#Load Data Using Pandas
data = pd.read_csv('myDataFile.csv')
dataFeatures = data.drop(classLabel,axis=1).values
dataPhenotypes = data[classLabel].values

#Shuffle Data Before CV
formatted = np.insert(dataFeatures,dataFeatures.shape[1],dataPhenotypes,1)
np.random.shuffle(formatted)
dataFeatures = np.delete(formatted,-1,axis=1)
dataPhenotypes = formatted[:,-1]

#Initialize eLCS Model
model = eLCS(learningIterations = 5000)

#3-fold CV
print(np.mean(cross_val_score(model,dataFeatures,dataPhenotypes,cv=3)))
```

## License
Please see the repository [license](https://github.com/UrbsLab/scikit-eLCS/blob/master/LICENSE) for the licensing and usage information for scikit-eLCS.

Generally, we have licensed scikit-eLCS to make it as widely usable as possible.

## Installation
scikit-eLCS is built on top of the following Python packages:
<ol>
  <li> numpy </li>
  <li> pandas </li>
  <li> scikit-learn </li>
</ol>

Once the prerequisites are installed, you can install scikit-eLCS with a pip command:
```
pip/pip3 install scikit-elcs
```
We strongly recommend you use Python 3. scikit-eLCS does not support Python 2, given its depreciation in Jan 1 2020. If something goes wrong during installation, make sure that your pip is up to date and try again.
```
pip/pip3 install --upgrade pip
```

## Contributing to scikit-eLCS
scikit-eLCS is an open source project and we'd love if you could suggest changes!

<ol>
  <li> Fork the project repository to your personal account and clone this copy to your local disk</li>
  <li> Create a branch from master to hold your changes: (e.g. <b>git checkout -b my-contribution-branch</b>) </li>
  <li> Commit changes on your branch. Remember to never work on any other branch but your own! </li>
  <li> When you are done, push your changes to your forked GitHub repository with <b>git push -u origin my-contribution-branch</b> </li>
  <li> Create a pull request to send your changes to the scikit-eLCS maintainers for review. </li>
</ol>

**Before submitting your pull request**

If your contribution changes eLCS in any way, make sure you update the Jupyter Notebook documentation and the README with relevant details. If your contribution involves any code changes, update the project unit tests to test your code changes, and make sure your code is properly commented to explain your rationale behind non-obvious coding practices.

**After submitting your pull request**

After submitting your pull request, Travis CI will run all of the project's unit tests. Check back shortly after submitting to make sure your code passes these checks. If any checks come back failed, do your best to address the errors.
