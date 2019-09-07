# scikit-eLCS

This package includes a sci-kit compatible Python implementation of eLCS, a supervised learning variant of the Learning Classifier System. In general, Learning Classifier Systems (LCSs) are a classification of Rule Based Machine Learning Algorithms that have been shown to perform well on problems involving high amounts of heterogeneity and epistasis. Well designed LCSs are also highly human interpretable. LCS variants have been shown to adeptly handle supervised and reinforced, classification and regression, online and offline learning problems, as well as missing or unbalanced data. These characteristics of versatility and interpretability give LCSs a wide range of potential applications, notably those in biomedicine. This package is **still under active development** and we encourage you to check back on this repository for updates.

eLCS, or Educational Learning Classifier System, implements the core components of a Michigan-Style Learning Classifier System (where the system's genetic algorithm operates on a rule level,  evolving a population of rules with each their own parameters) in an easy to understand way, while still being highly functional in solving ML problems.

While Learning Classifier Systems are commonly applied to genetic analyses, where epistatis (i.e. feature interactions) is common, the eLCS algorithm implemented in this package can be applied to almost any supervised classification data set and supports:

* Feature sets that are discrete/categorical, continuous-valued or a mix of both
* Data with missing values
* Binary endpoints (i.e., classification)
* Multi-class endpoints (i.e., classification)
* Continuous endpoints (i.e., regression)

Built into this code, is a strategy to 'automatically' detect from the loaded data, these relevant characteristics so that they don't need to be parameterized at initialization.

## License
Please see the repository [license](https://github.com/UrbsLab/scikit-eLCS/blob/master/LICENSE) for the licensing and usage information for scikit-rebate.

Generally, we have licensed scikit-eLCS to make it as widely usable as possible.

## Usage
We have designed the eLCS algorithm to be integrated directly into scikit-learn machine learning workflows. For example, below is an example of a 3-fold cross-validation being run on a complete dataset of the 6-bit Multiplexer problem.

```python
from sklearn.model_selection import cross_val_score
from eLCS import eLCS

data = pd.read_csv('Multiplexer6.csv',sep=',') #Puts data from csv into indexable np arrays
dataFeatures, dataPhenotypes = data.drop('class', axis=1).values, data['class'].values

clf = eLCS(learningIterations=5000) #You can add parameters if you want, but default values are preprogrammed

print(np.mean(cross_val_score(clf, dataFeatures, dataPhenotypes)))
>>> 0.9922480620155039

```
For more information on the eLCS algorithm and how to use it, please refer to our [usage documentation](https://urbslab.github.io/scikit-eLCS/).

## Contributing to scikit-eLCS
Scikit eLCS is still under active development and we will announced when we are set up for 3rd party contributions!
