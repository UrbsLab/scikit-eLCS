Master Status: [![Build Status](https://travis-ci.com/UrbsLab/scikit-eLCS.svg?branch=master)](https://travis-ci.com/UrbsLab/scikit-eLCS)

# scikit-eLCS

This package includes a sci-kit compatible Python implementation of eLCS, a supervised learning variant of the Learning Classifier System. In general, Learning Classifier Systems (LCSs) are a classification of Rule Based Machine Learning Algorithms that have been shown to perform well on problems involving high amounts of heterogeneity and epistasis. Well designed LCSs are also highly human interpretable. LCS variants have been shown to adeptly handle supervised and reinforced, classification and regression, online and offline learning problems, as well as missing or unbalanced data. These characteristics of versatility and interpretability give LCSs a wide range of potential applications, notably those in biomedicine. This package is **still under active development** and we encourage you to check back on this repository for updates.

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

* Approximate Accuracy (tracked via trackingFrequency param)
* Average Population Generality (tracked via trackingFrequency param)
* Macropopulation Size
* Micropopulation Size
* Match Set, Correct Set Sizes
* Number of classifiers subsumed/deleted/covered
* Number of crossover/mutation operations performed
* Times for matching, deletion, subsumption, selection, evaluation

These values can then be exported as a csv after training is complete for analysis using the built in "exportIterationTrackingDataToCSV" method.

In addition, the package includes functionality that allows detailed training evaluation to be done at given iterations during the training process. At each interval, the package saves a snapshot of the rule population, along with evaluation accuracy and instance coverage. These snapshots of the rule population (including the final rule population) can then be exported as a csv after training is complete for analysis using the built in "exportRulePopulationAtIterationToCSV" and "exportFinalRulePopulationToCSV" methods.

For more information on the eLCS algorithm and how to use it, please refer to our [usage documentation](https://urbslab.github.io/scikit-eLCS/) and the Jupyter Notebooks inside this repository.

## License
Please see the repository [license](https://github.com/UrbsLab/scikit-eLCS/blob/master/LICENSE) for the licensing and usage information for scikit-eLCS.

Generally, we have licensed scikit-eLCS to make it as widely usable as possible.

## Contributing to scikit-eLCS
Scikit eLCS is still under active development and we will announced when we are set up for 3rd party contributions!
