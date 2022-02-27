# Multivariate Regression Analysis and Gradient Boosting Project

## Introduction

In this project I will be presenting the concepts of Multivaraite Regression Analysis and Gradient Boosting. I will explain the theory behind these concepts and then apply the algorithms to the "Cars.csv" & "Boston Housing Prices.csv" data sets where we have multiple input features and one ouput variable to be considered. For both methods, the final crossvalidated mean square error and absolute square error are reported to compare which method achieves better results.

## Theory

### Multivariate Regression Analysis

Multivaraite Regression is a supervised machine learning algorithm that is an extension of multiple regresion with one dependent variable and multiple input features. In general, we want 
![render]https://quicklatex.com/cache3/83/ql_727e19e5864eccfa9f70d31a3b061683_l3.png
where **F** represents the model, or the regressor, that we consider. For our variable selection, we want to select only the features that are relevant to our model. If the functional input/output model is 
![render]https://quicklatex.com/cache3/3c/ql_1fa137109f9bfbcded6478061847233c_l3.png
then it is possible that only a subset of variables are relevant and we need to eliminate the variables that we think are not useful. To represent variable selection in a functional way, we can thikn of multiplying each variable from the model by a binary weight. For example, a weight of zero represents an unimportant feature and a weight of one represents a relevant feature. We can show this by multiplying our model by the weights: 
![render]https://quicklatex.com/cache3/79/ql_55b97fff23ce5bd352b9b9fe1a3ee379_l3.png
where the weights are either zero or one. In addition, the vector of binary weights, *w*, gives us a "sparsity pattern" for variable selection.


### Gradient Boosting


## Modeling Approach


## Conclusion
