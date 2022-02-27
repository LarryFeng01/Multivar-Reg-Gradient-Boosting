# Multivariate Regression Analysis and Gradient Boosting Project

## Introduction

In this project I will be presenting the concepts of Multivariate Regression Analysis and Gradient Boosting. I will explain the theory behind these concepts and then apply the algorithms to the "Cars.csv" & "Boston Housing Prices.csv" data sets where we have multiple input features and one output variable to be considered. For both methods, the final cross-validated mean square error and absolute error are reported to compare which method achieves better results.

## Theory

### Multivariate Regression Analysis

Multivariate Regression is a supervised machine learning algorithm that is an extension of multiple regression with one dependent variable and multiple input features. In general, we want 
![render](https://quicklatex.com/cache3/83/ql_727e19e5864eccfa9f70d31a3b061683_l3.png)
where **F** represents the model, or the regressor, that we consider. For our variable selection, we want to select only the features that are relevant to our model. If the functional input/output model is 
![render](https://quicklatex.com/cache3/3c/ql_1fa137109f9bfbcded6478061847233c_l3.png)
then it is possible that only a subset of variables are relevant, and we need to eliminate the variables that we think are not useful. To represent variable selection in a functional way, we can think of multiplying each variable from the model by a binary weight. For example, a weight of zero represents an unimportant feature and a weight of one represents a relevant feature. We can show this by multiplying our model by the weights: 
![render](https://quicklatex.com/cache3/79/ql_55b97fff23ce5bd352b9b9fe1a3ee379_l3.png)
where the weights are either zero or one. In addition, the vector of binary weights, *w*, gives us a "sparsity pattern" for variable selection.

In the case of multiple linear regression we have 
![render](https://quicklatex.com/cache3/7f/ql_6a5def16ddf50e35f533e7a4951d677f_l3.png)
and the sparsity pattern means that a subset of the betas are equal to zero. So we assume that
![render](https://quicklatex.com/cache3/db/ql_dd306140cac2ecad44ebb06f29f627db_l3.png)
and we want the coefficients, beta.

To solve this equation, we multiply all variables in the equation by transpose X, solve the equation for beta, and then take the expectation of the equation:
[render](https://quicklatex.com/cache3/bb/ql_b96c2be0f7e2943936a690af12525abb_l3.png)

So, we will take these theoretical equations into consideration to continue with our code and make an algorithm that we can apply to some data sets.

### Gradient Boosting

Gradient Boosting is one of the most powerful, supervised, algorithms in the field of machine learning. Since gradient boosting is one of the boosting algorithms it is used to minimize bias error in the models.

Boosting is an ensemble method, which is a way of combining predictions from several models into one. It does so by taking each predictor sequentially and modeling it based on its previous result's error. 

The algorithm can be used for predicting not only continuous target variables but also categorical target variables. When used as a regressor (continuous), the cost function is the MSE.

Assuming we have a regressor **F**, we make a prediction F(xi) for the observation xi. To improve the predictions, we can regard **F** as a "weak learner" and therefore train a decision tree (h) where the new output is yi-F(xi). Therefore, there are increased chances that the new regressor (F + h) is better than the old regressor **F**.


## Modeling Approach

Similar to our other regressions, we need our previous kernels of Tricubic, Epanechnikov, and Locally Weighted Regression. But this time, we will use an edited version of Boosted_lwr() and a new function booster():
```
def boosted_lwr(X,y,xnew,kern,tau,intercept,model_boosting,nboost):
  Fx = lw_reg(X,y,X,kern,tau,intercept)
  output = booster(X,y,xnew,kern,tau,model_boosting,nboost)
  return output 

def booster(X,y,xnew,kern,tau,model_boosting,nboost):
  Fx = lw_reg(X,y,X,kern,tau,True)
  Fx_new = lw_reg(X,y,xnew,kern,tau,True)
  new_y = y - Fx
  output = Fx
  output_new = Fx_new
  for i in range(nboost):
    model_boosting.fit(X,new_y)
    output += model_boosting.predict(X)
    output_new += model_boosting.predict(xnew)
    new_y = y - output
  return output_new
```
Now, we import the data, standardize it, and send it through our models. First let's take a look at the algorithms being applied to the Cars.csv dataset:
```
mse_blwr = []
mse_xgb = []
mae_blwr = []
mae_xgb = []

for i in range(5):
  kf = KFold(n_splits=10,shuffle=True,random_state=i)
  # this is the Cross-Validation Loop
  for idxtrain, idxtest in kf.split(X):
    xtrain = scale.fit_transform(X[idxtrain])
    xtest = scale.transform(X[idxtest])
    ytest = y[idxtest]
    xtest = X[idxtest]
    
    yhat_blwr = boosted_lwr(xtrain,ytrain,xtest,Tricubic,1,True,model_boosting,2)
    model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=1)
    model_xgb.fit(xtrain,ytrain)
    yhat_xgb = model_xgb.predict(xtest)

    mse_blwr.append(mse(ytest,yhat_blwr))
    mse_xgb.append(mse(ytest,yhat_xgb))
    mae_blwr.append(mae(ytest,yhat_blwr))
    mae_xgb.append(mae(ytest,yhat_xgb))
print('The Cross-validated Mean Squared Error for Boosted LWR is : '+str(np.mean(mse_blwr)))
print('The Cross-validated Mean Absolute Error for Boosted LWR is : '+str(np.mean(mae_blwr)))
print('The Cross-validated Mean Squared Error for XGB is : '+str(np.mean(mse_xgb)))
print('The Cross-validated Mean Absolute Error for XGB is : '+str(np.mean(mae_xgb)))
```
Through this loop, we print the cross-validated MSE and MAE for the boosted locally weighted regression and xgboost models. The results are as follows:
```
The Cross-validated Mean Squared Error for Boosted LWR is : 16.64637417328024
The Cross-validated Mean Absolute Error for Boosted LWR is : 2.919317014780272
The Cross-validated Mean Squared Error for XGB is : 16.559417572167884
The Cross-validated Mean Absolute Error for XGB is : 2.988851259944227
```
Next, let's use the Boston housing data set where our input features are 'rooms', 'distance', and 'crime'. Our dependent variable is the price of the house, 'cmedv'. Running the same algorithm above, we get:
```
The Cross-validated Mean Squared Error for Boosted LWR is : 25.718465193602828
The Cross-validated Mean Absolute Error for Boosted LWR is : 3.185977260729185
The Cross-validated Mean Squared Error for XGB is : 27.233277716221878
The Cross-validated Mean Absolute Error for XGB is : 3.47179031384518
```
## Conclusion

As reported by our results, the boosted locally weighted regression (BLWR) is very close in errors for the cars data set compared to the xgboost model. BLWR has a slightly higher MSE, but also a slightly lower MAE. This means that the two models are very comparable to each other, and I am not able to choose a "better" model between the two just yet. If we look at the results for the Boston housing data, BLWR has a lower error for both, and it is significantly lower. So, although I would need more tests to get a better average of the errors, with the current results I would claim that BLWR is the better method.
