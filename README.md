# Analysis My Machine Learning Projects 

In this so called project, I will study and analyze all the things that I have learned while doing small machine learning projects. 

First, we will look into all the ML models that I have used till this point from scikit-learn module. Then we will look into other functions of scikit-learn that helps us to preprocess our data and evaluate the model.

Finally, we will study all the helper function that we used from pandas, numpy and seaborn library.

## ML Models from *scikit-learn*
###  1. LinearRegression()
It is a type of supervised machine learning model. This model tries to find a linear relationship (basically a straight line) between the input features (independent variables) and the target variable (dependent variable). It's one of the simplest machine learning model to start with.

I used LinearRegression() model in the following projects :-
1. [Medical Insurance Cost Prediction](https://github.com/sreedevnair/Medical-Insurance-Cost-Prediction)
2. [Predicting A Country's Olympic Medal Count](https://github.com/sreedevnair/Predicting-A-Country-s-Olympic-Medal-Count)
3. [Price Prediction of Used Car using ML](https://github.com/sreedevnair/Price-prediction-of-used-cars-using-basic-ML)

### 2. LogisticRegression()
Logistic regression is also a supervised machine learning algorithm used for classification problems. Logistic regression predicts **probabilities** that help us classify data into categories (for example, deciding if an email is spam or not). It then uses a threshold value for the predicted probability to decide to which category the given input belongs to.

Instead of drawing a straight line, logistic regression draws an "S"-shaped curve (called a sigmoid function) that squashes the output between 0 and 1.

I used LogisticRegression() model in the following projects :-
1. [Credit Card Detection using Basic ML](https://github.com/sreedevnair/Credit-card-fraud-detection-using-ML)
2. [Heart Disease Detection using ML](https://github.com/sreedevnair/Heart-disease-prediction-using-basic-ML)

### 3. Lasso()
Lasso stands for Least Absolute Shrinkage and Selection Operator. It is a type of linear regression that adds a regularization term to the model. It achieves this by adding a penalty term to the traditional linear regression model, which encourages sparse solutions, where some coefficients are forced to be exactly zero.

While linear regression uses everything to make a prediction, Lasso helps by trimming away the less important stuff.

I used Lasso() model in the following projects :-
1. [Price Prediction of Used Car using ML](https://github.com/sreedevnair/Price-prediction-of-used-cars-using-basic-ML)
2. [Price Prediction of Used Cars using ML Part 2](https://github.com/sreedevnair/Price-prediction-of-used-cars-using-ML-part-2)

### 4. SVM()
A Support Vector Machine (SVM) is a machine learning model used mainly for classification tasks, but it can also be used for regression. It works by finding a boundary that separates different classes of data points, like dividing apples and oranges on a graph. 

It focuses on the data points closest to the boundary, called **support vectors**, and tries to maximize the space between these points and the boundary to ensure a clear separation. If the data isn't easily separable by a straight line, SVM uses a trick called the "kernel" to transform the data into a higher dimension where it becomes easier to separate the classes.

### 5. RandomForestTree()
It is an ensemble learning method that builds multiple decision trees and combines their results to make more accurate and robust predictions. By averaging multiple decision trees, random forests reduce the risk of overfitting, which is common with individual decision trees.

[Wine Quality Prediction using ML](https://github.com/sreedevnair/Wine-quality-prediction-using-basic-ML)

### 6. XGBRegressor()
