# Analysis My Machine Learning Projects 

In this "*project*", I will study and analyze all the things that I have learned while doing small machine learning projects. 

First, we will look into all the ML models that I have used till this point from `scikit-learn` module. Then we will look into other functions of `scikit-learn` that helps us to preprocess our data and evaluate the model.

Finally, we will study all the helper function that we used from `pandas`, `numpy` and `seaborn` library.

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

Project : [Loan Status Prediction using ML](https://github.com/sreedevnair/Loan-approval-prediction-using-basic-ML)

### 5. RandomForestTree()
It is an ensemble learning method that builds multiple decision trees and combines their results to make more accurate and robust predictions. By averaging multiple decision trees, random forests reduce the risk of overfitting, which is common with individual decision trees.

Project : [Wine Quality Prediction using ML](https://github.com/sreedevnair/Wine-quality-prediction-using-basic-ML)

### 6. XGBRegressor()
It is a machine learning model provided by the XGBoost library, which stands for "Extreme Gradient Boosting." It's specifically designed for regression tasks, where the goal is to predict a continuous target variable.

XGBoost uses a technique called gradient boosting, which builds an ensemble of decision trees, where each new tree corrects the errors of the previous ones. This iterative process improves model accuracy.

Project : [Predicting Calories Burnt During Exercise](https://github.com/sreedevnair/Calories-Burnt-During-Exercise-Prediction)

### 7. CatBoostRegressor()
It is a machine learning model used to predict numerical values, especially when working with datasets that include categorical features. It can handle such categorical features without needing to convert them into numbers first.

It also uses the gradient boosting technique. It automatically deals with missing values and supports various data preprocessing steps, making it easier to use with less manual intervention.

Project : [Predicting Sales using Upgini and CatBoost](https://github.com/sreedevnair/Sales-Forecasting-using-upgini-and-catboost)

<br>

## Helper Functions from *scikit-learn*

### 1. train_test_split()
##### *from sklearn.model_selection import train_test_split*
We use this fucntion to split our dataset into training and testing set. We need to pass the feature set and the target set as the parameter. We can also pass other parameters :-
1. `test_size` : By default it's .25, which means 25% of the dataset will be used for testing and the other 75% for the training.
2. `random_state` : This is to make sure that our data split remains same during every execution.
3. `stratiify` : It basically ensures that the data split have the same proportions every time.

### 2. sklearn.metrics
This module provides various functions to evaluate the performance of machine learning models. It includes a variety of metrics that can be used to assess models for both classification and regression tasks. For example, `mean_absolute_error()`, `r2_score`, `accuracy_score` etc

### 3. OneHotEncoder
##### *from sklearn.preprocessing import OneHotEncoder*
It is bascially used to encode categorical values into numerical values for the machine learning model.

[1.2 Using sklearn OneHotEncoder() Class](https://github.com/sreedevnair/Price-prediction-of-used-cars-using-ML-part-2?tab=readme-ov-file#2-to-check-for-prediction-improvement-using-the-categorical-feature)

<br>

## Helper Functions from *pandas*
### 1. `isnull().sum()`
While pre-procssing the dataset, we first check whether any of the features have any missing values. We do this by using `.isnull()` function. This is function will return the complete DataFrame with *True* or *False* values. If the value is not present (i.e. a Null value), it will show *True*, otherwise *False*.

To check the total number of Null values present in the DataFrame, we use `.isnull().sum()`. This will return Series with the number of Null values in each column (feature).

### 2. `value_counts()`
We use this function to check the occurences of unique values of a particular column.

Syntax :- `<df_name>[<'column_name'>].value_counts()`

### 3. `unique()`
It is used to get all the unique values present in that column.

Syntax :- `<df_name>[<'column_name'>].unique()`

### 4. `dropna()`
We use this function to get a new DataFrame with all the NaN values dropped. If we want to save the changes in the original DataFrame, then we need to specify the parameter `inplace=True`.

Syntax :- `<df_name>.dropna(inplace=True)`

### 5. `corr()`
To find the correlation between each non-categorical features, we DataFrame.corr() function. But before using that, we first need to drop all the categorical values as Pandas can't find correlation between them.

It will return a matrix with correlation of each column with each other.

If it's a positive correlation, that means that the features are directly correlated. If it's a negative correlation, that means that the features are inversely correlated.

### 6. `replace()`
We mainly use this function to encode categorical values into numerical values.  Inside this function, first we need to pass a dictionary. Inside this dictionary, we use column name as keys and another set of dictionary as it's value. Inside the nested dictionary, we pass the original values of that column as keys and the new data that we want to replace it as the value.

### 7. `sample()`
This function is used to randomly select number of rows from a dataset.

Syntax :-` <df_name>.sample(number_of_rows)`

<br>

## Helper Functions from *Numpy*
### 1. `np.median()`
Computes the median of an array, which is the middle value in a sorted dataset.

### 2. `np.std()`
Calculates the standard deviation, a measure of how spread out the values in a dataset are.

### 3. `np.reshape()`
Reshapes an array to a new shape.

### 4. `np.dot()`
Computes the dot product of two arrays, essential in linear algebra operations used in many machine learning algorithms.

<br>

## Helper Functions from *Seaborn*

### 1. `sns.distplot()`
It stands for Distribution Plot and is used for plotting distributions of continuous variables, mainly through a histogram.

### 2. `sns.countplot()`
It is used for visualizing the count of observations in each category of a categorical variable, mainly through bar graph.

### 3. `sns.heatmap()`
To visualize the correlation in a better way, we use a heatmap. Heatmaps are used to show relationships between two variables, one plotted on each axis.

We create heatmaps using `sns.heatmap()` function. If the correlation between each feature is more, the darker will be the color.

### 4. `sns.lmplot()`
It stands for Linear Model plot. It is basically used to plot 2 variables kinda in the form of scatter plot.

<br>

-----
