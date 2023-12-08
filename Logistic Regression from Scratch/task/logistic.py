# Imported packages
import numpy as np
import pandas as pd
from math import log
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

"""Stage 1: Sigmoid function

Description

In this project, we will work on a classification algorithm that makes 
predictions when a dependent variable assumes discrete values. Logistic 
regression is arguably the simplest solution. In the case of binary 
classification (class 0 or class 1), it uses a sigmoid function to estimate how 
likely an observation belongs to class 1.

we will work with the Wisconsin Breast Cancer Dataset from the sklearn library.
We also want to standardize the features as they are measured in different 
units using Z-standardization

Objectives

1 - Create the CustomLogisticRegression class
2 - Create the __init__ method
3 - Create the sigmoid method
4 - Create the predict_proba method

"""

"""Stage 2: Gradient descent with MSE

Description

In this stage, we need to estimate the coef_ (weight) values by gradient descent 
on the Mean squared error cost function. Gradient descent is an optimization 
technique for finding the local minimum of a cost function by first-order 
differentiating. To be precise, we're going to implement the Stochastic 
gradient descent (SGD).

Objectives

1 - Implement the fit_mse method
2 - Implement the predict method

"""

"""Stage 3: Log-Loss

Description

The Mean squared error cost function produces a non-convex graph with the local 
and global minimums when applied to a sigmoid function. If a weight value is 
close to a local minimum, gradient descent minimizes the cost function by the 
local (not global) minimum. This presents grave limitations to the Mean squared 
error cost function if we apply it to binary classification tasks. The Log-loss 
cost function may help to overcome this issue.

Objectives

Implement the fit_log_loss method in class CustomLogisticRegression

"""

# Load the dataset
data = load_breast_cancer(as_frame=True)
X = data.data[['worst concave points', 'worst perimeter', 'worst radius']]
y = data.target

# Standardize X
for feature in X.columns.tolist():
    feature_mean = X[feature].mean()
    feature_std = X[feature].std()
    X[feature] = (X[feature] - feature_mean) / feature_std

# Split the datasets to training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                    random_state=43)


class CustomLogisticRegression:
    """A simple logistic regression model."""

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.coef_ = None
        self.epoch = []

    def sigmoid(self, t):
        return 1 / (1 + np.exp(- t))

    def predict_proba(self, row, coef_):
        """Predict the probability that <row> belongs to Class 1, given the
        weights <coef_> for each feature."""
        if self.fit_intercept:
            t = np.dot(row, coef_[1:]) + np.array([coef_[0]])
        else:
            t = np.dot(row, coef_)
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train):
        """Update the <self.coef_> attribute by estimating the optimal weight
        values using the Gradient Descent method with the Mean Squared Error
        cost function.

        We will start with all weight values being equal to zero.

        """
        if self.fit_intercept:
            count = len(X_train.columns.tolist()) + 1
        else:
            count = len(X_train.columns.tolist())

        # Initialize the weights
        coef_ = np.zeros(count)

        # Determining the number of rows
        N = len(X_train)

        # Training loop
        for _ in range(self.n_epoch):
            errors = []
            i = 0
            for _, row in X_train.iterrows():
                y_hat = self.predict_proba(row, coef_)
                # Update all weights
                if self.fit_intercept:
                    ind = 1
                    for value in row:
                        coef_[ind] = coef_[ind] - self.l_rate * (
                                y_hat - y_train.iloc[i]) * y_hat * (
                                             1 - y_hat) * value
                        ind = ind + 1
                    coef_[0] = coef_[0] - self.l_rate * (
                            y_hat - y_train.iloc[i]) * y_hat * (
                                       1 - y_hat)
                else:
                    ind = 0
                    for value in row:
                        coef_[ind] = coef_[ind] - self.l_rate * (
                                y_hat - y_train.iloc[i]) * y_hat * (
                                             1 - y_hat) * value
                        ind = ind + 1
                error = ((y_hat - y_train.iloc[i]) ** 2) * (1 / N)
                errors.append(error.item())
                i = i + 1
            self.epoch.append(errors)

        self.coef_ = coef_

    def fit_log_loss(self, X_train, y_train):

        if self.fit_intercept:
            count = len(X_train.columns.tolist()) + 1
        else:
            count = len(X_train.columns.tolist())

        # Initialize the weights
        coef_ = np.zeros(count)

        # Determine number of rows
        N = len(X_train)

        # Training loop
        for _ in range(self.n_epoch):
            errors = []
            i = 0
            for _, row in X_train.iterrows():
                y_hat = self.predict_proba(row, coef_)
                # Update all weights
                if self.fit_intercept:
                    ind = 1
                    for value in row:
                        coef_[ind] = coef_[ind] - (self.l_rate * (
                                y_hat - y_train.iloc[i]) * value) / N
                        ind = ind + 1
                    coef_[0] = coef_[0] - (self.l_rate * (
                            y_hat - y_train.iloc[i])) / N
                else:
                    ind = 0
                    for value in row:
                        coef_[ind] = coef_[ind] - (self.l_rate * (
                                y_hat - y_train.iloc[i]) * value) / N
                        ind = ind + 1
                error = (y_train.iloc[i] * log(y_hat) + (
                        (1 - y_train.iloc[i]) * log(1 - y_hat))) * (
                                - 1 / N)
                errors.append(error.item())
                i = i + 1
            self.epoch.append(errors)

        self.coef_ = coef_

    def predict(self, X_test, cut_off=0.5):

        predictions = self.predict_proba(X_test.to_numpy(), self.coef_)
        predictions[predictions >= cut_off] = 1
        predictions[predictions < cut_off] = 0
        return predictions


# Load the dataset
data = load_breast_cancer(as_frame=True)
X = data.data[['worst concave points', 'worst perimeter', 'worst radius']]
y = data.target

# Standardize X
for feature in X.columns.tolist():
    feature_mean = X[feature].mean()
    feature_std = X[feature].std()
    X.loc[:, feature] = (X.loc[:, feature] - feature_mean) / feature_std

# Split the datasets to training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                    random_state=43)

# Fit a model for each cost function method
log_loss_model = CustomLogisticRegression(fit_intercept=True, l_rate=0.01,
                                          n_epoch=1000)
log_loss_model.fit_log_loss(X_train, y_train)
mse_model = CustomLogisticRegression(fit_intercept=True, l_rate=0.01,
                                     n_epoch=1000)
mse_model.fit_mse(X_train, y_train)

# Fit a model using sklearn
sklearn_model = LogisticRegression(fit_intercept=True)
sklearn_model.fit(X_train, y_train)

# Determine the error values for the first and last epoch of training for
# fit_mse and fit_log_loss methods
mse_first_epoch_error = mse_model.epoch[0]
mse_last_epoch_error = mse_model.epoch[-1]

log_loss_first_epoch_error = log_loss_model.epoch[0]
log_loss_last_epoch_error = log_loss_model.epoch[-1]

# Predict y_hat values for the test set with all three models
mse_y_hat = mse_model.predict(X_test)
log_loss_y_hat = log_loss_model.predict(X_test)
sklearn_y_hat = sklearn_model.predict(X_test)

# Calculate the accuracy scores for the test set for all three models
mse_accuracy = accuracy_score(y_test.to_numpy(), mse_y_hat)
log_loss_accuracy = accuracy_score(y_test.to_numpy(), log_loss_y_hat)
sklearn_accuracy = accuracy_score(y_test.to_numpy(), sklearn_y_hat)

# Printing the required dictionary
output_dict = {'mse_accuracy': mse_accuracy,
               'logloss_accuracy': log_loss_accuracy,
               'sklearn_accuracy': sklearn_accuracy,
               'mse_error_first': mse_first_epoch_error,
               'mse_error_last': mse_last_epoch_error,
               'logloss_error_first': log_loss_first_epoch_error,
               'logloss_error_last': log_loss_last_epoch_error}

print(output_dict, end='\n\n')

# Printing the answers for the questions

min_mse_first = format(min(mse_first_epoch_error), '.5f')
min_mse_last = format(min(mse_last_epoch_error), '.5f')
max_logloss_first = format(max(log_loss_first_epoch_error), '.5f')
max_logloss_last = format(max(log_loss_last_epoch_error), '.5f')

print(f"""Answers to the questions:
1) {min_mse_first}
2) {min_mse_last}
3) {max_logloss_first}
4) {max_logloss_last}
5) expanded
6) expanded
""")
