# Titanic Survival Prediction using Logistic Regression

A machine learning project that predicts whether a passenger survived the Titanic disaster using **Logistic Regression**. 
This project demonstrates the Implementation of Logistic Regression Algorithm from Scratch to evaluate the Titanic Disaster Dataset.
**This project was implemented using only Numpy and Pandas**

---

## Project Overview

The goal of this project is to build a binary classification model that predicts passenger survival based on demographic and travel information.

This project was built to practice:

* Data preprocessing
* Handling missing values
* Understand Log Loss (Binary Cross Entropy Loss)
* Training a Logistic Regression model from scratch
* Evaluating classification performance
* Making predictions on unseen data

---

## Dataset

The dataset used is the **Titanic: Machine Learning from Disaster** dataset from Kaggle.

### Files

* `train.csv` – Training dataset containing survival labels.
* `test.csv` – Test dataset used for prediction.
* `gender_submission.csv` – Sample submission file.

---

## Features Used

| Feature  | Description                       |
| -------- | --------------------------------- |
| Pclass   | Passenger class (1st, 2nd, 3rd)   |
| Sex      | Passenger gender                  |
| Age      | Passenger age                     |
| SibSp    | Number of siblings/spouses aboard |
| Parch    | Number of parents/children aboard |
| Fare     | Ticket fare                       |
| Embarked | Port of embarkation               |

The following columns were removed because they provide little predictive value or require more advanced feature engineering:

* PassengerId
* Name
* Ticket
* Cabin
* Embarked

---
**I am still learning, thus I decided to remove this for the time being.**

## Data Preprocessing

The following preprocessing steps were performed:

* Filled missing values in **Age** using the median.
* Encoded **Sex** as:
  * Male → 0
  * Female → 1
    
* Split the dataset into training and testing sets.

---

## Model

**Algorithm**

* Logistic Regression

The model was implemented from scratch.

---

## Mathematics

### Linear Model

The linear combination of features is

$[
z = \mathbf{X}\mathbf{w} + b
]$

where

($\mathbf{X}$) is the feature matrix
($\mathbf{w}$) is the weight vector
(b) is the bias

---
### Sigmoid Function

The predicted probability is computed using the sigmoid function

[
$\hat{y}=\sigma(z)=\frac{1}{1+e^{-z}}$
]

This converts the linear output into a probability between 0 and 1.
---
Binary Cross-Entropy Loss

For a dataset containing (m) training examples,

[
$J(\mathbf{w},b)=
-\frac{1}{m}
\sum_{i=1}^{m}
\left[
y^{(i)}\log\left(\hat{y}^{(i)}\right)
+
\left(1-y^{(i)}\right)
\log\left(1-\hat{y}^{(i)}\right)
\right]$
]

The objective of training is to minimize this loss.
---

### Gradient Descent Updates

The gradients are computed as

[
$dW=\frac{1}{m}X^{T}(A-Y)$
]

[
$db=\frac{1}{m}\sum(A-Y)$
]

The parameters are updated using

[
$W:=W-\alpha dW$
]

[
$b:=b-\alpha db$
]

where ($\alpha$) is the learning rate.

---
Vectorized Forward Propagation

The forward pass is fully vectorized:

[
$Z = XW + b$
]

[
$A=\sigma(Z)$
]

where

(Z) is the linear output
(A) is the predicted probability vector

Vectorization allows all training examples to be processed simultaneously without explicit Python loops.
---

<img src="./TITANIC/dead-alive.png" alt=".Titanic" width="500"/>

## Kaggle Result

The trained model was also evaluated on the Kaggle Titanic competition.

**Public Leaderboard**

* **Score 0.75358**

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib


## Future Improvements

Some possible improvements include:

* Feature engineering using passenger titles (Mr, Mrs, Miss, etc.)
* Creating a Family Size feature
* Extracting cabin deck information
* Hyperparameter tuning
* Comparing Logistic Regression with:

  * Decision Trees
  * Random Forests
  * Support Vector Machines
  * Gradient Boosting
  * XGBoost

---

## Key Learning Outcomes

Through this project, I learned how to:

* Clean real-world datasets
* Handle missing values
* Feature Engineering
* Train a binary classification model
* Evaluate model performance
* Interpret classification metrics
* Build an end-to-end machine learning pipeline

---
## Note
Most of this README.md file is written using ChatGPT.
If you happen to find any error, please ignore it 😀

## License

This project is for educational purposes.
