# Importing Modules
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Pipeline library for Training
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error

"""Data preprocessing"""
# Load Data
X = pd.read_csv('C:/VS CODE/Projects/Classification/Kaggle Task/train.csv') # Copy your data path
X_test = pd.read_csv('C:/VS CODE/Projects/Classification/Kaggle Task/test.csv') # Copy your data path
print(X.shape, X_test.shape)

# Remove rows with missing target, seperate target from predictors
X.dropna(axis=0, subset=['Survived'], inplace=True)
y = X.Survived
X.drop(['Survived'], axis=1, inplace=True)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X.columns if X[cname].nunique() < 10 and X[cname].dtype == 'object']

# Select numerical columns
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['float64', 'int64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X = X[my_cols].copy()
X_test = X[my_cols].copy()

X_ = pd.read_csv('C:/VS CODE/Projects/Classification/Kaggle Task/train.csv')

f, ax = plt.subplots(1, 2, figsize=(12, 4))

X_['Survived'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0])

sns.countplot(x="Pclass", hue='Survived', data=X_, ax=ax[1])
plt.show()

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

"""Modeling & Evaluate Score"""

rf_clf = RandomForestClassifier()

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf_clf)
])

print(cross_val_score(clf, X, y, cv=10).mean())

"""Model Evaluation

   Confusion Matrix
"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# train_test_split: 80%, 20%
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2)

# Modeling
rf_clf = RandomForestClassifier()

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf_clf)
])

# Train
clf.fit(X_train, y_train)
preds = clf.predict(X_val)

# Confusion matrix
cm = confusion_matrix(y_val, preds)
sns.heatmap(cm, annot=True, fmt="d")
plt.show()


"""Accuracy"""

clf.score(X_val, y_val)

accuracy_score(y_val, preds)

"""Classification report"""

from sklearn.metrics import classification_report
print(classification_report(y_val, preds))

"""Auc-Roc curve"""


from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


fpr, tpr, thresholds = roc_curve(y_val, preds)

plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.xlim([-0.02, 1])
plt.ylim([0, 1.02])
plt.legend(loc="lower right")
plt.show()

"""Logistic Loss"""

from sklearn.metrics import log_loss
log_loss(y_val, preds)