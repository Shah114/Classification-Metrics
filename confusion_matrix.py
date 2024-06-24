# Importing modules
import numpy as np
import pandas as pd

# Loading Data
data = pd.read_csv('C:/VS CODE/Projects/Classification/data.csv') # copy path of your data and paste here
df = data.copy()
df.head()

df.columns

df.info()

# Preprocessing
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df.diagnosis = le.fit_transform(df['diagnosis'])

df.drop(["Unnamed: 32"], axis=1, inplace=True)

df.head()

df.isnull().sum()


"""**Decision Tree clf**"""

# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

X = df.drop(["diagnosis"], axis=1) #features
y = df["diagnosis"] # target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train model
clf = DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

# Finding Accuracy
predictors = df.drop(['diagnosis'], axis=1)
target = df['diagnosis']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.25, random_state=0)

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('Naive Bayes', GaussianNB()))
models.append(('Decision Tree (CART)', DecisionTreeClassifier()))
models.append(('K-NN', KNeighborsClassifier()))
models.append(('SVM', SVC()))
models.append(('RandomForest Classiffier', RandomForestClassifier()))

for name, model in models:
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    from sklearn import metrics

    print("%s -> ACC: %%%.2f" %(name, metrics.accuracy_score(y_test, y_pred)*100))

# Making Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred,labels=[1,0]))
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(confusion_matrix(y_test, y_pred),annot=True,lw =2,cbar=False)
plt.ylabel("True Values")
plt.xlabel("Predicted Values")
plt.title("CONFUSION MATRIX VISUALIZATION")
plt.show()

# Finding F1 score
from sklearn.metrics import f1_score
f1_score(y_test, y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Making Reciever Operating Characteristic
import sklearn.metrics as metrics

probs = model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Reciever Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC=%0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Finding Logistic Loss (Log loss)
from sklearn.metrics import log_loss
log_loss(y_test, y_pred)