import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv("wine.data", header=None)

x = data.iloc[:, 1:]
y = data.iloc[:, 0]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=5)

standardScaler = StandardScaler()
xTrain = standardScaler.fit_transform(xTrain)
xTest = standardScaler.transform(xTest)

labelEncoder = LabelEncoder()
yTrain = labelEncoder.fit_transform(yTrain)
yTest = labelEncoder.transform(yTest)

classifier = KNeighborsClassifier()
classifier.fit(xTrain, yTrain)

yPred = classifier.predict(xTest)

matrix = confusion_matrix(yTest, yPred)

sns.heatmap(matrix, annot=True, fmt="d")
plt.title("confusion matrix")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.show()

print(classification_report(yTest, yPred))
