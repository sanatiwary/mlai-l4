import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("data.csv")
print(data.head(10))

x = data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = data["species"]
print(x.head())
print(y.head())

from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=5)
print(xTrain)
print(xTest)
print(yTrain)
print(yTest)

from sklearn.preprocessing import StandardScaler, LabelEncoder

standardScaler = StandardScaler()
standardScaler.fit_transform(xTrain)

labelEncoder = LabelEncoder()
labelEncoder.fit_transform(yTrain)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()
classifier.fit(xTrain, yTrain)

standardScaler.transform(xTest)
yPred = classifier.predict(xTest)

labelEncoder.transform(yTest)

from sklearn.metrics import classification_report, confusion_matrix

matrix = confusion_matrix(yTest, yPred)

sns.heatmap(matrix, annot=True, fmt="d")
plt.title("confusion matrix")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.show()

print(classification_report(yTest, yPred))