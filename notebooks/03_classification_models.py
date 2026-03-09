# Classification models

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

data = load_iris()

X = data.data
y = data.target

model = RandomForestClassifier()

model.fit(X, y)

predictions = model.predict(X)

print("Accuracy:", accuracy_score(y, predictions))
