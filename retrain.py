import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import joblib
import os

df = pd.read_csv("data/Telco-Customer-Churn.csv")
df = df.drop("customerID", axis=1)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
y = df["Churn"]
X = df.drop("Churn", axis=1)
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(knn, "models/knn_model.pkl")
joblib.dump(lr, "models/logistic_model.pkl")
joblib.dump(dt, "models/decision_tree.pkl")
joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(X.columns, "models/feature_columns.pkl")

print("Models and scaler saved successfully.")
