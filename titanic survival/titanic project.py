import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("train.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Info")
print(df.info())

# ----------------------------
# Handle Missing Values
# ----------------------------
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
df["Fare"].fillna(df["Fare"].median(), inplace=True)

# ----------------------------
# Encode Categorical Variables
# ----------------------------
encoder = LabelEncoder()

df["Sex"] = encoder.fit_transform(df["Sex"])
df["Embarked"] = encoder.fit_transform(df["Embarked"])

# ----------------------------
# Feature Engineering
# ----------------------------

# Family Size
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Age Category
df["AgeCategory"] = pd.cut(df["Age"],
                           bins=[0,12,59,100],
                           labels=[0,1,2])

df["AgeCategory"] = df["AgeCategory"].astype(int)

# ----------------------------
# Drop Unwanted Columns
# ----------------------------
df.drop(["Name","Ticket","Cabin","PassengerId"], axis=1, inplace=True)

# ----------------------------
# Features and Target
# ----------------------------
X = df.drop("Survived", axis=1)
y = df["Survived"]

# ----------------------------
# Train Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================
# Logistic Regression
# ============================

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("\n========== Logistic Regression ==========")

print("Accuracy:",
      accuracy_score(y_test, lr_pred))

print(classification_report(y_test, lr_pred))

print("Confusion Matrix")
print(confusion_matrix(y_test, lr_pred))

# ============================
# Random Forest
# ============================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\n========== Random Forest ==========")

print("Accuracy:",
      accuracy_score(y_test, rf_pred))

print(classification_report(y_test, rf_pred))

print("Confusion Matrix")
print(confusion_matrix(y_test, rf_pred))

# ============================
# ROC Curve
# ============================

prob = rf.predict_proba(X_test)[:,1]

fpr,tpr,threshold = roc_curve(y_test, prob)

roc_auc = auc(fpr,tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr,tpr,label="AUC = %0.2f"%roc_auc)
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ============================
# Feature Importance
# ============================

importance = rf.feature_importances_

features = X.columns

feature_df = pd.DataFrame({
    "Feature":features,
    "Importance":importance
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print(feature_df)

plt.figure(figsize=(8,5))
plt.bar(feature_df["Feature"],
        feature_df["Importance"])

plt.xticks(rotation=45)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.show()

# ============================
# Best Model
# ============================

lr_acc = accuracy_score(y_test, lr_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print("\n--------------------------------")

if rf_acc > lr_acc:
    print("Best Model : Random Forest")
else:
    print("Best Model : Logistic Regression")

print("--------------------------------")