import os
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 1. Load Data
train_df = pd.read_csv("train_sampled.csv")
test_df = pd.read_csv("test_sampled.csv")

TARGET = "satisfaction"
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET].map({"neutral or dissatisfied": 0, "satisfied": 1})

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET].map({"neutral or dissatisfied": 0, "satisfied": 1})

# 2. Impute missing values using training median
arrival_median = X_train["Arrival Delay in Minutes"].median()
X_train["Arrival Delay in Minutes"] = X_train[
    "Arrival Delay in Minutes"
].fillna(arrival_median)
X_test["Arrival Delay in Minutes"] = X_test["Arrival Delay in Minutes"].fillna(
    arrival_median
)

# Explicitly identify column types (fixes Pandas4Warning)
cat_cols = ["Gender", "Customer Type", "Type of Travel", "Class"]
num_cols = [c for c in X_train.columns if c not in cat_cols]

# 3. Preprocessor Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ]
)

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

os.makedirs("model", exist_ok=True)
joblib.dump(preprocessor, "model/preprocessor.joblib")
joblib.dump(arrival_median, "model/arrival_median.joblib")

# 4. Define All 5 Models (CalibratedClassifierCV replaces deprecated SVC probability=True)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10, min_samples_split=10, random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": CalibratedClassifierCV(
        estimator=SVC(kernel="rbf", C=1.0, random_state=42), ensemble=False
    ),
}

# 5. Train, Evaluate & Save Artifacts
print("=" * 82)
print(
    f"{'ML Model Name':<24} | {'Acc':<6} | {'AUC':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'MCC':<6}"
)
print("=" * 82)

for name, clf in models.items():
    clf.fit(X_train_proc, y_train)

    # Save model artifact
    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(clf, os.path.join("model", filename))

    # Evaluate
    y_pred = clf.predict(X_test_proc)
    y_proba = clf.predict_proba(X_test_proc)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    print(
        f"{name:<24} | {acc:.4f} | {auc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {mcc:.4f}"
    )

print("=" * 82)
test_df.to_csv("test_data.csv", index=False)
print("Artifacts saved successfully...")