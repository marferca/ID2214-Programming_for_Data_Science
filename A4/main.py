import os.path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import external
import feature_extraction

# Load datasets
df_train = pd.read_csv(os.path.join("out", "training.csv"))
df_validation = pd.read_csv(os.path.join("out", "validation.csv"))
df_test = pd.read_csv(os.path.join("out", "test.csv"))
df_test_final = pd.read_csv(os.path.join("out", "test_smiles.csv"))

# Get fingerprints from data sets
X_train, y_train = feature_extraction.smiles_to_fps_labels(df_train)
X_validation, y_validation = feature_extraction.smiles_to_fps_labels(df_validation)
X_test, y_test = feature_extraction.smiles_to_fps_labels(df_test)
X_test_final, _ = feature_extraction.smiles_to_fps_labels(df_test_final)
print("len(X_train) = {0}".format(len(X_train)))
print("len(X_validation) = {0}".format(len(X_validation)))
print("len(X_test) = {0}".format(len(X_test)))

# Random forest classifier
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0, class_weight="balanced")
clf.fit(X_train, y_train)


pred_prob = clf.predict_proba(X_validation)
fpr, tpr, thresholds = metrics.roc_curve(y_validation, pred_prob[:,1])
AUC = metrics.auc(fpr, tpr)
acc = clf.score(X_validation, y_validation, sample_weight=None)
print("AUC: {0}".format(AUC))
print("Acc: {0}".format(acc))
y_pred = clf.predict(X_validation)
external.plot_confusion_matrix(y_validation, y_pred, classes=np.array([0,1]))
plt.show()

