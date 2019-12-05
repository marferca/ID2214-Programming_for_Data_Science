import os.path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import external
import feature_extraction
import pickle

# Load datasets
df_train = pd.read_csv(os.path.join("out", "training.csv"))
df_validation = pd.read_csv(os.path.join("out", "validation.csv"))
df_est = pd.read_csv(os.path.join("out", "test.csv"))
df_test = pd.read_csv(os.path.join("out", "test_smiles.csv")) # This is the one without labels

# Random Forest Evaluation ---------------------------------------------------------------------------------------------
fp_length_value = 1024
n_estimators_value = 200
max_depth_value = 40
min_samples_split_value = 4
class_weight_value = "balanced"

X_train, y_train = feature_extraction.smiles_to_fps_labels(df_train, fp_length=fp_length_value)
X_validation, y_validation = feature_extraction.smiles_to_fps_labels(df_validation, fp_length=fp_length_value)
X_est, y_est = feature_extraction.smiles_to_fps_labels(df_est, fp_length=fp_length_value)
X_test, _ = feature_extraction.smiles_to_fps_labels(df_test, fp_length=fp_length_value)

clf = RandomForestClassifier(n_estimators=n_estimators_value,
                             max_depth=max_depth_value,
                             min_samples_split=min_samples_split_value,
                             class_weight=class_weight_value,
                             random_state=0,
                             criterion="gini")
clf.fit(X_train, y_train)
acc = clf.score(X_validation, y_validation, sample_weight=None)
pred_prob = clf.predict_proba(X_validation)
fpr, tpr, thresholds = metrics.roc_curve(y_validation, pred_prob[:, 1])
AUC = metrics.auc(fpr, tpr)

acc_est = clf.score(X_est, y_est, sample_weight=None)
pred_prob_est = clf.predict_proba(X_est)
fpr, tpr, thresholds = metrics.roc_curve(y_est, pred_prob_est[:, 1])
AUC_est = metrics.auc(fpr, tpr)
print("acc_val: {0}; AUC_val: {1}; acc_est: {2}; AUC_est: {3}".format(acc, AUC, acc_est, AUC_est))
y_est_pred = clf.predict(X_est)
external.plot_confusion_matrix(y_est, y_est_pred, classes=np.array([0, 1]), normalize=True, title="RF Normalized confusion matrix")
plt.show()

# Save the model to disk
#filename = 'RF_best_model.sav'
#pickle.dump(clf, open(os.path.join("out", filename), 'wb'))
