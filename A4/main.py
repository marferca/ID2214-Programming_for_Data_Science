import os.path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import external
import feature_extraction
from progress.bar import Bar

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
#print("len(X_train) = {0}".format(len(X_train)))
#print("len(X_validation) = {0}".format(len(X_validation)))
#print("len(X_test) = {0}".format(len(X_test)))


# Random Forest Evaluation ---------------------------------------------------------------------------------------------
#n_estimators_values = [10,25,50,75,100,200,300]
n_estimators_values = [10,25,50]
max_depth_values = [10,20,30]
#max_depth_values = [10,20,30,40,50,60,70]
min_samples_split_values = [2,4]
#min_samples_split_values = [2,4,6,8]
class_weight_values = [None,"balanced"]

parameters = [(n_estimators,max_depth,min_samples,class_weight)
              for n_estimators in n_estimators_values
              for max_depth in max_depth_values
              for min_samples in min_samples_split_values
              for class_weight in class_weight_values]

best_AUC = 0
best_AUC_parameters = None
best_AUC_model = None
best_AUC_idx = None
results_randomforest_df = pd.DataFrame(index=range(len(parameters)), columns=["n_estimators", "max_depth", "min_samples_split", "class_weight", "AUC_validation", "AUC_test", "AUC_estimated"])
for i in Bar('Processing').iter(range(len(parameters))):
    clf = RandomForestClassifier(n_estimators=parameters[i][0],
                                 max_depth=parameters[i][1],
                                 min_samples_split=parameters[i][2],
                                 class_weight=parameters[i][3],
                                 random_state=0,
                                 criterion="gini")
    clf.fit(X_train, y_train)
    pred_prob = clf.predict_proba(X_validation)
    fpr, tpr, thresholds = metrics.roc_curve(y_validation, pred_prob[:, 1])
    AUC = metrics.auc(fpr, tpr)


    results_randomforest_df.loc[i] = [parameters[i][0],
                                      parameters[i][1],
                                      parameters[i][2],
                                      parameters[i][3],
                                      AUC,None,None]

    if AUC > best_AUC:
        best_AUC = AUC
        best_AUC_parameters = parameters[i]
        best_AUC_model = clf
        best_AUC_idx = i

pred_prob_test = clf.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_prob_test[:, 1])
AUC_test = metrics.auc(fpr, tpr)
results_randomforest_df.loc[i]["AUC_test"] = AUC_test

# End of Random Forest Evaluation --------------------------------------------------------------------------------------



print(results_randomforest_df)




#clf = RandomForestClassifier(n_estimators=100, max_depth=10, criterion="gini", min_samples_split=2, random_state=0, class_weight="balanced")
#clf.fit(X_train, y_train)


#pred_prob = clf.predict_proba(X_validation)
#fpr, tpr, thresholds = metrics.roc_curve(y_validation, pred_prob[:,1])
#AUC = metrics.auc(fpr, tpr)
#acc = clf.score(X_validation, y_validation, sample_weight=None)
#print("AUC: {0}".format(AUC))
#print("Acc: {0}".format(acc))
#y_pred = clf.predict(X_validation)
#external.plot_confusion_matrix(y_validation, y_pred, classes=np.array([0,1]))
#plt.show()

