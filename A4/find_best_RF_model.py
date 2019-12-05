import os.path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import external
import feature_extraction
from progress.bar import Bar
import pickle


# Load datasets (they contain smiles + active(classes))
df_train = pd.read_csv(os.path.join("out", "training.csv"))         # Load training dataframe
df_validation = pd.read_csv(os.path.join("out", "validation.csv"))  # Load validation dataframe
df_est = pd.read_csv(os.path.join("out", "test.csv"))               # Load test dataframe (aka estimation dataset)
df_test = pd.read_csv(os.path.join("out", "test_smiles.csv"))       # Load test dataframe (this is the one without labels)


# Random Forest Evaluation ---------------------------------------------------------------------------------------------
fp_length_values = [128, 256, 512, 1024, 2048]              # Different number of bits (fingerprint) FEATURE EXTRACTION
n_estimators_values = [10, 25, 50, 75, 100, 200, 300]       # Define different number of trees       MODEL
max_depth_values = [10, 20, 30, 40, 50, 60, 70]             # Define max depth trees                 MODEL
min_samples_split_values = [2, 4, 6, 8, 16]                 # Define min samples to split            MODEL
class_weight_values = [None, "balanced"]                    # Define class weight mode               MODEL

# Create all possible combinations for above parameters values
parameters = [(fp_length, n_estimators, max_depth, min_samples, class_weight)
              for fp_length in fp_length_values
              for n_estimators in n_estimators_values
              for max_depth in max_depth_values
              for min_samples in min_samples_split_values
              for class_weight in class_weight_values]


# Initialize variables to store best model parameters
best_AUC = 0                                                # Initialize variable to store best AUC value
best_AUC_parameters = None                                  # Initialize variable to store best AUC parameters
best_AUC_model = None                                       # Initialize variable to store best AUC model
best_AUC_idx = None                                         # Initialize variable to store best AUC parameters index
last_fp_length = 0                                          # Initialize last fp length (This reduces computation time)
results_RF_df = pd.DataFrame(index=range(len(parameters)),  # Initializes the dataframe where the results will be stored
                                       columns=["Model",
                                                "fp_length",
                                                "n_estimators",
                                                "max_depth",
                                                "min_samples_split",
                                                "class_weight",
                                                "Acc_validation",
                                                "AUC_validation",
                                                "Acc_est",
                                                "AUC_est"])

for i in Bar('Processing').iter(range(len(parameters))):    # For each parameter combination
    if last_fp_length != parameters[i][0]:  # If the last fingerprint length is != from the current one extract features
        last_fp_length = parameters[i][0]   # Update fingerprint length value
        # Extract features from training and validation dataframes
        X_train, y_train = feature_extraction.smiles_to_fps_labels(df_train, fp_length=parameters[i][0])
        X_validation, y_validation = feature_extraction.smiles_to_fps_labels(df_validation, fp_length=parameters[i][0])

    # Define new model with i parameters
    clf = RandomForestClassifier(n_estimators=parameters[i][1],
                                 max_depth=parameters[i][2],
                                 min_samples_split=parameters[i][3],
                                 class_weight=parameters[i][4],
                                 random_state=0,
                                 criterion="gini")
    clf.fit(X_train, y_train)                                               # Fit model with training dataset
    acc = clf.score(X_validation, y_validation, sample_weight=None)         # Compute accuracy (validation)
    pred_prob = clf.predict_proba(X_validation)                             # Get prediction probabilities (validation)
    fpr, tpr, thresholds = metrics.roc_curve(y_validation, pred_prob[:, 1]) # Get FPR, TPR to compute AUC (validation)
    AUC = metrics.auc(fpr, tpr)                                             # Get AUC (validation)

    # Fill parameters and results of the actual model
    results_RF_df.loc[i] = ["RF",
                            parameters[i][0],
                            parameters[i][1],
                            parameters[i][2],
                            parameters[i][3],
                            parameters[i][4],
                            acc,
                            AUC,
                            None,
                            None]

    if AUC > best_AUC:  # Check if the current model is better in terms of AUC, if so store its parameters
        best_AUC = AUC                          # Update best AUC
        best_AUC_parameters = parameters[i]     # Update best AUC parameters
        best_AUC_model = clf                    # Update best AUC model
        best_AUC_idx = i                        # Update best parameters index

# Plot best model
print("BEST MODEL")
print("Parameters: {0}; Best AUC: {1}; best_AUC_idx: {2};".format(best_AUC_parameters,best_AUC,best_AUC_idx))
print("")

# For the best model calculate accuracy and AUC on our test set (estimation test)
X_est, y_est = feature_extraction.smiles_to_fps_labels(df_est, fp_length=best_AUC_parameters[0])
acc_est = best_AUC_model.score(X_est, y_est, sample_weight=None)
results_RF_df.loc[best_AUC_idx]["Acc_est"] = acc_est
pred_prob_est = best_AUC_model.predict_proba(X_est)
fpr, tpr, thresholds = metrics.roc_curve(y_est, pred_prob_est[:, 1])
AUC_est = metrics.auc(fpr, tpr)
results_RF_df.loc[best_AUC_idx]["AUC_est"] = AUC_est
results_RF_df.to_csv(os.path.join("out", "RF_results.csv"))

# Save best Random Forest model to disk
filename = 'RF_best_model.sav'
pickle.dump(best_AUC_model, open(os.path.join("out", filename), 'wb'))

y_est_pred = best_AUC_model.predict(X_est)
external.plot_confusion_matrix(y_est, y_est_pred, classes=np.array([0,1]))
plt.show()

# End of Random Forest Evaluation --------------------------------------------------------------------------------------
