import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import initial data
df = pd.read_csv(os.path.join("initial_data", "training_smiles.csv"))

# Get input features and labels from the out set
X = df["SMILES"]
y = df["ACTIVE"]

# Split into training validation and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)


# Save to csv files
training = pd.DataFrame(data=pd.concat([X_train,y_train], axis=1), columns=["SMILES","ACTIVE"])
validation = pd.DataFrame(data=pd.concat([X_validation,y_validation], axis=1), columns=["SMILES","ACTIVE"])
test = pd.DataFrame(data=pd.concat([X_test,y_test], axis=1), columns=["SMILES","ACTIVE"])
training.to_csv(os.path.join("out", "training.csv"))
validation.to_csv(os.path.join("out", "validation.csv"))
test.to_csv(os.path.join("out", "test.csv"), columns="SMILES")


# Analyze the out set
#labels = np.array(df["ACTIVE"])
#counts, _ = np.histogram(labels, bins=2)
#density = counts/np.sum(counts)
#print(density)
#plt.hist(labels)
#plt.show()
# We have imbalanced out!!!

# Resampling the dataset (undersampling)
#df_tmp = df.copy()
#labels = np.array(df_tmp["ACTIVE"])
#counts, _ = np.histogram(labels, bins=2)
#active_df = df_tmp.loc[df_tmp["ACTIVE"] == 0]
#inactive_df = df_tmp.loc[df_tmp["ACTIVE"] == 1]
#resamp_df = pd.concat([active_df.sample(int(counts[1]+(0.25)*counts[1])),inactive_df])
#print("Length of the new dataset: {0}".format(len(resamp_df)))
#df = resamp_df
#labels = np.array(df["ACTIVE"])
#counts, _ = np.histogram(labels, bins=2)
#density = counts/np.sum(counts)
#print(density)




