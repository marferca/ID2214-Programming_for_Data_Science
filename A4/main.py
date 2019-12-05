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

# Load datasets
df_train = pd.read_csv(os.path.join("out", "training.csv"))
df_validation = pd.read_csv(os.path.join("out", "validation.csv"))
df_est = pd.read_csv(os.path.join("out", "test.csv"))
df_test = pd.read_csv(os.path.join("out", "test_smiles.csv")) # This is the one without labels
