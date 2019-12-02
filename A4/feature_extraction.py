import os.path
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem


# Generate fingerprints: Morgan fingerprint with radius 2
def smiles_to_fps_labels(df, fp_radius=2, fp_length=256):
    df_tmp = df["SMILES"]
    fps = []
    for smiles in df_tmp:
        molecule = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(
                 molecule, fp_radius, nBits=fp_length)
        fps.append(fp.ToBitString())
    fps = np.array(fps)
    fps = np.array([list(fp) for fp in fps], dtype=np.float32)
    if "ACTIVE" in df:
        labels = df["ACTIVE"]
        labels = np.array(labels)
    else:
        labels = 0
    return fps, labels
