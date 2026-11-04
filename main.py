import torch
import pandas as pd
import os
import glob

SEEDS = [42, 789, 256]

def load_data(data_folder:str) -> pd.DataFrame:
    """Finds all .xlsx files in the data folder and loads required sheets from the first one as a dictionary of DataFrames.

    Returns:
        dict: A dictionary of DataFrames, each corresponding to a specific sheet in the first .xlsx file found.
    Args:
        data_folder (str): The path to the folder containing the .xlsx files.
    """
    data_files = glob.glob(os.path.join(data_folder, "*.xlsx"))
    if not data_files:
        raise FileNotFoundError(f"No .xlsx files found in {data_folder}")
    if len(data_files) > 1:
        raise RuntimeError(f"Multiple .xlsx files found in {data_folder}: {data_files}")
    data_file = data_files[0]
    amhst_data = pd.read_excel(data_file, sheet_name="Amhst 5hr-daily '18-'20")
    davis3_data = pd.read_excel(data_file, sheet_name="Davis 5hr-daily '14-'16")
    davis6_data = pd.read_excel(data_file, sheet_name="Davis 5hr-daily '11-'16")
    huron3_data = pd.read_excel(data_file, sheet_name="Huron 5hr-daily '14-'16")
    huron6_data = pd.read_excel(data_file, sheet_name="Huron 5hr-daily '11-'16")
    sntbarb3_data = pd.read_excel(data_file, sheet_name="Snt.Barb 5hr-daily '14-'16")
    sntbarb6_data = pd.read_excel(data_file, sheet_name="Snt.Barb 5hr-daily '11-'16")
    lajolla3_data = pd.read_excel(data_file, sheet_name="LaJolla 5hr-daily '14-'16")
    lajolla6_data = pd.read_excel(data_file, sheet_name="LaJolla 5hr-daily '11-'16")
    data = {
        "amhst": amhst_data,
        "davis3": davis3_data,
        "davis6": davis6_data,
        "huron3": huron3_data,
        "huron6": huron6_data,
        "sntbarb3": sntbarb3_data,
        "sntbarb6": sntbarb6_data,
        "lajolla3": lajolla3_data,
        "lajolla6": lajolla6_data,
    }
    return data

# ----------- Load Data -----------
data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),"data")
data = load_data(data_folder)
print(f"Loaded data. Found {len(data)} datasets: {list(data.keys())}")

# ----------- Preprocessing ----------------
# Drop 4 missing output power rows in Amhst dataset
data["amhst"] = data["amhst"].dropna(subset=["Output Power"])


# ----------- Problem 1: Supervised Classification -------------
# Sky-condition class: Multi-class (3) - classify from Clear-Sky Index k = GHI / Clearsky GHI
#   Clear: 0.85<=k<=1.0, Partly Cloudy: 0.4<=k<0.85, Cloudy: k<0.4
#   Must exclude GHI and Clearsky GHI as features, as they define the label
# Generation-regime class: Multi-class (3) - classify Output Power into Low / Medium / High for each city

# Balanced accuracy - mean of per-class recall. Prefer over raw accuracy
# Make a confusion matrix, per-class metrics, find sky/regime classes that are the hardest to classify

# Use KNN, Ensemble of Naive Bayes with Bagging, maybe randomForest of decision trees, and use MLP as deep learning option


# ------------- Problem 2: Supervised Regression ---------------
# Predict Output Power from all 21 input features
# 3 different setups: 
#   1. Same-city: chronological 80/20 split in one city
#   2. Cross-city (zero-shot): train on one city, test on another with no finetuning. Normalize per city
#   3. Sequence variant: feed the last K=12 steps, predict the next Output Power (short-term forecast)

# Root-mean-square-error (RMSE) or Mean-absolute-error (MAE) for each method and setup
# Give every regression number as mean +/- std over >= 3 random seeds. Always report RMSE with nRMSE for cross-city result comparisons
# Learning-curve, or prediction-vs-truth plot

# Vector shaped data, don't use CNN or recurrent. Use Multilayer Perceptron (MLP) for deep learning solution. Use linear regression as a baseline

# Sequence variant: Use Vanilla RNN and compare to LSTM and GRU variants

# -------------- Problem 3: Dimension Reduction ----------------
# Reduce feature vector to d = 2, 5, and 10 components, then use as inputs to Problem 1 and Problem 2 models
# Choose d based on explained-variance curve / reconstruction error curve
# Compare results with and without dimension reduction

# Show reconstruction error for variational autoencoder (VAE) / autoencoder (AE), as well as explained-variance for PCA/KPCA
#   Reconstruction MSE - mean squared error between original features (x) and reconstruction (x-hat), lower preserves more info
#       Plot reconstruction MSE against the number of components, picking a number d at the elbow of the curve
#   Explained-variance ratio - fraction of variance captured by first d components (1 is lossless), show the cumulative curve
# Report classification balanced accuracy, and regression RMSE / MAE, comparing to the results without dimension reduction
# Visualize with 2-d t-SNE or UMAP scatter colored by class labels or Output-Power bins
#   Do nearest neighbors in the embedding space to visualize class separability and structure. Also just inspect scatter for separated structure

# ----------- Problem 4: Semi-supervised Learning --------------
# Pick base task (either problem 1 or 2)
# Randomly withold labels so only 10%, 30%, and 50% of the data have labels for training
# Test-set remains fully labeled for evaluation

# Plot label-efficiency curve, comparing metric to % labels given
# Compare model to a supervised-only model trained only on the % of data that has labels (ie ignore unlabeled data fro supervised baseline)
#   Report gain over supervised-only at each fraction
# Evaluate if at 10% labels, we outperform the supervised-only baseline
# Area under curve (AUC) of label-efficiency curve

# --------------- Problem 5: Transfer Learning -----------------
# Source = Davis6, Target = Amhst3
# Baselines to beat: Zero-shot = source model applied directly to target, Few-shot = model trained only on target
# Target-domain FMSE/accuracy for Zero-shot, Few-shot, and Transfer Learning, in one table

# Zero-shot performance - metric for Target for source trained model, no target fine-tuning
# Few-shot performance - metric for Target after updating on k target samples (ex: k= 10, 50, 100)
# Transfer Gain - gain = metric_target - metric_target_only_baseline (model trained only on target data)
