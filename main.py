import torch
import pandas as pd
import os
import glob

SEED = 42

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


# ------------- Problem 2: Supervised Regression ---------------


# -------------- Problem 3: Dimension Reduction ----------------


# ----------- Problem 4: Semi-supervised Learning --------------


# --------------- Problem 5: Transfer Learning -----------------