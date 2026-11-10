import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import OneHotEncoder

def load_raw_data(data_folder:str) -> dict:
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

def load_preprocessed_data(data_folder:str) -> dict:
    """Finds all preprocessed .csv files in the data folder and loads them as a dictionary of DataFrames.

    Returns:
        dict: A dictionary of DataFrames, each corresponding to a specific preprocessed .csv file found."""
    data_files = glob.glob(os.path.join(data_folder, "*_preprocessed.csv"))
    if not data_files:
        _preprocess_data(data_folder)
        data_files = glob.glob(os.path.join(data_folder, "*_preprocessed.csv"))
    data = {}
    for file in data_files:
        city_name = os.path.basename(file).replace("_preprocessed.csv", "")
        data[city_name] = pd.read_csv(file)
    return data

def _preprocess_data(data_folder:str):
    # ----------- Load Data -----------
    data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),"data")
    data = load_raw_data(data_folder)
    print(f"Loaded data. Found {len(data)} datasets: {list(data.keys())}")

    # ----------- Preprocessing ----------------
    # Drop 4 missing output power rows in Amhst dataset
    data["amhst"] = data["amhst"].dropna(subset=["Output Power"])

    for city in data.keys():
        df = data[city]
        # Create Clear-Sky Index labels
        df["Clear-Sky Index"] = df["GHI"] / df["Clearsky GHI"]
        df["Sky Condition"] = pd.cut(df["Clear-Sky Index"], bins=[-float("inf"), 0.4, 0.85, float("inf")], labels=["Cloudy", "Partly-Cloudy", "Clear"])

        # Days elapsed before the start of each month (non-leap year), indexed by month 1-12
        month_offsets = np.array([0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])

        # Day of year, 1..365. Feb 29 -> 31 + 29 = 60, the same value as Mar 1 (59 + 1),
        # so leap days are handled with no special case.
        df["Day_n"] = month_offsets[df["Month"].to_numpy()] + df["Day"]

        # Periodic encodings
        day_angle  = 2 * np.pi * (df["Day_n"] - 1) / 365
        hour_angle = 2 * np.pi * (df["Hour"] + df["Minute"] / 60) / 24

        df["Day_sin"]  = np.sin(day_angle)
        df["Day_cos"]  = np.cos(day_angle)
        df["Hour_sin"] = np.sin(hour_angle)
        df["Hour_cos"] = np.cos(hour_angle)

        # Encode "Cloud Type" as a categorical feature using one-hot encoding
        cloud_cats = pd.Categorical(df["Cloud Type"], categories=range(13))
        cloud_dummies = pd.get_dummies(cloud_cats, prefix="Cloud", dtype=int)
        cloud_dummies.index = df.index  # Match df indexes
        df = pd.concat([df, cloud_dummies], axis=1)

        df = df.drop(columns=["Month", "Day", "Hour", "Minute", "Day_n", "Clear-Sky Index", "Cloud Type"])
        df.to_csv(os.path.join(data_folder, f"{city}_preprocessed.csv"), index=False)
        print(f"Preprocessed data for {city} and saved to {city}_preprocessed.csv")