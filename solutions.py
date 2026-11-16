import numpy as np
import pandas as pd
from models import train_and_predict_nb, train_and_predict_knn, train_and_predict_rf, train_and_predict_mlpc, train_and_predict_linreg, train_and_predict_mlpr, train_and_predict_sequence_reg, RecurrentRegressionModel, TransformerRegressorModel
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, root_mean_squared_error, mean_absolute_error
from numpy.lib.stride_tricks import sliding_window_view

# ----------- Problem 1: Supervised Classification -------------
# Sky-condition class: Multi-class (3) - classify from Clear-Sky Index k = GHI / Clearsky GHI
#   Clear: 0.85<=k<=1.0, Partly-Cloudy: 0.4<=k<0.85, Cloudy: k<0.4
#   Must exclude GHI and Clearsky GHI as features, as they define the label
# Generation-regime class: Multi-class (3) - classify Output Power into Low / Medium / High for each city

# Balanced accuracy - mean of per-class recall. Prefer over raw accuracy
# Make a confusion matrix, per-class metrics, find sky/regime classes that are the hardest to classify

# Use Naive Bayes, KNN, and randomForest of decision trees, and use MLP as deep learning option

def problem1_1(preprocessed_data:dict, seed:int):
    # Sky condition classification - Clear-Sky Index (k) = GHI / Clearsky GHI
    # Same-city classification: chronological 75/25 split in each city
    for city in preprocessed_data.keys():
        # load preprocessed data for city, split into train/test sets, removing unapplicable features
        df = preprocessed_data[city]

        # Create Clear-Sky Index labels
        df["Clear-Sky Index"] = df["GHI"] / df["Clearsky GHI"]
        df["Sky Condition"] = pd.cut(df["Clear-Sky Index"], bins=[-float("inf"), 0.4, 0.85, float("inf")], labels=["Cloudy", "Partly-Cloudy", "Clear"])

        train_df = df.iloc[:int(len(df)*0.75)]
        train_labels = train_df['Sky Condition'].to_numpy()
        # Drop year - should not be used as a training feature for future predictions, drop all features that define the label
        train_features = train_df.drop(columns=['Year', 'GHI', 'Clearsky GHI', 'Sky Condition', "Clear-Sky Index"])
        test_df = df.iloc[int(len(df)*0.75):]
        test_features = test_df.drop(columns=['Year', 'GHI', 'Clearsky GHI', 'Sky Condition', "Clear-Sky Index"])
        test_labels = test_df['Sky Condition'].to_numpy()
        df = df.drop(columns=["Clear-Sky Index","Sky Condition"])

        # Drop cloud encoding features for Naive Bayes, as creating guassian distributions for categorical features is not appropriate. Keep for other models 
        # Naive Bayes
        nb_predictions = train_and_predict_nb(train_features.filter(cond=~(train_features.columns.str.startswith("Cloud_")), axis=1), train_labels, test_features.filter(cond=~(test_features.columns.str.startswith("Cloud_")), axis=1))

        classes = np.unique(train_labels)
        print("\n\nCity: ", city)
        print(balanced_accuracy_score(test_labels, nb_predictions))
        print(confusion_matrix(test_labels, nb_predictions, labels=classes))


        # KNN
        knn_predictions = train_and_predict_knn(train_features, train_labels, test_features)

        print(balanced_accuracy_score(test_labels, knn_predictions))
        print(confusion_matrix(test_labels, knn_predictions, labels=classes))

        # Random Forest
        rf_predictions = train_and_predict_rf(train_features, train_labels, test_features, seed=seed)

        print(balanced_accuracy_score(test_labels, rf_predictions))
        print(confusion_matrix(test_labels, rf_predictions, labels=classes))

        # MLP Classifier
        mlpc_predictions = train_and_predict_mlpc(train_features, train_labels, test_features, seed=seed)

        print(balanced_accuracy_score(test_labels, mlpc_predictions))
        print(confusion_matrix(test_labels, mlpc_predictions, labels=classes))

def problem1_2(preprocessed_data:dict, seed:int):
    # Generation regime classification - Labeling Output Power as low / medium / high
    # Same-city classification: chronological 75/25 split in each city
    for city in preprocessed_data.keys():
        # load preprocessed data for city, split into train/test sets, removing unapplicable features
        df = preprocessed_data[city]
        train_df = df.iloc[:int(len(df)*0.75)]

        # Split the training data into terciles, 3 equally sized populations, based on output power
        train_power = train_df['Output Power'].to_numpy()
        power_tercile_boundaries = np.quantile(train_power, [1/3, 2/3])
        # Use the boundaries to label each sample as 0 (low), 1 (medium), or 2 (high)
        train_labels = np.digitize(train_power, power_tercile_boundaries)

        # Drop year - should not be used as a training feature for future predictions, drop all features that define the label, encoding existing features
        train_features = train_df.drop(columns=['Year','Output Power'])

        # Use the same boundaries to generate test labels
        test_df = df.iloc[int(len(df)*0.75):]
        test_power = test_df['Output Power'].to_numpy()
        test_labels = np.digitize(test_power, power_tercile_boundaries)
        test_features = test_df.drop(columns=['Year', 'Output Power'])

        # Drop cloud encoding features for Naive Bayes, as creating guassian distributions for categorical features is not appropriate. Keep for other models 
        # Naive Bayes
        nb_predictions = train_and_predict_nb(train_features.filter(cond=~(train_features.columns.str.startswith("Cloud_")), axis=1), train_labels, test_features.filter(cond=~(test_features.columns.str.startswith("Cloud_")), axis=1))

        classes = np.unique(train_labels)
        print("\n\nCity: ", city)
        print(balanced_accuracy_score(test_labels, nb_predictions))
        print(confusion_matrix(test_labels, nb_predictions, labels=classes))

        # KNN
        knn_predictions = train_and_predict_knn(train_features, train_labels, test_features)

        print(balanced_accuracy_score(test_labels, knn_predictions))
        print(confusion_matrix(test_labels, knn_predictions, labels=classes))

        # Random Forest
        rf_predictions = train_and_predict_rf(train_features, train_labels, test_features, seed=seed)

        print(balanced_accuracy_score(test_labels, rf_predictions))
        print(confusion_matrix(test_labels, rf_predictions, labels=classes))

        # MLP Classifier
        mlpc_predictions = train_and_predict_mlpc(train_features, train_labels, test_features, seed=seed)

        print(balanced_accuracy_score(test_labels, mlpc_predictions))
        print(confusion_matrix(test_labels, mlpc_predictions, labels=classes))

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

# Sequence variant: Use Vanilla RNN and compare to LSTM and GRU variants, use transformer for deep learning.

def problem2_same_city(preprocessed_data:dict, seed:int):
    for city in preprocessed_data.keys():
        # load preprocessed data for city, split into train/test sets, removing unapplicable features
        df = preprocessed_data[city]
        train_df = df.iloc[:int(len(df)*0.80)]
        train_labels = train_df['Output Power'].to_numpy()
        # Drop year - should not be used as a training feature for future predictions, drop all features that define the label, encoding existing features
        train_features = train_df.drop(columns=['Year','Output Power'])

        test_df = df.iloc[int(len(df)*0.80):]
        test_labels = test_df['Output Power'].to_numpy()
        test_features = test_df.drop(columns=['Year', 'Output Power'])

        # Find mean of observed test values for Normalized RMSE
        mean_test_label = np.mean(test_labels)

        # Linear Regression
        linreg_predictions = train_and_predict_linreg(train_features, train_labels, test_features)
        print("\n\nCity: ", city)
        print("\nLinear Regression")
        print(f"RMSE:\t{root_mean_squared_error(test_labels, linreg_predictions)}")
        print(f"NRMSE:\t{root_mean_squared_error(test_labels, linreg_predictions) / mean_test_label}")
        print(f"MAE:\t{mean_absolute_error(test_labels, linreg_predictions)}")

        # MLP Regressor
        mlpr_predictions = train_and_predict_mlpr(train_features, train_labels, test_features, seed=seed)
        print("\nMLP Regressor")
        print(f"RMSE:\t{root_mean_squared_error(test_labels, mlpr_predictions)}")
        print(f"NRMSE:\t{root_mean_squared_error(test_labels, mlpr_predictions) / mean_test_label}")
        print(f"MAE:\t{mean_absolute_error(test_labels, mlpr_predictions)}")
        

def problem2_cross_city(preprocessed_data:dict, seed:int):
    """Train on complete Davis (6yr) dataset, test on Huron, Santa Barbara, and La Jolla."""
    sixyr_datasets = ['davis6','huron6','sntbarb6','lajolla6']
    for train_city in sixyr_datasets:
        train_df = preprocessed_data[train_city]

        train_labels = train_df['Output Power'].to_numpy()
        # Drop year - should not be used as a training feature for future predictions, drop all features that define the label, encoding existing features
        train_features = train_df.drop(columns=['Year','Output Power'])

        zscore_train_labels = (train_labels - np.mean(train_labels)) / np.std(train_labels)

        test_cities = sixyr_datasets.copy()
        test_cities.remove(train_city)
        for city in test_cities:
            test_df = preprocessed_data[city]
            test_labels = test_df['Output Power'].to_numpy()
            # Normalize Output per city, as each city has a different "max" Output Power scale, using Z-score normalization
            zscore_test_labels = (test_labels - np.mean(test_labels)) / np.std(test_labels)
            test_features = test_df.drop(columns=['Year', 'Output Power'])
            # Find mean of observed test values for Normalized RMSE
            mean_test_label = np.mean(test_labels)

            # Linear Regression
            linreg_predictions = train_and_predict_linreg(train_features, train_labels, test_features)
            norm_linreg_prediction = train_and_predict_linreg(train_features, zscore_train_labels, test_features) * np.std(test_labels) + np.mean(test_labels)
            print(f"\n\n{train_city} to {city}")
            print("\nLinear Regression")
            print(f"RMSE:\t{root_mean_squared_error(test_labels, linreg_predictions)}")
            print(f"NRMSE:\t{root_mean_squared_error(test_labels, linreg_predictions) / mean_test_label}")
            print(f"MAE:\t{mean_absolute_error(test_labels, linreg_predictions)}")

            print("\nNormalized Linear Regression")
            print(f"RMSE:\t{root_mean_squared_error(test_labels, norm_linreg_prediction)}")
            print(f"NRMSE:\t{root_mean_squared_error(test_labels, norm_linreg_prediction) / mean_test_label}")
            print(f"MAE:\t{mean_absolute_error(test_labels, norm_linreg_prediction)}")

            # MLP Regressor
            mlpr_predictions = train_and_predict_mlpr(train_features, train_labels, test_features, seed=seed)
            norm_mlpr_predictions = train_and_predict_mlpr(train_features, zscore_train_labels, test_features, seed=seed) * np.std(test_labels) + np.mean(test_labels)
            print("\nMLP Regressor")
            print(f"RMSE:\t{root_mean_squared_error(test_labels, mlpr_predictions)}")
            print(f"NRMSE:\t{root_mean_squared_error(test_labels, mlpr_predictions) / mean_test_label}")
            print(f"MAE:\t{mean_absolute_error(test_labels, mlpr_predictions)}")

            print("\nNormalized MLP Regessor")
            print(f"RMSE:\t{root_mean_squared_error(test_labels, norm_mlpr_predictions)}")
            print(f"NRMSE:\t{root_mean_squared_error(test_labels, norm_mlpr_predictions) / mean_test_label}")
            print(f"MAE:\t{mean_absolute_error(test_labels, norm_mlpr_predictions)}")

def problem2_sequence_variant(preprocessed_data:dict, seed:int):
    # Use Vanilla RNN and compare to LSTM and GRU variants, use transformer for deep learning
    # "build sliding windows of the last K=12 steps to predict the next step(s)"
    for city in preprocessed_data.keys():
        df = preprocessed_data[city]
        train_df = df.iloc[:int(len(df)*0.80)]
        train_labels = train_df['Output Power'].to_numpy()
        # Drop year - should not be used as a training feature for future predictions, drop all features that define the label, encoding existing features
        train_features = train_df.drop(columns=['Year'])

        test_df = df.iloc[int(len(df)*0.80):]
        test_labels = test_df['Output Power'].to_numpy()
        test_features = test_df.drop(columns=['Year'])

        # Find mean of observed test values for Normalized RMSE
        mean_test_label = np.mean(test_labels)

        K = 12

        # Numeric arrays, scaled with TRAIN-only statistics
        tr = train_features.to_numpy(dtype=np.float32)
        te = test_features.to_numpy(dtype=np.float32)
        mu, sd = tr.mean(axis=0), tr.std(axis=0)
        sd[sd == 0] = 1.0                      # e.g. a Cloud_ column that never occurs in train
        tr, te = (tr - mu) / sd, (te - mu) / sd

        # Scale the target too (train stats only); invert after predicting
        y_mu, y_sd = train_labels.mean(), train_labels.std()
        y_tr = (train_labels - y_mu) / y_sd

        # Train windows: rows i..i+K-1 -> target at row i+K
        X_train = np.ascontiguousarray(sliding_window_view(tr, K, axis=0).transpose(0, 2, 1)[:-1])
        y_train = y_tr[K:]

        # Test windows: prepend the last K train rows as context, so every test row gets a prediction
        te_ctx = np.concatenate([tr[-K:], te])
        X_test = np.ascontiguousarray(sliding_window_view(te_ctx, K, axis=0).transpose(0, 2, 1)[:-1])
        # X_test has len(test_df) windows; window j predicts test row j, so test_labels needs no slicing

        F = tr.shape[1]
        models = {
            "RNN":          RecurrentRegressionModel(F, rnn_type="rnn"),
            "LSTM":         RecurrentRegressionModel(F, rnn_type="lstm"),
            "GRU":          RecurrentRegressionModel(F, rnn_type="gru"),
            "Transformer":  TransformerRegressorModel(F, seq_len=K),
        }

        print(f"\n\n{city}")
        for name, model in models.items():
            scaled_pred = train_and_predict_sequence_reg(model, X_train, y_train, X_test, seed=seed)
            prediction = scaled_pred * y_sd + y_mu          # back to real Output Power units
            rmse = root_mean_squared_error(test_labels, prediction)
            print(f"\n{name}")
            print(f"RMSE:\t{rmse}")
            print(f"NRMSE:\t{rmse / mean_test_label}")
            print(f"MAE:\t{mean_absolute_error(test_labels, prediction)}")

    print()