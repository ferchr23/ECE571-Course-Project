import numpy as np
import pandas as pd
import torch
import json
import os
import matplotlib
import matplotlib.pyplot as plt
from torch import nn
from models import train_and_predict_nb, train_and_predict_knn, train_and_predict_rf, train_and_predict_mlpc, train_and_predict_linreg, train_and_predict_mlpr, train_linreg, train_mlpr, predict_model, train_and_predict_sequence_reg, RecurrentRegressionModel, TransformerRegressorModel, AutoEncoder, train_autoencoder
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, root_mean_squared_error, mean_absolute_error, ConfusionMatrixDisplay
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA, KernelPCA

SKY_CLASSES = ["Clear", "Partly-Cloudy", "Cloudy"]
POWER_CLASSES = np.array(["Low", "Medium", "High"])

def _fit_and_scale(x_tr):
    # One-hot encoded feaures and periodic encoded features do not need scaling, already constained to set ranges
    no_scaler = [c for c in x_tr.columns if c.startswith("Cloud_") or c.endswith(("_sin","_cos"))]

    to_scale = [c for c in x_tr.columns if c not in no_scaler]
    # Reorder columns, so those scaled are first, set to return pandas df
    scaler = ColumnTransformer([("num", StandardScaler(), to_scale)], remainder="passthrough", verbose_feature_names_out=False)
    scaler.set_output(transform="pandas")
    X_tr = scaler.fit_transform(x_tr)
    return X_tr, scaler

def _save_confusion_matrix(y, y_hat, classes, out_dir, problem, city, model, seed):
    """Plot a confusion matrix and save it as a PNG, returning its path"""
    os.makedirs(out_dir, exist_ok=True)
    file_name = f"{problem}_{city}_{model}_seed({seed}).png".replace(" ", "-")
    path = os.path.join(out_dir,file_name)

    confmat = ConfusionMatrixDisplay.from_predictions(y, y_hat, labels=classes, cmap="Blues", colorbar=False)
    confmat.ax_.set_title(f"Problem {problem}: {model} - {city} (seed {seed})")
    confmat.figure_.tight_layout()
    confmat.figure_.savefig(path, dpi=150)
    plt.close(confmat.figure_)

    return path

def _save_json(data, out_dir, problem, city, seed):
    os.makedirs(out_dir, exist_ok=True)
    json_path = f"{problem}_{city}_seed{seed}_results.json".replace(" ", "-")
    path = os.path.join(out_dir, json_path)

    json_str = json.dumps(data, indent=4)
    with open(path, "w") as f:
        f.write(json_str)
    
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
        df = preprocessed_data[city].copy()
        save_dir = f"results/Problem1_1/{city}"

        # Create Clear-Sky Index labels
        df["Clear-Sky Index"] = df["GHI"] / df["Clearsky GHI"]
        df["Sky Condition"] = pd.cut(df["Clear-Sky Index"], bins=[-float("inf"), 0.4, 0.85, float("inf")], labels=["Cloudy", "Partly-Cloudy", "Clear"], right=False)

        train_df = df.iloc[:int(len(df)*0.75)]
        Y_tr = train_df['Sky Condition'].to_numpy()
        # Drop year - should not be used as a training feature for future predictions, drop all features that define the label
        train_features = train_df.drop(columns=['Year', 'GHI', 'Clearsky GHI', 'Sky Condition', "Clear-Sky Index"])
        test_df = df.iloc[int(len(df)*0.75):]
        test_features = test_df.drop(columns=['Year', 'GHI', 'Clearsky GHI', 'Sky Condition', "Clear-Sky Index"])
        Y_te = test_df['Sky Condition'].to_numpy()

        X_tr, scaler = _fit_and_scale(train_features)
        X_te = scaler.transform(test_features)

        # Drop cloud encoding features for Naive Bayes, as creating guassian distributions for categorical features is not appropriate. Keep for other models 
        # Naive Bayes
        nb_predictions = train_and_predict_nb(X_tr.filter(cond=~(X_tr.columns.str.startswith("Cloud_")), axis=1), Y_tr, X_te.filter(cond=~(X_te.columns.str.startswith("Cloud_")), axis=1))

        classes = SKY_CLASSES

        nb_results = {
            "model": "Naive Bayes",
            "balanced accuracy": balanced_accuracy_score(Y_te, nb_predictions),
            "confusion matrix": _save_confusion_matrix(Y_te, nb_predictions, classes, save_dir, "1.1", city, "NaiveBayes", seed)            
        }

        print("\n\nProcessing City: ", city)
        # print("Naive Bayes Balanced Accuracy: ",balanced_accuracy_score(Y_te, nb_predictions))
        # print("Naive Bayes Confusion Matrix:\n",confusion_matrix(Y_te, nb_predictions, labels=classes))

        # KNN
        knn_predictions = train_and_predict_knn(X_tr, Y_tr, X_te)

        knn_results = {
            "model": "KNN",
            "balanced accuracy": balanced_accuracy_score(Y_te, knn_predictions),
            "confusion matrix": _save_confusion_matrix(Y_te, knn_predictions, classes, save_dir, "1.1", city, "KNN", seed)            
        }

        # print("KNN Balanced Accuracy: ",balanced_accuracy_score(Y_te, knn_predictions))
        # print("KNN Confusion Matrix:\n",confusion_matrix(Y_te, knn_predictions, labels=classes))

        # Random Forest
        rf_predictions = train_and_predict_rf(X_tr, Y_tr, X_te, seed=seed)

        rf_results = {
            "model": "Random Forest",
            "balanced accuracy": balanced_accuracy_score(Y_te, rf_predictions),
            "confusion matrix": _save_confusion_matrix(Y_te, rf_predictions, classes, save_dir, "1.1", city, "RandomForest", seed)            
        }

        # print("Random Forest Balanced Accuracy: ",balanced_accuracy_score(Y_te, rf_predictions))
        # print("Random Forest Confusion Matrix:\n",confusion_matrix(Y_te, rf_predictions, labels=classes))

        # MLP Classifier
        mlpc_predictions = train_and_predict_mlpc(X_tr, Y_tr, X_te, seed=seed)

        mlp_results = {
            "model": "Multi Layer Perceptron",
            "balanced accuracy": balanced_accuracy_score(Y_te, mlpc_predictions),
            "confusion matrix": _save_confusion_matrix(Y_te, mlpc_predictions, classes, save_dir, "1.1", city, "MLPClassifier", seed)            
        }

        # print("MLP Balanced Accuracy: ",balanced_accuracy_score(Y_te, mlpc_predictions))
        # print("MLP Confusion Matrix:\n",confusion_matrix(Y_te, mlpc_predictions, labels=classes))

        data = {
            "problem": "1",
            "part": "Sky-condition Classifier",
            "city": city,
            "results": [nb_results, knn_results, rf_results, mlp_results]
        }
        _save_json(data, save_dir, "1.1", city, seed)

def problem1_2(preprocessed_data:dict, seed:int):
    # Generation regime classification - Labeling Output Power as low / medium / high
    # Same-city classification: chronological 75/25 split in each city
    for city in preprocessed_data.keys():
        # load preprocessed data for city, split into train/test sets, removing unapplicable features
        df = preprocessed_data[city]
        save_dir = f"results/Problem1_2/{city}"
        train_df = df.iloc[:int(len(df)*0.75)]

        # Split the training data into terciles, 3 equally sized populations, based on output power
        train_power = train_df['Output Power'].to_numpy()
        power_tercile_boundaries = np.quantile(train_power, [1/3, 2/3])
        # Use the boundaries to label each sample as 0 (low), 1 (medium), or 2 (high)
        Y_tr = POWER_CLASSES[np.digitize(train_power, power_tercile_boundaries)]

        # Drop year - should not be used as a training feature for future predictions, drop all features that define the label, encoding existing features
        train_features = train_df.drop(columns=['Year','Output Power'])

        # Use the same boundaries to generate test labels
        test_df = df.iloc[int(len(df)*0.75):]
        test_power = test_df['Output Power'].to_numpy()
        Y_te = POWER_CLASSES[np.digitize(test_power, power_tercile_boundaries)]
        test_features = test_df.drop(columns=['Year', 'Output Power'])

        X_tr, scaler = _fit_and_scale(train_features)
        X_te = scaler.transform(test_features)

        # Drop cloud encoding features for Naive Bayes, as creating guassian distributions for categorical features is not appropriate. Keep for other models 
        # Naive Bayes
        nb_predictions = train_and_predict_nb(X_tr.filter(cond=~(X_tr.columns.str.startswith("Cloud_")), axis=1), Y_tr, X_te.filter(cond=~(X_te.columns.str.startswith("Cloud_")), axis=1))

        classes = POWER_CLASSES
        nb_results = {
            "model": "Naive Bayes",
            "balanced accuracy": balanced_accuracy_score(Y_te, nb_predictions),
            "confusion matrix": _save_confusion_matrix(Y_te, nb_predictions, classes, save_dir, "1.2", city, "NaiveBayes", seed)            
        }

        print("\n\nProcessing City: ", city)
        # print("Naive Bayes Balanced Accuracy: ",balanced_accuracy_score(Y_te, nb_predictions))
        # print("Naive Bayes Confusion Matrix:\n",confusion_matrix(Y_te, nb_predictions, labels=classes))

        # KNN
        knn_predictions = train_and_predict_knn(X_tr, Y_tr, X_te)

        knn_results = {
            "model": "KNN",
            "balanced accuracy": balanced_accuracy_score(Y_te, knn_predictions),
            "confusion matrix": _save_confusion_matrix(Y_te, knn_predictions, classes, save_dir, "1.2", city, "KNN", seed)            
        }

        # print("KNN Balanced Accuracy: ",balanced_accuracy_score(Y_te, knn_predictions))
        # print("KNN Confusion Matrix:\n",confusion_matrix(Y_te, knn_predictions, labels=classes))

        # Random Forest
        rf_predictions = train_and_predict_rf(X_tr, Y_tr, X_te, seed=seed)

        rf_results = {
            "model": "Random Forest",
            "balanced accuracy": balanced_accuracy_score(Y_te, rf_predictions),
            "confusion matrix": _save_confusion_matrix(Y_te, rf_predictions, classes, save_dir, "1.2", city, "RandomForest", seed)            
        }

        # print("Random Forest Balanced Accuracy: ",balanced_accuracy_score(Y_te, rf_predictions))
        # print("Random Forest Confusion Matrix:\n",confusion_matrix(Y_te, rf_predictions, labels=classes))

        # MLP Classifier
        mlpc_predictions = train_and_predict_mlpc(X_tr, Y_tr, X_te, seed=seed)

        mlp_results = {
            "model": "Multi Layer Perceptron",
            "balanced accuracy": balanced_accuracy_score(Y_te, mlpc_predictions),
            "confusion matrix": _save_confusion_matrix(Y_te, mlpc_predictions, classes, save_dir, "1.2", city, "MLPClassifier", seed)            
        }

        # print("MLP Balanced Accuracy: ",balanced_accuracy_score(Y_te, mlpc_predictions))
        # print("MLP Confusion Matrix:\n",confusion_matrix(Y_te, mlpc_predictions, labels=classes))

        data = {
            "problem": "1.2",
            "part": "Output Power Classifier",
            "city": city,
            "results": [nb_results, knn_results, rf_results, mlp_results]
        }
        _save_json(data, save_dir, "1.2", city, seed)

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
        save_dir = f"results/Problem2_1/{city}"

        train_df = df.iloc[:int(len(df)*0.80)]
        train_labels = train_df['Output Power'].to_numpy()
        # Drop year - should not be used as a training feature for future predictions, drop all features that define the label, encoding existing features
        train_features = train_df.drop(columns=['Year','Output Power'])

        test_df = df.iloc[int(len(df)*0.80):]
        Y_te = test_df['Output Power'].to_numpy()
        test_features = test_df.drop(columns=['Year', 'Output Power'])

        X_tr, scaler = _fit_and_scale(train_features)
        X_te = scaler.transform(test_features)

        y_scaler = StandardScaler()
        Y_tr = y_scaler.fit_transform(train_labels.reshape(-1, 1)).ravel()

        # Find mean of observed test values for Normalized RMSE
        mean_test_label = np.mean(Y_te)

        # Linear Regression
        scaled_linreg_predictions = train_and_predict_linreg(X_tr, Y_tr, X_te)
        linreg_predictions = y_scaler.inverse_transform(scaled_linreg_predictions.reshape(-1, 1)).ravel()
        print("\n\nProcessing City: ", city)
        # print("\nLinear Regression")
        # print(f"RMSE:\t{root_mean_squared_error(Y_te, linreg_predictions)}")
        # print(f"NRMSE:\t{root_mean_squared_error(Y_te, linreg_predictions) / mean_test_label}")
        # print(f"MAE:\t{mean_absolute_error(Y_te, linreg_predictions)}")

        linreg_results = {
            "model": "Linear Regression",
            "rmse": root_mean_squared_error(Y_te, linreg_predictions),
            "normalized rmse": root_mean_squared_error(Y_te, linreg_predictions) / mean_test_label,
            "mean absolute error": mean_absolute_error(Y_te, linreg_predictions)
        }

        # MLP Regressor
        scaled_mlpr_predictions = train_and_predict_mlpr(X_tr, Y_tr, X_te, seed=seed)
        mlpr_predictions = y_scaler.inverse_transform(scaled_mlpr_predictions.reshape(-1, 1)).ravel()
        # print("\nMLP Regressor")
        # print(f"RMSE:\t{root_mean_squared_error(Y_te, mlpr_predictions)}")
        # print(f"NRMSE:\t{root_mean_squared_error(Y_te, mlpr_predictions) / mean_test_label}")
        # print(f"MAE:\t{mean_absolute_error(Y_te, mlpr_predictions)}")

        mlp_results = {
            "model": "Multi Layer Perceptron Regressor",
            "rmse": root_mean_squared_error(Y_te, mlpr_predictions),
            "normalized rmse": root_mean_squared_error(Y_te, mlpr_predictions) / mean_test_label,
            "mean absolute error": mean_absolute_error(Y_te, mlpr_predictions)
        }

        data = {
            "problem": "2.1",
            "part": "Same-city Regression",
            "city": city,
            "results": [linreg_results, mlp_results]
        }
        _save_json(data, save_dir, "2.1", city, seed)
        

def problem2_cross_city(preprocessed_data:dict, seed:int):
    """Train on complete Davis (6yr) dataset, test on Huron, Santa Barbara, and La Jolla."""
    sixyr_datasets = ['davis6','huron6','sntbarb6','lajolla6']
    for train_city in sixyr_datasets:
        train_df = preprocessed_data[train_city]

        train_labels = train_df['Output Power'].to_numpy()
        # Drop year - should not be used as a training feature for future predictions, drop all features that define the label, encoding existing features
        train_features = train_df.drop(columns=['Year','Output Power'])

        X_tr, scaler = _fit_and_scale(train_features)
        
        # Normalize labels by maximum Output Power. This is an "oracle-scale" change. Otherwise residuals are dominated by scale differences between cities, likely a result of the size of their solar grid installation. I should report the effect of this change, saving output both with and without the normalization.
        train_labels = train_labels / np.max(train_labels)
        
        y_scaler = StandardScaler()
        Y_tr = y_scaler.fit_transform(train_labels.reshape(-1, 1)).ravel()

        # Different setup here compared to other problems. Model is trained once on a single city and then tested three times. Split this up for efficiency
        linreg_model = train_linreg(X_tr, Y_tr)
        mlpr_model = train_mlpr(X_tr, Y_tr, seed=seed)

        test_cities = sixyr_datasets.copy()
        test_cities.remove(train_city)
        for city in test_cities:
            test_df = preprocessed_data[city]
            save_dir = f"results/Problem2_2/{train_city}"
            Y_te = test_df['Output Power'].to_numpy()
            test_features = test_df.drop(columns=['Year', 'Output Power'])
            # Find mean of observed test values for Normalized RMSE
            mean_test_label = np.mean(Y_te)

            X_te = scaler.transform(test_features)

            # Linear Regression
            scaled_linreg_predictions = predict_model(linreg_model, X_te)
            linreg_predictions = y_scaler.inverse_transform(scaled_linreg_predictions.reshape(-1, 1)).ravel()

            # Undo normalization using test city max power - this is the part that requires Oracle knowledge
            linreg_predictions = linreg_predictions * np.max(Y_te)
            print(f"\n\nProcessing {train_city} to {city}")
            # print("\nLinear Regression")
            # print(f"RMSE:\t{root_mean_squared_error(Y_te, linreg_predictions)}")
            # print(f"NRMSE:\t{root_mean_squared_error(Y_te, linreg_predictions) / mean_test_label}")
            # print(f"MAE:\t{mean_absolute_error(Y_te, linreg_predictions)}")

            linreg_results = {
                "model": "Linear Regression",
                "rmse": root_mean_squared_error(Y_te, linreg_predictions),
                "normalized rmse": root_mean_squared_error(Y_te, linreg_predictions) / mean_test_label,
                "mean absolute error": mean_absolute_error(Y_te, linreg_predictions)
            }

            # MLP Regressor
            scaled_mlpr_predictions = predict_model(mlpr_model, X_te)
            mlpr_predictions = y_scaler.inverse_transform(scaled_mlpr_predictions.reshape(-1, 1)).ravel()
            # Undo normalization using test city max power - this is the part that requires Oracle knowledge
            mlpr_predictions = mlpr_predictions * np.max(Y_te)
            # print("\nMLP Regressor")
            # print(f"RMSE:\t{root_mean_squared_error(Y_te, mlpr_predictions)}")
            # print(f"NRMSE:\t{root_mean_squared_error(Y_te, mlpr_predictions) / mean_test_label}")
            # print(f"MAE:\t{mean_absolute_error(Y_te, mlpr_predictions)}")

            mlp_results = {
                "model": "Multi Layer Perceptron Regressor",
                "rmse": root_mean_squared_error(Y_te, mlpr_predictions),
                "normalized rmse": root_mean_squared_error(Y_te, mlpr_predictions) / mean_test_label,
                "mean absolute error": mean_absolute_error(Y_te, mlpr_predictions)
            }

            data = {
                "problem": "2.2",
                "part": "Cross-city Regression",
                "training city": train_city,
                "target city": city,
                "results": [linreg_results, mlp_results]
            }
            _save_json(data, save_dir, "2.2", f"{train_city}-{city}", seed)

def problem2_sequence_variant(preprocessed_data:dict, seed:int):
    # Use Vanilla RNN and compare to LSTM and GRU variants, use transformer for deep learning
    # "build sliding windows of the last K=12 steps to predict the next step(s)"
    for city in preprocessed_data.keys():
        df = preprocessed_data[city]
        save_dir = f"results/Problem2_3/{city}"

        train_df = df.iloc[:int(len(df)*0.80)]
        train_labels = train_df['Output Power'].to_numpy()
        # Drop year - should not be used as a training feature for future predictions, drop all features that define the label, encoding existing features
        train_features = train_df.drop(columns=['Year'])

        test_df = df.iloc[int(len(df)*0.80):]
        Y_te = test_df['Output Power'].to_numpy()
        test_features = test_df.drop(columns=['Year'])

        # Find mean of observed test values for Normalized RMSE
        mean_test_label = np.mean(Y_te)

        X_tr, scaler = _fit_and_scale(train_features)
        X_tr = X_tr.to_numpy(dtype=np.float32)
        X_te = scaler.transform(test_features)
        y_scaler = StandardScaler()
        Y_tr = y_scaler.fit_transform(train_labels.reshape(-1, 1)).ravel()

        # Size of windows
        K = 12
                
        # Train windows: rows i..i+K-1 -> target at row i+K
        X_train = np.ascontiguousarray(sliding_window_view(X_tr, K, axis=0).transpose(0, 2, 1)[:-1])
        y_train = Y_tr[K:]

        # Test windows: prepend the last K train rows as context, so every test row gets a prediction
        te_ctx = np.concatenate([X_tr[-K:], X_te])
        X_test = np.ascontiguousarray(sliding_window_view(te_ctx, K, axis=0).transpose(0, 2, 1)[:-1])
        # X_test has len(test_df) windows; window j predicts test row j, so test_labels needs no slicing

        F = X_tr.shape[1]

        # Make a list of all sequential models, but wait to build them so the seed can be set each time
        builders = {
            "RNN":          lambda: RecurrentRegressionModel(F, rnn_type="rnn"),
            "LSTM":         lambda: RecurrentRegressionModel(F, rnn_type="lstm"),
            "GRU":          lambda: RecurrentRegressionModel(F, rnn_type="gru"),
            "Transformer":  lambda: TransformerRegressorModel(F, seq_len=K),
        }

        model_results = []
        # Iterate through all the regression models
        print(f"\n\n{city}")
        for name, build in builders.items():
            torch.manual_seed(seed)
            model = build()
            scaled_pred = train_and_predict_sequence_reg(model, X_train, y_train, X_test, seed=seed)
            prediction = y_scaler.inverse_transform(scaled_pred.reshape(-1, 1)).ravel()          # back to real Output Power units
            rmse = root_mean_squared_error(Y_te, prediction)
            # print(f"\n{name}")
            # print(f"RMSE:\t{rmse}")
            # print(f"NRMSE:\t{rmse / mean_test_label}")
            # print(f"MAE:\t{mean_absolute_error(Y_te, prediction)}")

            results = {
                "model": name,
                "rmse": rmse,
                "normalized rmse": rmse / mean_test_label,
                "mean absolute error": mean_absolute_error(Y_te, prediction)
            }
            model_results.append(results)

            data = {
                "problem": "2.3",
                "part": "Sequence Variant Regression",
                "city": city,
                "results": model_results
            }
            _save_json(data, save_dir, "2.3", city, seed)

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

# Use PCA and/or Kernel PCA to reduce the dimensionality, then use an Autoencoder as the deep learning option

def problem3_1_PCA(preprocessed_data:dict, d, seed:int):
    # For Problem 1_1
    for city in preprocessed_data.keys():
        # load preprocessed data for city, split into train/test sets, removing unapplicable features
        df = preprocessed_data[city].copy()

        # Create Clear-Sky Index labels
        df["Clear-Sky Index"] = df["GHI"] / df["Clearsky GHI"]
        df["Sky Condition"] = pd.cut(df["Clear-Sky Index"], bins=[-float("inf"), 0.4, 0.85, float("inf")], labels=["Cloudy", "Partly-Cloudy", "Clear"], right=False)

        train_df = df.iloc[:int(len(df)*0.75)]
        Y_tr = train_df['Sky Condition'].to_numpy()
        # Drop year - should not be used as a training feature for future predictions, drop all features that define the label
        train_features = train_df.drop(columns=['Year', 'GHI', 'Clearsky GHI', 'Sky Condition', "Clear-Sky Index"])
        test_df = df.iloc[int(len(df)*0.75):]
        test_features = test_df.drop(columns=['Year', 'GHI', 'Clearsky GHI', 'Sky Condition', "Clear-Sky Index"])
        Y_te = test_df['Sky Condition'].to_numpy()
        
        # One-hot encoded feaures and periodic encoded features do not need scaling, already constained to set ranges
        no_scaler = [c for c in train_features.columns if c.startswith("Cloud_") or c.endswith(("_sin","_cos"))]
        to_scale = [c for c in train_features.columns if c not in no_scaler]
        # Reorder columns, so those scaled are first, set to return pandas df
        scaler = ColumnTransformer([("num", StandardScaler(), to_scale)], remainder="passthrough", verbose_feature_names_out=False)
        scaler.set_output(transform="pandas")
        X_tr = scaler.fit_transform(train_features)
        X_te = scaler.transform(test_features)

        # --------------------- PCA Here ----------------
        dimension_reduction = PCA(n_components=d)
        Z_tr = dimension_reduction.fit_transform(X_tr)
        Z_te = dimension_reduction.transform(X_te)

        X_hat = dimension_reduction.inverse_transform(Z_te)
        reconstruction_mse = np.mean((X_te.to_numpy() - X_hat) **2)
        print("Reconstruction Error from AutoEncoder: ", reconstruction_mse)

        print("Explained Variance Ratio PCA: ", dimension_reduction.explained_variance_ratio_)

        # No columns to drop!
        nb_predictions = train_and_predict_nb(Z_tr, Y_tr, Z_te)

        classes = SKY_CLASSES
        print("\n\nCity: ", city)
        print("Naive Bayes Balanced Accuracy: ",balanced_accuracy_score(Y_te, nb_predictions))
        print("Naive Bayes Confusion Matrix:\n",confusion_matrix(Y_te, nb_predictions, labels=classes))

        # KNN
        knn_predictions = train_and_predict_knn(Z_tr, Y_tr, Z_te)

        print("KNN Balanced Accuracy: ",balanced_accuracy_score(Y_te, knn_predictions))
        print("KNN Confusion Matrix:\n",confusion_matrix(Y_te, knn_predictions, labels=classes))

        # Random Forest
        rf_predictions = train_and_predict_rf(Z_tr, Y_tr, Z_te, seed=seed)

        print("Random Forest Balanced Accuracy: ",balanced_accuracy_score(Y_te, rf_predictions))
        print("Random Forest Confusion Matrix:\n",confusion_matrix(Y_te, rf_predictions, labels=classes))

        # MLP Classifier
        mlpc_predictions = train_and_predict_mlpc(Z_tr, Y_tr, Z_te, seed=seed)

        print("MLP Balanced Accuracy: ",balanced_accuracy_score(Y_te, mlpc_predictions))
        print("MLP Confusion Matrix:\n",confusion_matrix(Y_te, mlpc_predictions, labels=classes))


    # For Problem 1_2
    for city in preprocessed_data.keys():
        # load preprocessed data for city, split into train/test sets, removing unapplicable features
        df = preprocessed_data[city]
        train_df = df.iloc[:int(len(df)*0.75)]

        # Split the training data into terciles, 3 equally sized populations, based on output power
        train_power = train_df['Output Power'].to_numpy()
        power_tercile_boundaries = np.quantile(train_power, [1/3, 2/3])
        # Use the boundaries to label each sample as 0 (low), 1 (medium), or 2 (high)
        Y_tr = POWER_CLASSES[np.digitize(train_power, power_tercile_boundaries)]

        # Drop year - should not be used as a training feature for future predictions, drop all features that define the label, encoding existing features
        train_features = train_df.drop(columns=['Year','Output Power'])

        # Use the same boundaries to generate test labels
        test_df = df.iloc[int(len(df)*0.75):]
        test_power = test_df['Output Power'].to_numpy()
        Y_te = POWER_CLASSES[np.digitize(test_power, power_tercile_boundaries)]
        test_features = test_df.drop(columns=['Year', 'Output Power'])

        # One-hot encoded feaures and periodic encoded features do not need scaling, already constained to set ranges
        no_scaler = [c for c in train_features.columns if c.startswith("Cloud_") or c.endswith(("_sin","_cos"))]
        to_scale = [c for c in train_features.columns if c not in no_scaler]
        # Reorder columns, so those scaled are first, set to return pandas df
        scaler = ColumnTransformer([("num", StandardScaler(), to_scale)], remainder="passthrough", verbose_feature_names_out=False)
        scaler.set_output(transform="pandas")
        X_tr = scaler.fit_transform(train_features)
        X_te = scaler.transform(test_features)

        # --------------------- PCA Here ----------------
        dimension_reduction = PCA(n_components=d)
        Z_tr = dimension_reduction.fit_transform(X_tr)
        Z_te = dimension_reduction.transform(X_te)

        X_hat = dimension_reduction.inverse_transform(Z_te)
        reconstruction_mse = np.mean((X_te.to_numpy() - X_hat) **2)
        print("Reconstruction Error from AutoEncoder: ", reconstruction_mse)

        print("Explained Variance Ratio PCA: ", dimension_reduction.explained_variance_ratio_)

        # No columns to drop!
        nb_predictions = train_and_predict_nb(Z_tr, Y_tr, Z_te)

        classes = POWER_CLASSES
        print("\n\nCity: ", city)
        print("Naive Bayes Balanced Accuracy: ",balanced_accuracy_score(Y_te, nb_predictions))
        print("Naive Bayes Confusion Matrix:\n",confusion_matrix(Y_te, nb_predictions, labels=classes))

        # KNN
        knn_predictions = train_and_predict_knn(Z_tr, Y_tr, Z_te)

        print("KNN Balanced Accuracy: ",balanced_accuracy_score(Y_te, knn_predictions))
        print("KNN Confusion Matrix:\n",confusion_matrix(Y_te, knn_predictions, labels=classes))

        # Random Forest
        rf_predictions = train_and_predict_rf(Z_tr, Y_tr, Z_te, seed=seed)

        print("Random Forest Balanced Accuracy: ",balanced_accuracy_score(Y_te, rf_predictions))
        print("Random Forest Confusion Matrix:\n",confusion_matrix(Y_te, rf_predictions, labels=classes))

        # MLP Classifier
        mlpc_predictions = train_and_predict_mlpc(Z_tr, Y_tr, Z_te, seed=seed)

        print("MLP Balanced Accuracy: ",balanced_accuracy_score(Y_te, mlpc_predictions))
        print("MLP Confusion Matrix:\n",confusion_matrix(Y_te, mlpc_predictions, labels=classes))

# Now use Kernel PCA, Yay!
def problem3_1_KPCA(preprocessed_data:dict, d, seed:int):
    # For Problem 1_1
    for city in preprocessed_data.keys():
        # load preprocessed data for city, split into train/test sets, removing unapplicable features
        df = preprocessed_data[city].copy()

        # Create Clear-Sky Index labels
        df["Clear-Sky Index"] = df["GHI"] / df["Clearsky GHI"]
        df["Sky Condition"] = pd.cut(df["Clear-Sky Index"], bins=[-float("inf"), 0.4, 0.85, float("inf")], labels=["Cloudy", "Partly-Cloudy", "Clear"], right=False)

        train_df = df.iloc[:int(len(df)*0.75)]
        Y_tr = train_df['Sky Condition'].to_numpy()
        # Drop year - should not be used as a training feature for future predictions, drop all features that define the label
        train_features = train_df.drop(columns=['Year', 'GHI', 'Clearsky GHI', 'Sky Condition', "Clear-Sky Index"])
        test_df = df.iloc[int(len(df)*0.75):]
        test_features = test_df.drop(columns=['Year', 'GHI', 'Clearsky GHI', 'Sky Condition', "Clear-Sky Index"])
        Y_te = test_df['Sky Condition'].to_numpy()
        
        # One-hot encoded feaures and periodic encoded features do not need scaling, already constained to set ranges
        no_scaler = [c for c in train_features.columns if c.startswith("Cloud_") or c.endswith(("_sin","_cos"))]
        to_scale = [c for c in train_features.columns if c not in no_scaler]
        # Reorder columns, so those scaled are first, set to return pandas df
        scaler = ColumnTransformer([("num", StandardScaler(), to_scale)], remainder="passthrough", verbose_feature_names_out=False)
        scaler.set_output(transform="pandas")
        X_tr = scaler.fit_transform(train_features)
        X_te = scaler.transform(test_features)

        # -------------- KPCA Here -------------------
        # Use rbf kernel, usually the best default option, simulates projecting up to an infinite dimension space
        # gamma is taken as default, 1/f for f number of features
        dimension_reduction = KernelPCA(n_components=d, kernel="rbf", gamma=None, fit_inverse_transform=True, eigen_solver="randomized", random_state=seed)
        
        # KPCA creates a n x n kernel matrix when fitting, so with the larger data sets that will take up multiple GB of memory
        # I ran out of memory. So we use a random sample of the training data, at maximum 3000 samples
        rng = np.random.default_rng(seed=seed)
        fit_idx = rng.choice(len(X_tr), size=min(3000, len(X_tr)), replace=False)
        dimension_reduction.fit(X_tr.iloc[fit_idx])

        # Now transform all training samples and test samples!
        Z_tr = dimension_reduction.transform(X_tr) 
        Z_te = dimension_reduction.transform(X_te)

        X_hat = dimension_reduction.inverse_transform(Z_te)
        reconstruction_mse = np.mean((X_te.to_numpy() - X_hat) **2)
        print("Reconstruction Error from AutoEncoder: ", reconstruction_mse)

        # Explained variance stand in for KPCA
        exp_var = dimension_reduction.eigenvalues_ / dimension_reduction.eigenvalues_.sum()
        print("Explained Variance in Kernel's Feature Space: ", exp_var)

        # No columns to drop!
        nb_predictions = train_and_predict_nb(Z_tr, Y_tr, Z_te)

        classes = SKY_CLASSES
        print("\n\nCity: ", city)
        print("Naive Bayes Balanced Accuracy: ",balanced_accuracy_score(Y_te, nb_predictions))
        print("Naive Bayes Confusion Matrix:\n",confusion_matrix(Y_te, nb_predictions, labels=classes))

        # KNN
        knn_predictions = train_and_predict_knn(Z_tr, Y_tr, Z_te)

        print("KNN Balanced Accuracy: ",balanced_accuracy_score(Y_te, knn_predictions))
        print("KNN Confusion Matrix:\n",confusion_matrix(Y_te, knn_predictions, labels=classes))

        # Random Forest
        rf_predictions = train_and_predict_rf(Z_tr, Y_tr, Z_te, seed=seed)

        print("Random Forest Balanced Accuracy: ",balanced_accuracy_score(Y_te, rf_predictions))
        print("Random Forest Confusion Matrix:\n",confusion_matrix(Y_te, rf_predictions, labels=classes))

        # MLP Classifier
        mlpc_predictions = train_and_predict_mlpc(Z_tr, Y_tr, Z_te, seed=seed)

        print("MLP Balanced Accuracy: ",balanced_accuracy_score(Y_te, mlpc_predictions))
        print("MLP Confusion Matrix:\n",confusion_matrix(Y_te, mlpc_predictions, labels=classes))


    # For Problem 1_2
    for city in preprocessed_data.keys():
        # load preprocessed data for city, split into train/test sets, removing unapplicable features
        df = preprocessed_data[city]
        train_df = df.iloc[:int(len(df)*0.75)]

        # Split the training data into terciles, 3 equally sized populations, based on output power
        train_power = train_df['Output Power'].to_numpy()
        power_tercile_boundaries = np.quantile(train_power, [1/3, 2/3])
        # Use the boundaries to label each sample as 0 (low), 1 (medium), or 2 (high)
        Y_tr = POWER_CLASSES[np.digitize(train_power, power_tercile_boundaries)]

        # Drop year - should not be used as a training feature for future predictions, drop all features that define the label, encoding existing features
        train_features = train_df.drop(columns=['Year','Output Power'])

        # Use the same boundaries to generate test labels
        test_df = df.iloc[int(len(df)*0.75):]
        test_power = test_df['Output Power'].to_numpy()
        Y_te = POWER_CLASSES[np.digitize(test_power, power_tercile_boundaries)]
        test_features = test_df.drop(columns=['Year', 'Output Power'])

        # One-hot encoded feaures and periodic encoded features do not need scaling, already constained to set ranges
        no_scaler = [c for c in train_features.columns if c.startswith("Cloud_") or c.endswith(("_sin","_cos"))]
        to_scale = [c for c in train_features.columns if c not in no_scaler]
        # Reorder columns, so those scaled are first, set to return pandas df
        scaler = ColumnTransformer([("num", StandardScaler(), to_scale)], remainder="passthrough", verbose_feature_names_out=False)
        scaler.set_output(transform="pandas")
        X_tr = scaler.fit_transform(train_features)
        X_te = scaler.transform(test_features)

        # -------------- KPCA Here -------------------
        # Use rbf kernel, usually the best default option, simulates projecting up to an infinite dimension space
        # gamma is taken as default, 1/f for f number of features
        dimension_reduction = KernelPCA(n_components=d, kernel="rbf", gamma=None, fit_inverse_transform=True, eigen_solver="randomized", random_state=seed)
        
        # KPCA creates a n x n kernel matrix when fitting, so with the larger data sets that will take up multiple GB of memory
        # I ran out of memory. So we use a random sample of the training data, at maximum 3000 samples
        rng = np.random.default_rng(seed=seed)
        fit_idx = rng.choice(len(X_tr), size=min(3000, len(X_tr)), replace=False)
        dimension_reduction.fit(X_tr.iloc[fit_idx])

        # Now transform all training samples and test samples!
        Z_tr = dimension_reduction.transform(X_tr) 
        Z_te = dimension_reduction.transform(X_te)

        X_hat = dimension_reduction.inverse_transform(Z_te)
        reconstruction_mse = np.mean((X_te.to_numpy() - X_hat) **2)
        print("Reconstruction Error from AutoEncoder: ", reconstruction_mse)

        # Explained variance stand in for KPCA
        exp_var = dimension_reduction.eigenvalues_ / dimension_reduction.eigenvalues_.sum()
        print("Explained Variance in Kernel's Feature Space: ", exp_var)

        # No columns to drop!
        nb_predictions = train_and_predict_nb(Z_tr, Y_tr, Z_te)

        classes = POWER_CLASSES
        print("\n\nCity: ", city)
        print("Naive Bayes Balanced Accuracy: ",balanced_accuracy_score(Y_te, nb_predictions))
        print("Naive Bayes Confusion Matrix:\n",confusion_matrix(Y_te, nb_predictions, labels=classes))

        # KNN
        knn_predictions = train_and_predict_knn(Z_tr, Y_tr, Z_te)

        print("KNN Balanced Accuracy: ",balanced_accuracy_score(Y_te, knn_predictions))
        print("KNN Confusion Matrix:\n",confusion_matrix(Y_te, knn_predictions, labels=classes))

        # Random Forest
        rf_predictions = train_and_predict_rf(Z_tr, Y_tr, Z_te, seed=seed)

        print("Random Forest Balanced Accuracy: ",balanced_accuracy_score(Y_te, rf_predictions))
        print("Random Forest Confusion Matrix:\n",confusion_matrix(Y_te, rf_predictions, labels=classes))

        # MLP Classifier
        mlpc_predictions = train_and_predict_mlpc(Z_tr, Y_tr, Z_te, seed=seed)

        print("MLP Balanced Accuracy: ",balanced_accuracy_score(Y_te, mlpc_predictions))
        print("MLP Confusion Matrix:\n",confusion_matrix(Y_te, mlpc_predictions, labels=classes))


def problem3_1_AutoEn(preprocessed_data:dict, d, seed:int):
    # For Problem 1_1
    for city in preprocessed_data.keys():
        # load preprocessed data for city, split into train/test sets, removing unapplicable features
        df = preprocessed_data[city].copy()

        # Create Clear-Sky Index labels
        df["Clear-Sky Index"] = df["GHI"] / df["Clearsky GHI"]
        df["Sky Condition"] = pd.cut(df["Clear-Sky Index"], bins=[-float("inf"), 0.4, 0.85, float("inf")], labels=["Cloudy", "Partly-Cloudy", "Clear"], right=False)

        train_df = df.iloc[:int(len(df)*0.75)]
        Y_tr = train_df['Sky Condition'].to_numpy()
        # Drop year - should not be used as a training feature for future predictions, drop all features that define the label
        train_features = train_df.drop(columns=['Year', 'GHI', 'Clearsky GHI', 'Sky Condition', "Clear-Sky Index"])
        test_df = df.iloc[int(len(df)*0.75):]
        test_features = test_df.drop(columns=['Year', 'GHI', 'Clearsky GHI', 'Sky Condition', "Clear-Sky Index"])
        Y_te = test_df['Sky Condition'].to_numpy()
        
        # One-hot encoded feaures and periodic encoded features do not need scaling, already constained to set ranges
        no_scaler = [c for c in train_features.columns if c.startswith("Cloud_") or c.endswith(("_sin","_cos"))]
        to_scale = [c for c in train_features.columns if c not in no_scaler]
        # Reorder columns, so those scaled are first, set to return pandas df
        scaler = ColumnTransformer([("num", StandardScaler(), to_scale)], remainder="passthrough", verbose_feature_names_out=False)
        scaler.set_output(transform="pandas")
        X_tr = scaler.fit_transform(train_features)
        X_te = scaler.transform(test_features)

        # -------------- Autoencoder Here -------------------
        # Create and train autoencoder for chosen dimension transformation
        torch.manual_seed(seed)
        auto_encoder = train_autoencoder(AutoEncoder(X_tr.shape[1], d), X_tr, seed=seed)
        with torch.no_grad():
            t_tr = torch.as_tensor(X_tr.to_numpy(), dtype=torch.float32)
            t_te = torch.as_tensor(X_te.to_numpy(), dtype=torch.float32)
            # Reduce dimension of training and test features
            Z_tr = auto_encoder.encode(t_tr).numpy()
            Z_te = auto_encoder.encode(t_te).numpy()
            reconstruction_mse = nn.functional.mse_loss(auto_encoder(t_te), t_te).item()

        print("Reconstruction Error from AutoEncoder: ", reconstruction_mse)

        # No columns to drop!
        nb_predictions = train_and_predict_nb(Z_tr, Y_tr, Z_te)

        classes = SKY_CLASSES
        print("\n\nCity: ", city)
        print("Naive Bayes Balanced Accuracy: ",balanced_accuracy_score(Y_te, nb_predictions))
        print("Naive Bayes Confusion Matrix:\n",confusion_matrix(Y_te, nb_predictions, labels=classes))

        # KNN
        knn_predictions = train_and_predict_knn(Z_tr, Y_tr, Z_te)

        print("KNN Balanced Accuracy: ",balanced_accuracy_score(Y_te, knn_predictions))
        print("KNN Confusion Matrix:\n",confusion_matrix(Y_te, knn_predictions, labels=classes))

        # Random Forest
        rf_predictions = train_and_predict_rf(Z_tr, Y_tr, Z_te, seed=seed)

        print("Random Forest Balanced Accuracy: ",balanced_accuracy_score(Y_te, rf_predictions))
        print("Random Forest Confusion Matrix:\n",confusion_matrix(Y_te, rf_predictions, labels=classes))

        # MLP Classifier
        mlpc_predictions = train_and_predict_mlpc(Z_tr, Y_tr, Z_te, seed=seed)

        print("MLP Balanced Accuracy: ",balanced_accuracy_score(Y_te, mlpc_predictions))
        print("MLP Confusion Matrix:\n",confusion_matrix(Y_te, mlpc_predictions, labels=classes))


    # For Problem 1_2
    for city in preprocessed_data.keys():
        # load preprocessed data for city, split into train/test sets, removing unapplicable features
        df = preprocessed_data[city]
        train_df = df.iloc[:int(len(df)*0.75)]

        # Split the training data into terciles, 3 equally sized populations, based on output power
        train_power = train_df['Output Power'].to_numpy()
        power_tercile_boundaries = np.quantile(train_power, [1/3, 2/3])
        # Use the boundaries to label each sample as 0 (low), 1 (medium), or 2 (high)
        Y_tr = POWER_CLASSES[np.digitize(train_power, power_tercile_boundaries)]

        # Drop year - should not be used as a training feature for future predictions, drop all features that define the label, encoding existing features
        train_features = train_df.drop(columns=['Year','Output Power'])

        # Use the same boundaries to generate test labels
        test_df = df.iloc[int(len(df)*0.75):]
        test_power = test_df['Output Power'].to_numpy()
        Y_te = POWER_CLASSES[np.digitize(test_power, power_tercile_boundaries)]
        test_features = test_df.drop(columns=['Year', 'Output Power'])

        # One-hot encoded feaures and periodic encoded features do not need scaling, already constained to set ranges
        no_scaler = [c for c in train_features.columns if c.startswith("Cloud_") or c.endswith(("_sin","_cos"))]
        to_scale = [c for c in train_features.columns if c not in no_scaler]
        # Reorder columns, so those scaled are first, set to return pandas df
        scaler = ColumnTransformer([("num", StandardScaler(), to_scale)], remainder="passthrough", verbose_feature_names_out=False)
        scaler.set_output(transform="pandas")
        X_tr = scaler.fit_transform(train_features)
        X_te = scaler.transform(test_features)

        # -------------- Autoencoder Here -------------------
        # Create and train autoencoder for chosen dimension transformation
        torch.manual_seed(seed)
        auto_encoder = train_autoencoder(AutoEncoder(X_tr.shape[1], d), X_tr, seed=seed)
        with torch.no_grad():
            t_tr = torch.as_tensor(X_tr.to_numpy(), dtype=torch.float32)
            t_te = torch.as_tensor(X_te.to_numpy(), dtype=torch.float32)
            # Reduce dimension of training and test features
            Z_tr = auto_encoder.encode(t_tr).numpy()
            Z_te = auto_encoder.encode(t_te).numpy()
            reconstruction_mse = nn.functional.mse_loss(auto_encoder(t_te), t_te).item()

        print("Reconstruction Error from AutoEncoder: ", reconstruction_mse)

        # No columns to drop!
        nb_predictions = train_and_predict_nb(Z_tr, Y_tr, Z_te)

        classes = POWER_CLASSES
        print("\n\nCity: ", city)
        print("Naive Bayes Balanced Accuracy: ",balanced_accuracy_score(Y_te, nb_predictions))
        print("Naive Bayes Confusion Matrix:\n",confusion_matrix(Y_te, nb_predictions, labels=classes))

        # KNN
        knn_predictions = train_and_predict_knn(Z_tr, Y_tr, Z_te)

        print("KNN Balanced Accuracy: ",balanced_accuracy_score(Y_te, knn_predictions))
        print("KNN Confusion Matrix:\n",confusion_matrix(Y_te, knn_predictions, labels=classes))

        # Random Forest
        rf_predictions = train_and_predict_rf(Z_tr, Y_tr, Z_te, seed=seed)

        print("Random Forest Balanced Accuracy: ",balanced_accuracy_score(Y_te, rf_predictions))
        print("Random Forest Confusion Matrix:\n",confusion_matrix(Y_te, rf_predictions, labels=classes))

        # MLP Classifier
        mlpc_predictions = train_and_predict_mlpc(Z_tr, Y_tr, Z_te, seed=seed)

        print("MLP Balanced Accuracy: ",balanced_accuracy_score(Y_te, mlpc_predictions))
        print("MLP Confusion Matrix:\n",confusion_matrix(Y_te, mlpc_predictions, labels=classes))

def problem3_2(preprocessed_data:dict, seed:int):
    print()