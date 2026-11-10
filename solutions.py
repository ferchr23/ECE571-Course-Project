import numpy as np
from models import train_and_predict_nb, train_and_predict_knn, train_and_predict_rf, train_and_predict_mlpc
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

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
        train_df = df.iloc[:int(len(df)*0.75)]
        train_labels = train_df['Sky Condition'].to_numpy()
        # Drop year - should not be used as a training feature for future predictions, drop all features that define the label
        train_features = train_df.drop(columns=['Year', 'GHI', 'Clearsky GHI', 'Sky Condition'])
        test_df = df.iloc[int(len(df)*0.75):]
        test_features = test_df.drop(columns=['Year', 'GHI', 'Clearsky GHI', 'Sky Condition'])
        test_labels = test_df['Sky Condition'].to_numpy()

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

        # Drop year - should not be used as a training feature for future predictions, drop all features that define the label, drop Sky Condition labels, encoding existing features
        train_features = train_df.drop(columns=['Year','Output Power', 'Sky Condition'])

        # Use the same boundaries to generate test labels
        test_df = df.iloc[int(len(df)*0.75):]
        test_power = test_df['Output Power'].to_numpy()
        test_labels = np.digitize(test_power, power_tercile_boundaries)
        test_features = test_df.drop(columns=['Year', 'Output Power', 'Sky Condition'])

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