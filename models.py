from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def train_and_predict_nb(train_features, train_labels, test_features):
    sklearn_nb = GaussianNB()
    sklearn_nb.fit(train_features, train_labels)
    sklearn_predictions = sklearn_nb.predict(test_features)
    return sklearn_predictions

def train_and_predict_knn(train_features, train_labels, test_features):
    sklearn_knn = KNeighborsClassifier(n_neighbors=3)
    sklearn_knn.fit(train_features, train_labels)
    sklearn_knn_predictions = sklearn_knn.predict(test_features)
    return sklearn_knn_predictions

def train_and_predict_rf(train_features, train_labels, test_features, seed):
    sklearn_rf = RandomForestClassifier(n_estimators=100, random_state=seed)
    sklearn_rf.fit(train_features, train_labels)
    sklearn_rf_predictions = sklearn_rf.predict(test_features)
    return sklearn_rf_predictions

def train_and_predict_mlpc(train_features, train_labels, test_features, seed):
    sklearn_mlpc = MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', max_iter=500, random_state=seed)
    sklearn_mlpc.fit(train_features, train_labels)
    sklearn_mlpc_predictions = sklearn_mlpc.predict(test_features)
    return sklearn_mlpc_predictions