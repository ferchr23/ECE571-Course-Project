from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import torch 
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

def _to_loader (features, labels, batch_size=64, shuffle=False):
    """Standardize windowed input features and labels to be a tensor dataset for PyTorch RNN models.
    
    args:
        features - features of dataset, shape of (N (number of windows), K (size of window), F (number of features))
        labels - labels of dataset (N (number of windows))
    """
    # Make features and labels tensors, make labels shape (N, 1)
    dataset = TensorDataset(torch.as_tensor(features, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.float32).unsqueeze(-1))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

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
    sklearn_mlpc = MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', max_iter=1000, random_state=seed)
    sklearn_mlpc.fit(train_features, train_labels)
    sklearn_mlpc_predictions = sklearn_mlpc.predict(test_features)
    return sklearn_mlpc_predictions

def train_and_predict_linreg(train_features, train_labels, test_features):
    sklearn_linreg = LinearRegression()
    sklearn_linreg.fit(train_features, train_labels)
    sklearn_linreg_predictions = sklearn_linreg.predict(test_features)
    return sklearn_linreg_predictions

def train_and_predict_mlpr(train_features, train_labels, test_features, seed):
    sklearn_mlpr = MLPRegressor(hidden_layer_sizes=(100,), activation='tanh', max_iter=1000, random_state=seed)
    sklearn_mlpr.fit(train_features, train_labels)
    sklearn_mlpr_predictions = sklearn_mlpr.predict(test_features)
    return sklearn_mlpr_predictions

def train_linreg(x_tr, y_tr):
    sklearn_linreg = LinearRegression()
    sklearn_linreg.fit(x_tr, y_tr)
    return sklearn_linreg

def train_mlpr(x_tr, y_tr, seed):
    sklearn_mlpr = MLPRegressor(hidden_layer_sizes=(100,), activation='tanh', max_iter=1000, random_state=seed)
    sklearn_mlpr.fit(x_tr, y_tr)
    return sklearn_mlpr

def predict_model(model, x_te):
    return model.predict(x_te)
    

class RecurrentRegressionModel(nn.Module):
    def __init__(self, num_feats, rnn_type, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        # If there is only 1 RNN layer, no dropout is needed, as it controls how much info is "saved"
        dropout = dropout if layers > 1 else 0.0
        rnn_type = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[rnn_type]
        # Create rnn layers of chosen type
        self.rnn = rnn_type(num_feats, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        # Create head linear layer to network, so an output can be taken
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        # input X is of shape (B (batches), K (size of window), F (number of features))
        out, _ = self.rnn(x)
        # out is of shape (B, K, hidden)
        # Feed in the last timestamp into the linear head to get output of shape (B, 1)
        return self.head(out[:, -1, :])

class TransformerRegressorModel(nn.Module):
    def __init__(self, num_feats, seq_len=12, d_model=64, nhead=4, layers=2, dropout=0.1):
        """
        
        num_feats - number of features in data
        seq_len - lenth of sequence in context window
        d_model - hidden dimensionality of model
        nhead - number of heads in multiheaded attention

        Positional encoding is used here, only 12 positions it works well
        d_model must be divisable by nhead
        Layer norm is done prior to attention, more reliable for small datasets
        """
        super().__init__()
        # Project the inputs from F (number of features) up to d_model (dimensionality of model)
        self.proj = nn.Linear(num_feats, d_model)
        # Learned positions in the sequence (Here sequence is size of the context window)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        # Making an Encoder only transformer model
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # project input to hidden size, and add learned positions
        h = self.proj(x) + self.pos
        # feed into transformer encoder layers
        h = self.encoder(h)
        # feed last token into the head layer, alternately may try h.mean(dim=1) in the future
        return self.head(h[:, -1, :])

def train_and_predict_sequence_reg(model, train_features, train_labels, test_features, seed, epochs=100, patience=10, lr=1e-3, batch_size=64, val_frac=0.15):
    torch.manual_seed(seed)
    model.to("cpu")

    # Create a chronological tail for validation on the training set
    n_val = int(len(train_features) * val_frac)
    # Shuffling the batches should not affect chronological splits of training and test, also have days encoded as cos / sin, so time can be calculated
    train_dl = _to_loader(train_features[:-n_val],train_labels[:-n_val],batch_size, shuffle=True)
    val_dl = _to_loader(train_features[-n_val:], train_labels[-n_val:],batch_size)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()
    best, best_state, bad = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to("cpu"), yb.to("cpu")
            opt.zero_grad()
            loss = loss_function(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        # Skip gradient tracking during evaluation to save memory
        with torch.no_grad():
            val = sum(loss_function(model(xb.to("cpu")), yb.to("cpu")).item() * len(xb) for xb, yb in val_dl) / n_val
            if val < best:
                best, bad = val, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break

    model.load_state_dict(best_state)
    model.eval()
    # Again, skip gradient tracking during eval to save memory
    with torch.no_grad():
        predictions = model(torch.as_tensor(test_features, dtype=torch.float32, device="cpu"))
    return predictions.cpu().numpy().ravel()

class AutoEncoder(nn.Module):
    def __init__(self, num_feats, target_d):
        super().__init__()
        int_d1 = num_feats - int((num_feats - target_d)/2)
        int_d2 = int_d1 - int((num_feats - target_d)/4)
        self.encoder = nn.Sequential(
            nn.Linear(num_feats, int_d1),
            nn.ReLU(),
            nn.Linear(int_d1, int_d2),
            nn.ReLU(),
            nn.Linear(int_d2, target_d)
        )
        self.decoder = nn.Sequential(
            nn.Linear(target_d, int_d2),
            nn.ReLU(),
            nn.Linear(int_d2, int_d1),
            nn.ReLU(),
            nn.Linear(int_d1, num_feats)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

def train_autoencoder(model, X_tr, seed, epochs=100, patience=10, lr=1e-3, batch_size=64, val_frac=0.15):
    torch.manual_seed(seed)
    X = torch.as_tensor(np.asarray(X_tr),dtype=torch.float32)

    # Create a chronological tail for validation on the training set
    n_val = int(len(X) * val_frac)
    train_dl = DataLoader(TensorDataset(X[:-n_val]),batch_size=batch_size, shuffle=True)
    X_val = X[-n_val:]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()
    best, best_state, bad = float("inf"), None, 0  

    for epoch in range(epochs):
        model.train()
        for xb, in train_dl:
            opt.zero_grad()
            loss = loss_function(model(xb), xb)
            loss.backward()
            opt.step()
        model.eval()

        # Skip gradient tracking during evaluation to save memory
        with torch.no_grad():
            val = loss_function(model(X_val), X_val).item()
            if val < best:
                best, bad = val, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break

    model.load_state_dict(best_state)
    model.eval()
    return model