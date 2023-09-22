import numpy as np

def detect_anomalies_autoencoder(data:np.array,
                                 threshold:float) -> np.array:
    import torch
    import torch.nn as nn

    # define the autoencoder model
    class Autoencoder(nn.Module):
        def __init__(self,n):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n, min(128, int(n/3))),
                nn.ReLU(),
                nn.Linear(min(128, int(n/3)), min(64, int(n/6))),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(min(64, int(n/6)), min(128, int(n/3))),
                nn.ReLU(),
                nn.Linear(min(128, int(n/3)), n),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x
        
    stdev = np.std(data)

    # convert data to PyTorch tensor
    data = torch.tensor(data, dtype=torch.float32)

    # initialize the model and optimizer
    model = Autoencoder(len(data))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.HuberLoss()

    # training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, data)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # evaluate the model on the training data
    with torch.no_grad():
        reconstructions = model(data)
        mse_loss = nn.MSELoss()(reconstructions, data)
        print(f'Mean Squared Error (MSE) on Training Data: {mse_loss.item()}')

    # calculate reconstruction errors
    reconstruction_errors = (reconstructions - data).pow(2)

    # detect anomalies based on reconstruction error
    anomalies = (reconstruction_errors > threshold*stdev).numpy()

    return np.where(anomalies)[0]

def detect_anomalies_isolation_forest(data : np.array,
                                      contamination = "auto"):
    from sklearn.ensemble import IsolationForest
    
    # create the Isolation Forest model
    model = IsolationForest(contamination=contamination)  # Adjust the contamination parameter as needed

    # fit the model to your data
    model.fit(data)

    # predict anomalies
    anomaly_scores = model.predict(data)

    # convert anomaly scores to a binary format (1 for normal, -1 for anomaly)
    is_anomaly = np.where(anomaly_scores == -1, True, False)

    return np.where(is_anomaly)[0]

def detect_anomalies_lof(data : np.array, contamination = "auto"):
    from sklearn.neighbors import LocalOutlierFactor

    # create an LOF model
    lof_model = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    
    # fit the model to the data and obtain anomaly scores
    anomaly_scores = lof_model.fit_predict(data)
    
    # convert anomaly scores to a binary format (1 for normal, -1 for anomaly)
    is_anomaly = np.where(anomaly_scores == -1, True, False)

    # get the indices where is_anomaly is True
    anomaly_indices = np.where(is_anomaly)[0]

    return anomaly_indices

def detect_anomalies_one_class_svm(data, nu=0.1):
    from sklearn.svm import OneClassSVM

    # create a One-Class SVM model
    one_class_svm_model = OneClassSVM(nu=nu)
    
    # fit the model to the data and obtain anomaly scores
    one_class_svm_model.fit(data)
    anomaly_scores = one_class_svm_model.decision_function(data)
    
    # convert anomaly scores to a binary format (1 for normal, -1 for anomaly)
    is_anomaly = np.where(anomaly_scores < 0, True, False)

    # get the indices where is_anomaly is True
    anomaly_indices = np.where(is_anomaly)[0]

    return anomaly_indices

def detect_anomalies_stdev(data,threshold=2):
    return np.where(np.abs(data - np.median(data)) >  2*np.std(data))[0]
