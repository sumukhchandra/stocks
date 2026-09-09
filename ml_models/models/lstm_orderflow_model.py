import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class LSTMOrderFlowModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1):
        super(LSTMOrderFlowModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

class LSTMTrainer:
    def __init__(self, input_dim, model_dir='models/saved_models/lstm'):
        self.model = LSTMOrderFlowModel(input_dim)
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.scaler = StandardScaler()

    def create_sequences(self, data, target, seq_length=32):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i : i + seq_length])
            y.append(target[i + seq_length])
        return np.array(X), np.array(y)

    def train(self, X_train, y_train, epochs=10, batch_size=64):
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        
        X_tensor = torch.FloatTensor(X_train_scaled)
        y_tensor = torch.FloatTensor(y_train).view(-1, 1)
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 2 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, 'lstm_model.pth'))
        import joblib
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.joblib'))

if __name__ == "__main__":
    data_path = 'data/processed/master_labeled_dataset.parquet'
    if os.path.exists(data_path):
        df = pd.read_parquet(data_path)
        # Select microstructure features
        features = ['vpin', 'imbalance', 'spread', 'volume_delta', 'ofi']
        features = [f for f in features if f in df.columns]
        
        X_data = df[features].values
        y_data = df['target'].values
        
        trainer = LSTMTrainer(input_dim=len(features))
        X_seq, y_seq = trainer.create_sequences(X_data, y_data, seq_length=16)
        
        print(f"Training LSTM on sequences: {X_seq.shape}")
        trainer.train(X_seq, y_seq, epochs=10)
        trainer.save()
        print("LSTM training complete.")
    else:
        print("Data not found.")
