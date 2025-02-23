import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


base_dir = "/home/am/Documents/Software Development/10_Academy Training/week-10/brent-data-analysis"

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, num_layers=2, data=None):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, 1)
        self.data = data

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last time step output
        return out

    def fit_lstm(self, epochs=100, batch_size=32):
        """Fit an LSTM model for time series forecasting."""
        print(f"\n{'*'*70}\n")
        print("Fitting LSTM model.\n")

        # Preprocess data: Normalize and create sequences
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(self.data[['Price']].values)

        def create_sequences(data, seq_length):
            sequences = []
            labels = []
            for i in range(len(data) - seq_length):
                sequences.append(data[i:i+seq_length])
                labels.append(data[i+seq_length])
            return np.array(sequences), np.array(labels)

        seq_length = 30  # Example sequence length (lookback period)
        X, y = create_sequences(scaled_data, seq_length)

        # Split data into train and test sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Convert data to PyTorch tensors
        train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
        test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                  torch.tensor(y_test, dtype=torch.float32))

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        # Initialize and train LSTM model
        lstm_model = LSTMModel(input_size=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

        train_losses = []  # List to keep track of the training loss

        for epoch in range(epochs):
            lstm_model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = lstm_model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_epoch_loss)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss}')

        # Plot the training loss over epochs
        plt.figure(figsize=(10, 5))
        plt.plot(range(epochs), train_losses, label="Training Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('LSTM Training Loss')
        plt.legend()
        plt.savefig(f"{base_dir}/notebooks/plots/lstm_training_loss.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Evaluate the model on the test set
        lstm_model.eval()
        predictions = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = lstm_model(X_batch)
                predictions.append(outputs.numpy())

        predictions = np.concatenate(predictions, axis=0)
        predictions = scaler.inverse_transform(predictions)

        # Print LSTM model performance metrics
        mse_lstm = mean_squared_error(self.test['Price'], predictions)
        rmse_lstm = np.sqrt(mse_lstm)
        mae_lstm = mean_absolute_error(self.test['Price'], predictions)
        r2_lstm = r2_score(self.test['Price'], predictions)

        print(f"📊 LSTM Performance:")
        print(f"✅ RMSE: {rmse_lstm}")
        print(f"✅ MAE: {mae_lstm}")
        print(f"✅ R² Score: {r2_lstm}")

        # Plot predictions vs actual values
        plt.figure(figsize=(10, 5))
        plt.plot(self.test.index, self.test['Price'], label="Actual Price")
        plt.plot(self.test.index, predictions, linestyle="dashed", label="Predicted Price (LSTM)")
        plt.legend()
        plt.title("LSTM Predictions vs Actual")
        plt.savefig(f"{base_dir}/notebooks/plots/lstm_prediction.png", dpi=300, bbox_inches="tight")
        plt.show()

        return lstm_model
