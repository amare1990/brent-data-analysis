import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


base_dir = "/home/am/Documents/Software Development/10_Academy Training/week-10/brent-data-analysis"


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Use the last time step output


class LSTMTimeSeries:
    def __init__(self, data, seq_length=30, epochs=100, batch_size=32):
        self.data = data
        self.seq_length = seq_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.model = LSTMModel(input_size=1)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def preprocess_data(self, data, is_training=True): # Add data argument with default to None
        """Preprocess data by normalizing and creating sequences."""
        if is_training:  # Only fit_transform during training
            scaled_data = self.scaler.fit_transform(data[['Price']].values)
        else:  # Use transform during testing/evaluation
            scaled_data = self.scaler.transform(data[['Price']].values)

        def create_sequences(data, seq_length):
            sequences, labels = [], []
            for i in range(len(data) - seq_length):
                sequences.append(data[i:i + seq_length])
                labels.append(data[i + seq_length])
            return np.array(sequences), np.array(labels)

        # Fix: Call create_sequences and return the results
        X, y = create_sequences(scaled_data, self.seq_length)
        return X, y


    def train_model(self, X_train, y_train):
        """Train the LSTM model."""
        train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                   torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        train_losses = []
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_epoch_loss)
            print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {avg_epoch_loss}')

        return train_losses

    def evaluate_model(self, X_test, y_test):
        """Evaluate the LSTM model on the test set."""
        test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                  torch.tensor(y_test, dtype=torch.float32))
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        predictions = []
        self.model.eval()
        with torch.no_grad():
            for X_batch, _ in test_loader:
                output = self.model(X_batch)
                predictions.append(output.numpy())

        predictions = np.concatenate(predictions, axis=0)
        return self.scaler.inverse_transform(predictions)

    def plot_loss(self, train_losses):
        """Plot the training loss over epochs."""
        plt.figure(figsize=(10, 5))
        plt.plot(range(self.epochs), train_losses, label="Training Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('LSTM Training Loss')
        plt.legend()
        plt.savefig(f"{base_dir}/notebooks/plots/lstm_training_loss.png", dpi=300, bbox_inches="tight")
        plt.show()

    def plot_predictions(self, actual, predicted):
        """Plot predictions vs actual values."""
        plt.figure(figsize=(10, 5))
        plt.plot(actual.index, actual['Price'], label="Actual Price")
        plt.plot(actual.index, predicted, linestyle="dashed", label="Predicted Price (LSTM)")
        plt.legend()
        plt.title("LSTM Predictions vs Actual")
        plt.savefig(f"{base_dir}/notebooks/plots/lstm_prediction.png", dpi=300, bbox_inches="tight")
        plt.show()

    def calculate_metrics(self, actual, predicted):
        """Calculate performance metrics."""
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)

        return rmse, mae, r2

    def fit(self):
        """Fit an LSTM model for time series forecasting."""
        print(f"\n{'*' * 100}\n")
        print("Fitting LSTM model.\n")

        X, y = self.preprocess_data(data=self.data)

        # Split data into train and test sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Train the LSTM model
        train_losses = self.train_model(X_train, y_train)

        # Plot training loss
        self.plot_loss(train_losses)

        # Evaluate the model
        predictions = self.evaluate_model(X_test, y_test)

        # Create test DataFrame from original data
        test_size = len(self.data) - train_size
        test_data = self.data[train_size + self.seq_length:]  # Test DataFrame

        # Calculate performance metrics
        rmse, mae, r2 = self.calculate_metrics(test_data['Price'], predictions)
        # Store the metrics in dictionary
        perf_metrics = {
            "rmse": rmse, "mae": mae, "r2": r2
        }

        print(f"📊 LSTM Performance:")
        print(f"✅ RMSE: {rmse}")
        print(f"✅ MAE: {mae}")
        print(f"✅ R² Score: {r2}")

        # Plot predictions vs actual values
        self.plot_predictions(test_data, predictions)

        return self.model, perf_metrics
