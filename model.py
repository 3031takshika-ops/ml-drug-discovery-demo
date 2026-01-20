import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Generate synthetic molecular data
# -----------------------------
def generate_data(n_samples=500, n_features=10):
    np.random.seed(42)
    X = np.random.rand(n_samples, n_features)
    y = X.sum(axis=1) + np.random.normal(0, 0.1, size=n_samples)
    return X, y


# -----------------------------
# Neural network model
# -----------------------------
class MolecularPropertyPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


# -----------------------------
# Training loop
# -----------------------------
def train_model(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        predictions = model(x_batch).squeeze()
        loss = loss_fn(predictions, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    X, y = generate_data()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = MolecularPropertyPredictor(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(20):
        loss = train_model(model, train_loader, optimizer, loss_fn)
        print(f"Epoch {epoch + 1} | Training Loss: {loss:.4f}")

    print("Training complete.")
