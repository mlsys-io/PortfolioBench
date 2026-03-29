import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

class AR(nn.Module):
    def __init__(self, lag):
        super().__init__()
        self.lag = lag
        self.linear = nn.Linear(lag, 1, bias=True)
        
    def forward(self, x):
        return self.linear(x)
    
def train_one_epoch(loader: DataLoader, model: nn.Module, loss_fn, optimizer):
    model.train()
    size = len(loader.dataset)
    running_loss = 0.0
    for _, (xb, yb) in enumerate(loader):
        xb = xb.float()
        yb = yb.float()
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    epoch_loss = running_loss / size
    print(f"Average traning loss: {epoch_loss: .10e}")
    return epoch_loss

def train(train_dataset: Dataset, epoches: int, ar_model: nn.Module):
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(ar_model.parameters(), lr=1e-3)
    loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False)
    for _ in range(epoches):
        train_one_epoch(loader, ar_model, loss_fn, optimizer)

def build_train(data, T):
    X = []
    y = []
    n = len(data)
    for i in range(n - T):
        window = data[i : i + T]
        target = data[i + T]
        X.append(window)
        y.append(target)
        
    return np.array(X), np.array(y)

class ArDataset(Dataset):
    def __init__(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        self.X = X.float()
        self.y = y.float()
        self.y = self.y.unsqueeze(1)
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def infer_mu(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        mu = model(X_tensor).squeeze(1).cpu().numpy()
    return mu

def load_ar_model(model_path="./ar_model.pth", lag=90):
    model = AR(lag=lag)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

base_dir = Path(__file__).resolve().parent
model_path = base_dir / "ar_model.pth"
trained_ar_model = load_ar_model(model_path)

# if __name__ == "__main__":
#     print("1. Define model")
#     ar_model = AR(lag=90)
    
#     print("2. Load data")
#     data = load_data()
#     log_ret = data["log_return"].to_numpy()
#     log_ret = log_ret[np.isfinite(log_ret)]
#     X, y = build_train(log_ret, 90)
#     print(X[0], y[0])
    
#     print("3. Prepare training")
#     train_idx = int(len(X) * 0.85)
#     X_train = X[:train_idx]
#     y_train = y[:train_idx]
#     dataset = ArDataset(X_train, y_train)
    
#     print("4. Train")
#     train(dataset, 10, ar_model)
    
#     print("5. Test inference")
#     ar_model.eval()
#     X_test = X[train_idx:]
#     y_test = y[train_idx:]

#     mu_test = infer_mu(ar_model, X_test)
#     residuals = y_test - mu_test

#     mae = np.mean(np.abs(residuals))
#     print("Test MAE:", mae)
    
#     print("6. Save model")
#     torch.save(ar_model.state_dict(), "./ar_model.pth")