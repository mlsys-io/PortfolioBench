import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


class MLPClassifier(nn.Module):
    def __init__(self, in_dim, n_classes, epochs=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, n_classes),
            # nn.Softmax(dim=1) in actual output pass through softmax
        )

        self.epochs = epochs

    def forward(self, x):
        return self.model(x)

    def fit(self, train_data, y, batch_size=64):
        device = next(self.model.parameters()).device
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        X = torch.tensor(train_data, dtype=torch.float32)
        y = torch.tensor(y_encoded, dtype=torch.long)

        dataset = TensorDataset(X, y)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=(device.type == "cuda")
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        best_loss = float("inf")
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()         
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(loader)
            print(f"Epoch {epoch+1:02d} | Loss: {epoch_loss:.4f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(
                    self.model.state_dict(),
                    "model_best_weights.pt"
                )

        return True
    
    def test(self, dataloader, loss_fn=None):
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        self.model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                pred = self.model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        
        
