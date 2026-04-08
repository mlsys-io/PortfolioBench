import torch
import numpy as np
from pandas import DataFrame
from alpha.interface import IAlpha
import torch.nn as nn

class EventLstmModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm1 = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)

        self.lstm2 = nn.LSTM(input_size=50, hidden_size=30, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)

        self.lstm3 = nn.LSTM(input_size=30, hidden_size=20, batch_first=True)
        self.dropout3 = nn.Dropout(0.05)

        self.fc = nn.Linear(20, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        x, _ = self.lstm3(x)
        x = x[:, -1, :]

        x = self.dropout3(x)
        x = self.fc(x)

        return x

class EventLstmAlpha(IAlpha):
    def __init__(self, dataframe: DataFrame, model_path: str = "./alpha/event_stacked_lstm.pth", seq_len: int = 64, device: str = None, metadata: dict = None):
        super().__init__(dataframe, metadata)
        self.seq_len = seq_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load pre-trained LSTM model
        self.model = EventLstmModel()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()  # evaluation mode

    def process(self) -> DataFrame:
        if "close" not in self.dataframe.columns:
            raise ValueError("DataFrame must contain 'close' column")

        close_prices = self.dataframe["close"].values.astype(np.float32)
        predictions = [np.nan] * self.seq_len
        for i in range(self.seq_len, len(close_prices)):
            seq = close_prices[i - self.seq_len:i]
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)

            with torch.no_grad():
                pred = self.model(x).item()

            predictions.append(pred)

        self.dataframe["lstm_pred"] = predictions
        return self.dataframe