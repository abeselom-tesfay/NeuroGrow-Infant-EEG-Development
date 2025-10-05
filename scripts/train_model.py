# scripts/train_model.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from feature_extraction import create_dataset
from data_preprocessing import load_all_subjects

# Load data
raws = load_all_subjects()
X, y = create_dataset(raws)
dataset = TensorDataset(X, y)

# Train/val split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

# CNN+LSTM model
class CNN_LSTM(nn.Module):
    def __init__(self, n_channels=32, n_classes=4):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.fc = nn.Linear(128, n_classes)
        
    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN_LSTM(n_channels=X.shape[1], n_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(30):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    train_acc = correct / total
    print(f"Epoch {epoch+1}: Loss={train_loss:.3f}, Train Acc={train_acc:.3f}")

# Save model
torch.save(model.state_dict(), "models/cnn_lstm_eeg.pth")
print("Model saved to models/cnn_lstm_eeg.pth")
