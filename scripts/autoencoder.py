# scripts/autoencoder.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from feature_extraction import create_dataset
from data_preprocessing import load_all_subjects

class EEG_Autoencoder(nn.Module):
    def __init__(self, n_channels, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_channels, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_channels)
        )
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# Load data
raws = load_all_subjects()
X, _ = create_dataset(raws)
dataset = TensorDataset(X)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = EEG_Autoencoder(n_channels=X.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Train
for epoch in range(20):
    model.train()
    loss_total = 0
    for xb in loader:
        xb = xb[0].to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, xb)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
    print(f"Epoch {epoch+1}: Loss={loss_total:.3f}")
torch.save(model.state_dict(), "models/autoencoder_eeg.pth")
