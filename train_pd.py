import torch
from torch.utils.data import DataLoader
from dataset import PKPDDataset
from models import PD_LSTM
from utils import masked_mse
import config

dataset = PKPDDataset("data/QIC2025-EstDat_with_PK.csv", mode="PD")
loader = DataLoader(dataset, batch_size=1, shuffle=True)

model = PD_LSTM(config.PD_INPUT_DIM, config.HIDDEN_DIM).to(config.DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=config.LR)

for epoch in range(config.EPOCHS_PD):
    total_loss = 0
    for x, y, mask in loader:
        x, y, mask = x.to(config.DEVICE), y.to(config.DEVICE), mask.to(config.DEVICE)
        pred = model(x)
        loss = masked_mse(pred, y, mask)

        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item()

    print(f"[PD] Epoch {epoch}: Loss = {total_loss:.4f}")

torch.save(model.state_dict(), "pd_model.pt")
