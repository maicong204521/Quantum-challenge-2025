import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

import config
from dataset import PKPDDataset
from models import PK_LSTM, PD_LSTM
from utils import masked_mse


# ======================================================
# Utility: Train one epoch
# ======================================================
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for x, y, mask in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        pred = model(x)
        loss = masked_mse(pred, y, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ======================================================
# Step 1: Train PK model
# ======================================================
def train_pk():
    print("=== Training PK-LSTM ===")

    dataset = PKPDDataset("data/QIC2025-EstDat.csv", mode="PK")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = PK_LSTM(
        input_dim=config.PK_INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM
    ).to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    for epoch in range(1, config.EPOCHS_PK + 1):
        loss = train_epoch(model, loader, optimizer, config.DEVICE)
        print(f"[PK] Epoch {epoch:03d} | Loss: {loss:.6f}")

    torch.save(model.state_dict(), "pk_model.pt")
    print("✓ PK model saved to pk_model.pt")

    return model


# ======================================================
# Step 2: Generate PK predictions
# ======================================================
def generate_pk_predictions(pk_model):
    print("=== Generating PK predictions ===")

    pk_model.eval()
    df = pd.read_csv("data/QIC2025-EstDat.csv")

    df["PK_PRED"] = 0.0

    for sid in df["ID"].unique():
        sdf = df[df["ID"] == sid].sort_values("TIME")

        time = torch.tensor(sdf["TIME"].values, dtype=torch.float32)
        dt = torch.diff(time, prepend=time[:1])

        x = torch.stack([
            dt,
            torch.tensor(sdf["AMT"].values, dtype=torch.float32),
            torch.tensor(sdf["EVID"].values, dtype=torch.float32),
            torch.tensor(sdf["BW"].values, dtype=torch.float32),
            torch.tensor(sdf["COMED"].values, dtype=torch.float32),
            torch.tensor(sdf["CMT"].values, dtype=torch.float32),
        ], dim=1).unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            pk_pred = pk_model(x).cpu().numpy().squeeze()

        df.loc[sdf.index, "PK_PRED"] = pk_pred

    out_path = "data/QIC2025-EstDat_with_PK.csv"
    df.to_csv(out_path, index=False)
    print(f"✓ PK predictions saved to {out_path}")

def evaluate_pk_rmse(pk_model):
    print("=== Evaluating PK RMSE ===")

    pk_model.eval()
    dataset = PKPDDataset("data/QIC2025-EstDat.csv", mode="PK")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    rmses = []

    with torch.no_grad():
        for x, y, mask in loader:
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)
            mask = mask.to(config.DEVICE)

            pred = pk_model(x)
            rmse = torch.sqrt(
                ((pred - y) ** 2 * mask).sum() / mask.sum().clamp(min=1)
            )
            rmses.append(rmse.item())

    print(f"✓ PK RMSE (mean over subjects): {sum(rmses)/len(rmses):.4f}")

# ======================================================
# Step 3: Train PD model
# ======================================================
def train_pd():
    print("=== Training PD-LSTM ===")

    dataset = PKPDDataset("data/QIC2025-EstDat_with_PK.csv", mode="PD")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = PD_LSTM(
        input_dim=config.PD_INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM
    ).to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    for epoch in range(1, config.EPOCHS_PD + 1):
        loss = train_epoch(model, loader, optimizer, config.DEVICE)
        print(f"[PD] Epoch {epoch:03d} | Loss: {loss:.6f}")

    torch.save(model.state_dict(), "pd_model.pt")
    print("✓ PD model saved to pd_model.pt")
    return model

def evaluate_pd_rmse(pd_model):
    print("=== Evaluating PD RMSE ===")

    pd_model.eval()
    dataset = PKPDDataset("data/QIC2025-EstDat_with_PK.csv", mode="PD")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    rmses = []

    with torch.no_grad():
        for x, y, mask in loader:
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)
            mask = mask.to(config.DEVICE)

            pred = pd_model(x)
            rmse = torch.sqrt(
                ((pred - y) ** 2 * mask).sum() / mask.sum().clamp(min=1)
            )
            rmses.append(rmse.item())

    print(f"✓ PD RMSE (mean over subjects): {sum(rmses)/len(rmses):.4f}")


# ======================================================
# Main execution
# ======================================================
if __name__ == "__main__":

    os.makedirs("data", exist_ok=True)

    print("===================================")
    print(" PK–PD LSTM End-to-End Training ")
    print("===================================")

    # Train PK
    pk_model = train_pk()

    # PK RMSE
    evaluate_pk_rmse(pk_model)

    # Generate PK predictions
    generate_pk_predictions(pk_model)

    # Train PD
    pd_model = train_pd()

    # PD RMSE
    evaluate_pd_rmse(pd_model)

    print("===================================")
    print(" Training completed successfully ")
    print("===================================")

