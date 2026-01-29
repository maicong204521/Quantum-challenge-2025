import torch
import pandas as pd
import matplotlib.pyplot as plt

import config
from models import PK_LSTM, PD_LSTM
def plot_subject(
    csv_path,
    model_path,
    subject_id,
    mode="PK"   # "PK" or "PD"
):
    """
    Plot Observed vs Predicted for one subject
    """

    df = pd.read_csv(csv_path)
    sdf = df[df["ID"] == subject_id].sort_values("TIME")

    time = torch.tensor(sdf["TIME"].values, dtype=torch.float32)
    dt = torch.diff(time, prepend=time[:1])

    BW = torch.tensor(sdf["BW"].values, dtype=torch.float32)
    COMED = torch.tensor(sdf["COMED"].values, dtype=torch.float32)

    # ===============================
    # Build input
    # ===============================
    if mode == "PK":
        AMT = torch.tensor(sdf["AMT"].values, dtype=torch.float32)
        EVID = torch.tensor(sdf["EVID"].values, dtype=torch.float32)
        CMT = torch.tensor(sdf["CMT"].values, dtype=torch.float32)

        x = torch.stack([dt, AMT, EVID, BW, COMED, CMT], dim=1).unsqueeze(0)

        model = PK_LSTM(
            input_dim=config.PK_INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

        mask = (sdf["DVID"] == 1) & (sdf["MDV"] == 0)
        ylabel = "PK (mg/L)"

    else:  # PD
        PK_PRED = torch.tensor(sdf["PK_PRED"].values, dtype=torch.float32)

        x = torch.stack([dt, BW, COMED, PK_PRED], dim=1).unsqueeze(0)

        model = PD_LSTM(
            input_dim=config.PD_INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

        mask = (sdf["DVID"] == 2) & (sdf["MDV"] == 0)
        ylabel = "Drug Effect (PD)"

    model.eval()

    with torch.no_grad():
        pred = model(x).squeeze().numpy()

    # ===============================
    # Plot
    # ===============================
    plt.figure(figsize=(7, 4))

    plt.plot(
        sdf["TIME"][mask],
        sdf["DV"][mask],
        "o-",
        label="Actual",
        markersize=6, linewidth=2,
                   alpha=0.7, color='blue'
    )

    plt.plot(
        sdf["TIME"][mask],
        pred[mask],
        "s-",
        label="Predicted",
        markersize=6, linewidth=2,
                   alpha=0.7,
        color = 'orange'
    )

    title = (
        f"({mode}) ID: {subject_id} | "
        f"BW: {sdf['BW'].iloc[0]}"
    )

    plt.title(title)
    plt.xlabel("Time (hours)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
plot_subject(
    csv_path="data/QIC2025-EstDat_with_PK.csv",
    model_path="pd_model_best.pt",
    subject_id=13,
    mode="PD"
)
