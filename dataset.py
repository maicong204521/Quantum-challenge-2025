import pandas as pd
import torch
from torch.utils.data import Dataset

class PKPDDataset(Dataset):
    def __init__(self, csv_path, mode="PK"):
        self.df = pd.read_csv(csv_path)
        self.mode = mode
        self.subjects = self.df["ID"].unique()

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        sid = self.subjects[idx]
        sdf = self.df[self.df["ID"] == sid].sort_values("TIME")

        time = torch.tensor(sdf["TIME"].values, dtype=torch.float32)
        dt = torch.diff(time, prepend=time[:1])

        BW = torch.tensor(sdf["BW"].values, dtype=torch.float32)
        COMED = torch.tensor(sdf["COMED"].values, dtype=torch.float32)
        AMT = torch.tensor(sdf["AMT"].values, dtype=torch.float32)
        EVID = torch.tensor(sdf["EVID"].values, dtype=torch.float32)
        CMT = torch.tensor(sdf["CMT"].values, dtype=torch.float32)

        DV = torch.tensor(sdf["DV"].values, dtype=torch.float32)
        MDV = torch.tensor(sdf["MDV"].values, dtype=torch.float32)
        DVID = torch.tensor(sdf["DVID"].values, dtype=torch.int64)

        if self.mode == "PK":
            mask = (DVID == 1) & (MDV == 0)
            y = DV
            x = torch.stack([dt, AMT, EVID, BW, COMED, CMT], dim=1)

        else:  # PD
            mask = (DVID == 2) & (MDV == 0)
            y = DV

            PK_PRED = torch.tensor(
                sdf["PK_PRED"].values,
                dtype=torch.float32
            )

            x = torch.stack([dt, BW, COMED, PK_PRED], dim=1)


        return x, y, mask
