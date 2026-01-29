import numpy as np
import pandas as pd
import torch
import math

import config
from models import PK_LSTM, PD_LSTM

# ============================================================
# PATHS
# ============================================================
DATA_CSV = "data/QIC2025-EstDat.csv"
PK_MODEL_PATH = "pk_model_best.pt"
PD_MODEL_PATH = "pd_model_best.pt"

# ============================================================
# SETTINGS
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BIOMARKER_THRESHOLD = 3.3  # ng/mL

DAILY_TAU_H = 24
WEEKLY_TAU_H = DAILY_TAU_H # 168

DAILY_STEP_MG = 0.5 
WEEKLY_STEP_MG = 5

N_CYCLES_DAILY = 50
N_CYCLES_WEEKLY = 8

DT_GRID_H = 0.5

CMT_DOSE = 1
CMT_OBS = 2

# ============================================================
# LOAD MODELS
# ============================================================
pk_model = PK_LSTM(config.PK_INPUT_DIM, config.HIDDEN_DIM)
pd_model = PD_LSTM(config.PD_INPUT_DIM, config.HIDDEN_DIM)

pk_model.load_state_dict(torch.load(PK_MODEL_PATH, map_location="cpu"))
pd_model.load_state_dict(torch.load(PD_MODEL_PATH, map_location="cpu"))

pk_model.to(DEVICE).eval()
pd_model.to(DEVICE).eval()

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATA_CSV)
subject_ids = df["ID"].unique()

print(f"Loaded {len(subject_ids)} subjects")

# ============================================================
# BUILD STEADY-STATE SEQUENCE (EXISTING SUBJECT)
# ============================================================
def build_sequence_existing_subject(
    sdf: pd.DataFrame,
    dose_mg: float,
    tau_h: float,
    n_cycles: int,
    dt_grid_h: float
):
    bw = float(sdf["BW"].iloc[0])
    comed = float(sdf["COMED"].iloc[0])

    t_end = n_cycles * tau_h
    t0_last = (n_cycles - 1) * tau_h

    obs_times = np.arange(t0_last, t_end + 1e-9, dt_grid_h)
    dose_times = np.arange(0.0, t_end + 1e-9, tau_h)

    times = np.unique(np.concatenate([dose_times, obs_times]))
    times.sort()

    is_dose = np.isin(times, dose_times)

    AMT = np.where(is_dose, dose_mg, 0.0).astype(np.float32)
    EVID = np.where(is_dose, 1.0, 0.0).astype(np.float32)
    CMT = np.where(is_dose, CMT_DOSE, CMT_OBS).astype(np.float32)

    BW = np.full_like(times, bw, dtype=np.float32)
    COMED = np.full_like(times, comed, dtype=np.float32)

    dt = np.diff(times, prepend=times[0]).astype(np.float32)

    x_pk = torch.tensor(
        np.stack([dt, AMT, EVID, BW, COMED, CMT], axis=1),
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    obs_mask = (~is_dose) & (times >= t0_last)

    return obs_mask, x_pk, BW, COMED

# ============================================================
# SIMULATE ONE SUBJECT
# ============================================================
@torch.no_grad()
def simulate_subject(
    sdf: pd.DataFrame,
    dose_mg: float,
    tau_h: float,
    n_cycles: int,
    dt_grid_h: float
):
    obs_mask, x_pk, BW, COMED = build_sequence_existing_subject(
        sdf, dose_mg, tau_h, n_cycles, dt_grid_h
    )

    pk_pred = pk_model(x_pk).squeeze(0)

    dt = x_pk.squeeze(0)[:, 0].cpu().numpy()

    x_pd = torch.tensor(
        np.stack([dt, BW, COMED, pk_pred.cpu().numpy()], axis=1),
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    pd_pred = pd_model(x_pd).squeeze(0).cpu().numpy()

    return pd_pred[obs_mask]

# ============================================================
# EVALUATE DOSE ON EXISTING POPULATION
# ============================================================
def evaluate_dose_existing_population(
    df_use: pd.DataFrame,
    dose_mg: float,
    tau_h: float,
    n_cycles: int,
    dt_grid_h: float
):
    subject_ids = df_use["ID"].unique()
    success = 0

    for sid in subject_ids:
        sdf = df_use[df_use["ID"] == sid].sort_values("TIME")
        biomarker = simulate_subject(
            sdf, dose_mg, tau_h, n_cycles, dt_grid_h
        )

        if len(biomarker) == 0:
            continue

        if np.all(biomarker < BIOMARKER_THRESHOLD):
            success += 1

    return success / len(subject_ids)

# ============================================================
# FIND MIN DOSE
# ============================================================
def find_min_dose_existing_population(
    df_use: pd.DataFrame,
    tau_h: float,
    n_cycles: int,
    dt_grid_h: float,
    target_prob: float,
    dose_step: float,
    dose_min: float,
    dose_max: float
):
    doses = np.arange(dose_min, dose_max + 1e-9, dose_step)
    sdf = df_use[(df_use["DOSE"] != 0)]

    for d in doses:
        rate = evaluate_dose_existing_population(
            sdf, d, tau_h, n_cycles, dt_grid_h
        )
        print(f"Dose {d:.2f} mg → {rate*100:.1f}% subjects")

        if rate >= target_prob:
            return d, rate

    return None, None

##################Weekly#####################################
def build_sequence_existing_subject_weekly(
    sdf: pd.DataFrame,
    weekly_dose_mg: float,
    n_cycles: int,          # số tuần
    dt_grid_h: float = 24.0 # daily grid (giống training)
):
    bw = float(sdf["BW"].iloc[0])
    comed = float(sdf["COMED"].iloc[0])

    tau_week = 168.0
    t_end = n_cycles * tau_week
    t0_last = (n_cycles - 1) * tau_week

    # ===== Time grid: mỗi 24h =====
    times = np.arange(0.0, t_end + 1e-9, dt_grid_h)

    # ===== Weekly bolus: mỗi 168h =====
    dose_times = np.arange(0.0, t_end + 1e-9, tau_week)
    is_dose_day = np.isin(times, dose_times)

    # ===== Event structure (GIỐNG TRAINING) =====
    # Mỗi ngày đều có event
    EVID = np.ones_like(times, dtype=np.float32)

    # AMT: chỉ có dose mỗi 7 ngày
    AMT = np.where(is_dose_day, weekly_dose_mg, 0.0).astype(np.float32)

    # CMT: chỉ là DOSE khi có AMT > 0, còn lại là OBS
    CMT = np.where(is_dose_day, CMT_DOSE, CMT_OBS).astype(np.float32)

    # ===== Covariates =====
    BW = np.full_like(times, bw, dtype=np.float32)
    COMED = np.full_like(times, comed, dtype=np.float32)

    # ===== dt =====
    dt = np.diff(times, prepend=times[0]).astype(np.float32)

    # ===== PK input =====
    x_pk = torch.tensor(
        np.stack([dt, AMT, EVID, BW, COMED, CMT], axis=1),
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    # ===== Observation mask: tuần cuối, bỏ ngày dose =====
    obs_mask = (times >= t0_last) & (~is_dose_day)

    return obs_mask, x_pk, BW, COMED


@torch.no_grad()
def simulate_subject_weekly(
    sdf: pd.DataFrame,
    weekly_dose_mg: float,
    n_cycles: int
):
    obs_mask, x_pk, BW, COMED = build_sequence_existing_subject_weekly(
        sdf, weekly_dose_mg, n_cycles
    )

    pk_pred = pk_model(x_pk).squeeze(0)

    dt = x_pk.squeeze(0)[:, 0].cpu().numpy()

    x_pd = torch.tensor(
        np.stack([dt, BW, COMED, pk_pred.cpu().numpy()], axis=1),
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    pd_pred = pd_model(x_pd).squeeze(0).cpu().numpy()

    return pd_pred[obs_mask]

def evaluate_weekly_dose_existing_population(
    df_use: pd.DataFrame,
    weekly_dose_mg: float,
    n_cycles: int,
    threshold: float = BIOMARKER_THRESHOLD
):
    subject_ids = df_use["ID"].unique()
    success = 0

    for sid in subject_ids:
        sdf = df_use[df_use["ID"] == sid].sort_values("TIME")
        biomarker = simulate_subject_weekly(
            sdf, weekly_dose_mg, n_cycles
        )

        if len(biomarker) == 0:
            continue

        if np.all(biomarker < threshold):
            success += 1

    return success / len(subject_ids)

def find_min_weekly_dose_existing_population(
    df_use: pd.DataFrame,
    n_cycles: int,
    target_prob: float,
    dose_step: float,
    dose_min: float,
    dose_max: float
):
    doses = np.arange(dose_min, dose_max + 1e-9, dose_step)
    sdf = df_use[df_use["DOSE"] != 0]

    for d in doses:
        rate = evaluate_weekly_dose_existing_population(
            sdf, d, n_cycles
        )
        print(f"Weekly dose {d:.1f} mg → {rate*100:.1f}% subjects")

        if rate >= target_prob:
            return d, rate

    return None, None


# ============================================================
# MAIN TASKS
# ============================================================
def run_all_tasks():

    max_dose_data = float(pd.to_numeric(df["DOSE"], errors="coerce").max())
    if math.isnan(max_dose_data):
        max_dose_data = 10.0

    daily_max = max(5.0, max_dose_data * 2)
    weekly_max = max(20.0, max_dose_data * 10)

    print("\n================ TASK 1 =================")
    print("Once-daily dosing, target = 90%")

    d_daily_90, _ = find_min_dose_existing_population(
        df,
        DAILY_TAU_H, N_CYCLES_DAILY, DT_GRID_H,
        target_prob=0.90,
        dose_step=DAILY_STEP_MG,
        dose_min=DAILY_STEP_MG,
        dose_max=daily_max
    )

    print("\n================ TASK 2 =================")
    print("Once-weekly dosing, target = 90%")

    # d_weekly_90, _ = find_min_dose_existing_population(
    #     df,
    #     WEEKLY_TAU_H, N_CYCLES_WEEKLY, DT_GRID_H,
    #     target_prob=0.90,
    #     dose_step=WEEKLY_STEP_MG,
    #     dose_min=WEEKLY_STEP_MG,
    #     dose_max=weekly_max
    # )

    print("\n================ TASK 2 =================")
    print("Once-weekly dosing (1 dose / week), target = 90%")

    d_weekly_90, r_weekly_90 = find_min_weekly_dose_existing_population(
        df_use=df,
        n_cycles=N_CYCLES_WEEKLY,     # số tuần để đạt steady-state
        target_prob=0.90,
        dose_step=WEEKLY_STEP_MG,     # ví dụ 5 mg
        dose_min=WEEKLY_STEP_MG,
        dose_max=weekly_max
    )

    

    # print("\n================ TASK 4 =================")
    # print("No COMED allowed (COMED = 0), target = 90%")

    # df_nocomed = df.copy()
    # df_nocomed["COMED"] = 0.0

    # d_daily_nocomed_90, _ = find_min_dose_existing_population(
    #     df_nocomed,
    #     DAILY_TAU_H, N_CYCLES_DAILY, DT_GRID_H,
    #     target_prob=0.90,
    #     dose_step=DAILY_STEP_MG,
    #     dose_min=DAILY_STEP_MG,
    #     dose_max=daily_max
    # )

    # d_weekly_nocomed_90, _ = find_min_dose_existing_population(
    #     df_nocomed,
    #     WEEKLY_TAU_H, N_CYCLES_WEEKLY, DT_GRID_H,
    #     target_prob=0.90,
    #     dose_step=WEEKLY_STEP_MG,
    #     dose_min=WEEKLY_STEP_MG,
    #     dose_max=weekly_max
    # )

    # print("\n================ TASK 5 =================")
    # print("Target relaxed to 75%")

    # d_daily_75, _ = find_min_dose_existing_population(
    #     df,
    #     DAILY_TAU_H, N_CYCLES_DAILY, DT_GRID_H,
    #     target_prob=0.75,
    #     dose_step=DAILY_STEP_MG,
    #     dose_min=DAILY_STEP_MG,
    #     dose_max=daily_max
    # )

    # d_weekly_75, _ = find_min_dose_existing_population(
    #     df,
    #     WEEKLY_TAU_H, N_CYCLES_WEEKLY, DT_GRID_H,
    #     target_prob=0.75,
    #     dose_step=WEEKLY_STEP_MG,
    #     dose_min=WEEKLY_STEP_MG,
    #     dose_max=weekly_max
    # )

    print("\n================ SUMMARY =================")
    print(f"Daily 90%  : {d_daily_90} mg")
    print(f"Weekly 90% : {d_weekly_90} mg")
    # print(f"Daily 75%  : {d_daily_75} mg")
    # print(f"Weekly 75% : {d_weekly_75} mg")

    bio = simulate_subject_weekly(df, 100.0, n_cycles=8)
    print(bio.min(), bio.max())

# ============================================================
if __name__ == "__main__":
    run_all_tasks()


