import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Model
PK_INPUT_DIM = 6
PD_INPUT_DIM = 4
HIDDEN_DIM = 128
NUM_LAYERS = 2

# Training
BATCH_SIZE = 32
LR = 1e-3
EPOCHS_PK = 2000
EPOCHS_PD = 4000

# Data
TIME_SCALE = 1.0
