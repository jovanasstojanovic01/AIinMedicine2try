import torch

DATA_DIR = "./data/GRAPE"
OUTPUT_DIR = "./outputs"
CHECKPOINT_DIR = "./checkpoints"
REFUGE_MODEL = "./REFUGE2/checkpoints"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
     
HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.5
BATCH_SIZE = 16
EPOCHS = 40
LR = 0.0005         
WEIGHT_DECAY = 1e-4 
IMG_SIZE = 512 