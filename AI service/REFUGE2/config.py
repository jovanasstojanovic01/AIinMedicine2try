import torch

DATA_DIR = "./data/REFUGE2"
CHECKPOINT_DIR = "./checkpoints"
OUTPUT_DIR = "./outputs"

IMG_SIZE = 512 

BATCH_SIZE = 4 
LEARNING_RATE = 1e-4
EPOCHS = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Multi-task loss težine (možemo ih fino podešavati kasnije)
# Pošto imamo 3 zadatka: segmentacija, lokalizacija (fovea), klasifikacija (glaukom)
LOSS_WEIGHTS = {
    "segmentation": 1.0,
    "localization": 0.5,
    "classification": 0.5
}