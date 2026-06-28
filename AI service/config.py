import torch

DATA_DIR = "./data/GRAPE"
DATA_PREP = "./data"
OUTPUT_DIR = "./outputs"
CHECKPOINT_DIR = "./checkpoints"
REFUGE_MODEL = "./REFUGE2/checkpoints"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# NAPOMENA: HIDDEN_SIZE=64 (povećan sa 32) i DROPOUT=0.3 (smanjen sa 0.5)
# su testirani nakon dodavanja 'has_cfp' i 'Interval_Years' feature-a.
# Suprotno intuiciji "manji dataset -> manji model", smanjenje
# hidden_size je DRASTIČNO pogoršalo R² (sa -0.9 na -11.8 u test
# scenariju), dok je povećanje na 64 omogućilo modelu da konačno uhvati
# signal (pozitivan R² umesto stalno negativnog). Provereno sanity-check
# testom sa potpuno nasumičnim targetom da ovo NIJE samo overfitting na
# mali validacioni skup — na nasumičnom targetu model ostaje na R²≈0,
# kako i treba.
#
# EPOCHS je povećan na 60 jer kriva uči sporo ali STABILNO čak i do
# epohe 60 (nema platoa) — sa hidden_size=64 možda i treba još epoha;
# vredi probati 80-100 i pratiti da li R² nastavlja da raste ili stagnira.
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.3
BATCH_SIZE = 16
EPOCHS = 60
LR = 0.0005
WEIGHT_DECAY = 1e-4
IMG_SIZE = 512 