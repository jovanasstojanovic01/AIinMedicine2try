import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    # SQLALCHEMY_DATABASE_URI = "postgresql://korisnik:lozinka@localhost:5432/ime_baze"
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'glaucoma.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'uploads')
    IMAGES_FOLDER = os.path.join(UPLOAD_FOLDER, 'images')
    MASKS_FOLDER = os.path.join(UPLOAD_FOLDER, 'masks')

    WEIGHTS_FOLDER = os.path.join(BASE_DIR, 'ml', 'weights')


    REFUGEUNET_WEIGHTS = os.getenv(
        "REFUGEUNET_WEIGHTS", os.path.join(WEIGHTS_FOLDER, "refuge_unet.pth")
    )
    GRU_WEIGHTS = os.getenv(
        "GRU_WEIGHTS", os.path.join(WEIGHTS_FOLDER,"gru.pth")
    )
    XGB_MODEL = os.getenv(
        "XGB_MODEL", os.path.join(WEIGHTS_FOLDER, "xgboost_model.json")
    )

    VF_POINTS = 61

    EXTRA_FEATURES = 5

    INPUT_FEATURES = VF_POINTS + EXTRA_FEATURES  

    MAX_TIMESTEPS = 10

    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT = 0.3

    CFP_IMAGE_SIZE = 512

    MAX_CONTENT_LENGTH = 16 * 1024 * 1024 