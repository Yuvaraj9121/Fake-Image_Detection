from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "training"
TEST_DIR = DATASET_DIR / "test"
PREDICTION_DIR = DATASET_DIR / "face_pred"
ORIGINAL_CHECKPOINT = ROOT_DIR / "model_checkpoint.hdf5"
MODEL_DIR = ROOT_DIR / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_model.keras"
RESULTS_DIR = ROOT_DIR / "results"
IMAGE_SIZE = (128, 128)
INPUT_SHAPE = (*IMAGE_SIZE, 3)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
CLASS_NAMES = ("fake", "real")
