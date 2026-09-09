# Real and Fake Face Detection

A reproducible TensorFlow/Keras image classifier for distinguishing real and fake face images.

## Current implementation

- Original notebook, dataset, and `model_checkpoint.hdf5` are preserved.
- New code lives under `src/` and uses a reproducible validation split from `dataset/training`.
- The test set is reserved for final evaluation.
- Shared preprocessing resizes images to 128x128 RGB and normalizes pixels to `[0, 1]`.

## Installation

Use Python 3.10 and install the pinned dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Commands

```powershell
python -m src.validate_dataset --output results/dataset_report.json
python -m src.train --epochs 10
python -m src.evaluate --model models/best_model.keras
python -m src.predict dataset/face_pred/check.jpg
streamlit run app.py
```

The original checkpoint remains at `model_checkpoint.hdf5`. New training output is saved as `models/best_model.keras`.

## Dataset

```text
dataset/
├── training/{fake,real}/
├── test/{fake,real}/
└── face_pred/
```

The available dataset contains 1,437 training images, 604 test images, and 4 unlabelled prediction images. Validation uses 286 images from the training directory, leaving 1,151 images for training.

## Measured test metrics

The 10-epoch model evaluated on the untouched 604-image test set produced:

| Metric | Result |
| --- | ---: |
| Accuracy | 0.5083 |
| Precision | 0.5042 |
| Recall | 0.9934 |
| F1 | 0.6689 |
| Specificity | 0.0232 |
| ROC-AUC | 0.6024 |

The confusion matrix is saved to `results/confusion_matrix.png`, and the full report is saved to `results/metrics.json`. These results are measured outputs, not a claim that the model is reliable for real-world authenticity decisions.

## Limitations

Predictions reflect the available dataset and are not proof of image authenticity. Subject-level leakage and generalization to unseen sources require additional validation.
