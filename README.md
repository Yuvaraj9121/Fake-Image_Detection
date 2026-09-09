# Real and Fake Face Detection

A reproducible TensorFlow/Keras image classifier for distinguishing real and fake face images.

## Current implementation

- Original notebook, dataset, and `model_checkpoint.hdf5` are preserved.
- New code lives under `src/` and uses a reproducible validation split from `dataset/training` only.
- The test set is reserved for final evaluation.
- Shared preprocessing converts images to RGB and resizes them to 224x224. EfficientNetB0 applies its model-compatible input scaling.
- Training uses conservative horizontal flips, small rotations, zoom, and contrast variation.
- The classifier uses an ImageNet-pretrained EfficientNetB0 backbone, global average pooling, dropout, and a sigmoid output.

## Installation

Use Python 3.10 and install the pinned dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Commands

```powershell
python -m src.validate_dataset --output results/dataset_report.json
python -m src.train --epochs 10 --fine-tune-epochs 5 --batch-size 16
python -m src.evaluate --model models/best_model.keras
python -m src.predict dataset/face_pred/check.jpg
streamlit run app.py
```

The original checkpoint remains at `model_checkpoint.hdf5`. New training output is saved as `models/best_model.keras` and `models/final_model.keras`.

## Dataset

```text
dataset/
├── training/{fake,real}/
├── test/{fake,real}/
└── face_pred/
```

The available dataset contains 1,437 training images, 604 test images, and 4 unlabelled prediction images. Training contains 658 fake and 779 real images. The untouched test set contains 302 fake and 302 real images. Validation uses 286 images from the training directory, leaving 1,151 images for training. All files are `.jpg`; the audit found no corrupt files or exact duplicate hashes.

## Measured test metrics

The 10-epoch head plus 5-epoch fine-tuned model evaluated on the untouched 604-image test set produced:

| Metric | Result |
| --- | ---: |
| Accuracy | 0.5579 |
| Precision | 0.5464 |
| Recall | 0.6821 |
| F1 | 0.6068 |
| Specificity | 0.4338 |
| ROC-AUC | 0.6033 |

The confusion matrix is `[[131, 171], [96, 206]]` in `[fake, real]` row/column order. The confusion matrix is saved to `results/confusion_matrix.png`; full metrics and the text report are saved to `results/metrics.json` and `results/classification_report.txt`. These results are measured outputs, not a claim that the model is reliable for real-world authenticity decisions. Compared with the old baseline, accuracy, real recall, F1, and specificity improved, while ROC-AUC is essentially unchanged.

## Limitations

Predictions reflect the available dataset and are not proof of image authenticity. Subject-level leakage and generalization to unseen sources require additional validation.
