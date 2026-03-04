# Facial Expression Recognition using CNN
> End-to-end deep learning pipeline trained on 35,887 facial images across 6 emotion classes

---

## Problem
Building a system that automatically classifies human emotions from facial images —
a core capability for human-computer interaction, behavioral analytics, and 
affective computing applications.

---

## Dataset — FER-2013
- **35,887** grayscale images, 48×48 pixels
- **6 emotion classes:** Angry, Fear, Happy, Sad, Surprise, Neutral
- Source: [Kaggle FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
- Note: 'Disgust' class dropped due to severe class imbalance

---

## Pipeline
```
Raw Images → DataFrame Creation → Train/Val/Test Split
→ Augmentation → CNN Training → Evaluation → Grad-CAM Visualization
```

---

## Model Architecture

| Layer | Details |
|-------|---------|
| Conv2D | 32 filters, 3×3, ReLU, same padding |
| MaxPooling + BatchNorm | Pool size 2×2 |
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling + BatchNorm | Pool size 2×2 |
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling + BatchNorm | Pool size 2×2 |
| Conv2D | 128 filters, 3×3, ReLU |
| MaxPooling + BatchNorm | Pool size 2×2 |
| Flatten → Dense(128) → Dense(6, softmax) | Classification head |

- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Epochs:** 20 with ModelCheckpoint (best weights saved)

---

## Preprocessing & Augmentation

- Grayscale conversion, rescaling (1/255)
- Width/height shift (±10%), rotation (±10%)
- Horizontal flipping
- 70/30 train/validation split

---

## Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix analysis across all 6 classes
- Grad-CAM heatmap visualization (gradient-weighted class activation mapping)
  to interpret which facial regions drive predictions

---

## Tech Stack
`Python` `TensorFlow/Keras` `OpenCV` `NumPy` `Pandas` `Scikit-learn` 
`Seaborn` `Matplotlib` `tf-explain`

---

## Key Learnings
- Preprocessing decisions (normalization, augmentation) impacted 
  generalization more than architectural depth
- Confusion matrix revealed class-level failures invisible in aggregate accuracy
- Grad-CAM confirmed the model focused on eye/mouth regions for most classes
- BatchNormalization after each pooling layer significantly stabilized training

---

## Project Structure
```
├── facial-expression-recognition.ipynb   # Full pipeline notebook
└── README.md
```

---

## How to Run
1. Download FER-2013 dataset from Kaggle
2. Update `DATA_PATH` in the notebook to your local path
3. Run cells sequentially — preprocessing → training → evaluation → Grad-CAM
