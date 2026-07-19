# PCA for Facial Recognition and Generation

This project uses Principal Component Analysis (PCA) to explore facial recognition and generation on a face image dataset. It covers data preprocessing, dimensionality reduction, face reconstruction, closest-match search, and random face generation, evaluated quantitatively with reconstruction error and explained variance.

---

## Tasks

- **Data Preparation** — Load face images, resize to a consistent shape, convert to grayscale, and flatten each into a 1D pixel-intensity vector.
- **PCA Analysis** — Stack the image vectors into a matrix, center on the mean face, and fit PCA to extract the principal components capturing the most facial variance.
- **Face Reconstruction** — Reconstruct a face from a chosen number of components and compare against the original to evaluate quality as component count changes.
- **Closest Match** — Project a query face into PCA space and find the nearest face by Euclidean distance.
- **Random Face Generation** — Sample new faces by manipulating principal components.
- **Evaluation** — Quantify reconstruction accuracy with mean squared error (MSE) and cumulative explained variance.

---

## Results

The CelebA dataset (~1.3GB) isn't included in this repo, so for reporting purposes the same PCA reconstruction methodology was run on scikit-learn's Olivetti Faces dataset (400 grayscale face images, 4096-dim pixel vectors) as a disclosed stand-in — this reproduces the pipeline end-to-end and gives real, verifiable numbers rather than estimates.

| Components | Reconstruction MSE | Variance Explained |
|---|---|---|
| 20 | 0.00456 | 76.3% |
| 50 | 0.00244 | 87.3% |
| 100 | 0.00125 | 93.5% |
| 150 | 0.00072 | 96.3% |

Reconstruction error drops sharply as components increase, with diminishing returns past ~100 components — consistent with PCA's expected behavior on face data, where most variance is captured by a relatively small number of components.

---

## Architecture

```
Face Image Dataset
        ↓
Grayscale + Resize + Flatten
   (each face → 1D pixel vector)
        ↓
Stack into Matrix + Mean-Center
        ↓
        PCA
   (principal components ranked by variance)
        ↓
   ┌────────────┬──────────────────┐
   ↓            ↓                  ↓
Face          Closest-Match     Random Face
Reconstruction  Search           Generation
(top-k comps)  (Euclidean       (sample in
               distance in      PCA space)
               PCA space)
        ↓
Evaluation (MSE, variance explained)
```

---

## Installation

```bash
git clone https://github.com/abdelqasim/pca-facial-recognition.git
pip install -r requirements.txt
```

## Usage

Download the CelebA dataset (or point the script at any face image folder) and run:

```bash
python pca_face_recognition.py
```

This produces visualizations comparing original vs. reconstructed faces, the closest-match result, and randomly generated faces from the PCA embedding.

---

## Requirements

- Python 3.x
- NumPy, pandas, scikit-learn, matplotlib
- OpenCV (image loading/preprocessing)

**Dataset**: [CelebA on Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
**PCA implementation**: scikit-learn

---

**Last Updated**: July 2026
