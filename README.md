# ECG Arrhythmia Classification using Deep Learning

A deep learning system that classifies individual heartbeats from ECG signals into **8 arrhythmia classes**, with **Grad-CAM explanations** that show which part of the beat drove each prediction, and a **Streamlit web app** for uploading ECG data and viewing results.

Trained and evaluated on the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) (PhysioNet).

## Results

Evaluated on **19,360 held-out beats**:

| Metric | Score |
|---|---|
| Accuracy | **97.1%** |
| Macro precision | 80.5% |
| Macro recall | 91.3% |
| Macro F1 | 84.9% |

Per-class performance:

| Class | Beat type | Precision | Recall | F1 | Test beats |
|---|---|---|---|---|---|
| N | Normal beat | 99.5% | 97.2% | 98.3% | 14,553 |
| L | Left bundle branch block beat | 99.2% | 99.0% | 99.1% | 1,323 |
| R | Right bundle branch block beat | 98.8% | 99.0% | 98.9% | 1,434 |
| V | Premature ventricular contraction | 91.6% | 96.4% | 93.9% | 1,356 |
| A | Atrial premature beat | 70.7% | 90.8% | 79.5% | 487 |
| a | Aberrated atrial premature beat | 59.5% | 83.3% | 69.4% | 30 |
| J | Nodal (junctional) premature beat | 72.2% | 76.5% | 74.3% | 17 |
| F | Fusion of ventricular and normal beat | 52.4% | 88.1% | 65.7% | 160 |

Accuracy is dominated by the large Normal class, so macro metrics are the better summary of how well the model handles rare arrhythmias. Most errors are normal beats confused with atrial, fusion or ventricular beats.

![Confusion matrix](ECG_Project%20-%20new/eval_outputs/confusion_matrix.png)

## Explainability (Grad-CAM)

Grad-CAM highlights the regions of the 360-sample beat that influenced the prediction, so a result can be checked against the waveform instead of being a black box.

![Grad-CAM example](ECG_Project%20-%20new/explain_examples/example_386.png)

## How it works

1. **Data:** MIT-BIH records are downloaded from PhysioNet with the `wfdb` library (42 records).
2. **Preprocessing:** wavelet denoising and robust scaling of the raw signal.
3. **Segmentation:** a 360-sample window is cut around each annotated R-peak, and each beat takes its label from the annotation.
4. **Class imbalance:** SMOTE oversampling on the training set only (the test set stays untouched).
5. **Model:** a 1D CNN with a dilated convolution block, batch normalization, residual connections and multi-head attention, followed by global average pooling and a dense softmax layer over 8 classes.
6. **Training:** checkpointing, early stopping and learning-rate reduction callbacks.
7. **Explainability:** Grad-CAM heatmaps over the input beat.
8. **App:** a Streamlit app that detects R-peaks in an uploaded trace, classifies each beat, and shows probabilities and Grad-CAM heatmaps.

## Project structure

All code lives in the `ECG_Project - new/` folder:

```
.
├── step1_visualize.py            # Download a record and plot signal + annotations
├── step2_preprocess.py           # Wavelet denoising and scaling
├── step3_segmentation.py         # R-peak beat segmentation
├── step4_build_dataset.py        # Build X/y beat dataset from MIT-BIH records
├── step5_make_balanced_data.py   # Train/test split and SMOTE on training data
├── step6_train_model.py          # Build and train the CNN + attention model
├── evaluate_model.py             # Confusion matrix, classification report, errors
├── step7_explainability.py       # Grad-CAM examples
├── step8_generate_test_files.py  # Export sample beats as CSVs for the app
├── app.py                        # Streamlit web app
├── best_model.h5                 # Trained model (best checkpoint)
├── sample_ecgs_for_upload/       # Example CSVs to try in the app
├── eval_outputs/                 # Confusion matrix and misclassified-beat summary
└── explain_examples/             # Saved Grad-CAM figures
```

## Getting started

**1. Clone and install**

```bash
git clone https://github.com/bharathreddy37040/ECG-Arrhythmia-Detection-with-Web-interface-and-Deep-Learning.git
cd ECG-Arrhythmia-Detection-with-Web-interface-and-Deep-Learning/"ECG_Project - new"
python -m venv venv
venv\Scripts\activate        # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

**2. Run the app with the included model**

```bash
streamlit run app.py
```

Upload any file from `sample_ecgs_for_upload/`, or your own CSV:

- a single column of amplitude values (one per line, optional header),
- either a **single beat of exactly 360 samples**, or a longer trace (several seconds) from which the app detects R-peaks and classifies each beat.

**3. Reproduce training (optional)**

Run the scripts in order. The data is downloaded automatically from PhysioNet the first time.

```bash
python step4_build_dataset.py
python step5_make_balanced_data.py
python step6_train_model.py
python evaluate_model.py
python step7_explainability.py
python step8_generate_test_files.py
```

Steps 1 to 3 are exploratory scripts for visualizing, denoising and segmenting a single record.

## Tech stack

Python, TensorFlow / Keras, NumPy, Pandas, scikit-learn, imbalanced-learn (SMOTE), scikit-image and PyWavelets (denoising), SciPy (peak detection), wfdb, OpenCV, Matplotlib, Seaborn, Streamlit.

## Limitations and future work

- **Beat-level split:** the train/test split is random across beats, so beats from the same patient can appear in both sets. A **patient-wise split** (splitting by record) would give a stricter estimate of how the model generalizes to new patients, and is the main planned improvement.
- **Validation:** a separate validation set, distinct from the test set, should be used for model selection.
- **Rare classes:** precision for Fusion (F) and Aberrated atrial (a) beats is lower than for the major classes. Class weighting or more data could help.
- **Not a medical device:** this is a learning and research project and must not be used for diagnosis.

## Dataset and references

- Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.* IEEE Engineering in Medicine and Biology Magazine, 2001.
- Goldberger AL et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation, 2000.

## Author

**Bharath Reddy Bandi** · [LinkedIn](https://www.linkedin.com/in/bharathreddy37040/) · [GitHub](https://github.com/bharathreddy37040)
