# Egyptian Sign Language (ESL) Real-Time Recognition System

## Project Overview

This project presents a **real-time Egyptian Sign Language (ESL) recognition system** that translates sign language gestures into text using a webcam.
The system combines **hand gesture recognition** and **facial emotion detection** to better capture the linguistic and expressive components of ESL.

The project is designed as a **graduation-level prototype**, focusing on feasibility, clarity, and real-time performance rather than large-scale deployment.

---

## Objectives

* Recognize **word-level Egyptian Sign Language signs** from live webcam input
* Integrate **facial emotion detection** as a non-manual linguistic cue
* Run **in real time** on a local machine
* Use **pre-trained models and lightweight classifiers** (no training from scratch)
* Deliver a **working demo application** suitable for academic evaluation

---

## System Architecture (High Level)

```
Webcam Input
     ↓
Frame Capture (OpenCV / Streamlit)
     ↓
MediaPipe Holistic
(Hand + Face Landmarks)
     ↓
Feature Extraction
     ↓
Sign Classifier (MLP / LSTM)
     ↓
Predicted Word
     ↓
Emotion Detection (DeepFace)
     ↓
Live Display (Text Overlay)
```

---

## Dataset

* **Dataset:** Egyptian Sign Language (ESL)
* **Structure:**

  * Word-level classes
  * Multiple videos per word
* **Usage:**

  * Videos are split into frames
  * Hand landmarks are extracted using MediaPipe
  * Facial emotion is inferred using a pre-trained model

Dataset link:
[https://data.mendeley.com/datasets/39tbt2jd7r/2](https://data.mendeley.com/datasets/39tbt2jd7r/2)

> ⚠️ Note: The dataset contains **sign labels only** (no facial expression labels).
> Facial emotion is inferred using pre-trained emotion models.

---

## Tech Stack

### Core Technologies

* **Python 3.9+**
* **OpenCV** – webcam capture & visualization
* **MediaPipe** – hand & face landmark detection
* **TensorFlow / Keras** – sign classification model
* **DeepFace** – facial emotion recognition
* **NumPy / Pandas** – data processing

### Demo Interface (not decided yet)

---

## Project Structure

```
esl-project/
├─ data/                      # metadata + small test videos (NOT the whole dataset)
├─ notebooks/
│  └─ train.ipynb
├─ src/
│  ├─ inference.py            # load model, predict_frame(), predict_sequence()
│  ├─ train_utils.py
│  └─ mediapipe_utils.py      # wrapper for holistic -> landmarks
├─ artifacts/
│  └─ model.h5
├─ results/
│  └─ val_results.csv
├─ requirements.txt         # direct file for utilities download
└─ README.md
```

---

## Installation and How to contribute

### 1. Clone the repository

```bash
git clone https://github.com/your-username/esl-sign-language-project.git
cd esl-sign-language-project
```
### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Demo

### Option A — Streamlit App (Recommended)

```bash
streamlit run app.py
```
* Opens a browser window
* Uses your webcam
* Displays:
  * Predicted ESL word
  * Facial emotion
  * Confidence score
---
### Option B — OpenCV Desktop Demo

```bash
python demo_opencv.py
```
* Opens a desktop window
* Press **Q** to exit

---

## Model Training (Optional)

Model training is performed in **Google Colab** using the notebook:

```
notebooks/train_model.ipynb
```

Steps:

1. Extract frames from videos
2. Extract MediaPipe landmarks
3. Train a lightweight classifier (MLP / LSTM)
4. Export model to `artifacts/model.h5.`

> Training is **not required** to run the demo if `model.h5` is already provided.

---

## 📊 Evaluation

* Accuracy evaluated on a held-out validation set
* Metrics:

  * Overall accuracy
  * Per-class accuracy
* Emotion detection is qualitative (no labeled ground truth)

---

## Limitations

* Limited vocabulary (word-level only)
* Performance affected by:

  * Lighting conditions
  * Camera angle
  * Distance from webcam
* Facial emotion is **inferred**, not trained on ESL-specific labels
* Not intended for production or public deployment

---

## Future Work

* Expand vocabulary size
* Sentence-level translation
* Better temporal modeling (Transformer-based)
* Mobile application
* Improved emotion modeling specific to ESL grammar

---

## Team

* Graduation Project — Faculty of Engineering
* Team Leader: *Abdullah Sherdy*
* Team Members: Sara Yasser, Mena Ali, Basel Hamdi, Waad Waled. 
* Year: *2025/2026*

---

## License

This project is for **academic and educational use only** under the MIT License.
