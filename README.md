# Emotion Classification from Audio (Speech and Song)

This project builds a **complete system to detect human emotions from audio files**.
It can process both **speech and song recordings** and automatically predict the emotion expressed in the audio.

The system is built using **machine learning (XGBoost)** and supports real-time predictions through a **Streamlit web application**.

---

##  Project Purpose

The purpose of this project is to:

* Develop an **end-to-end emotion recognition system** from audio.
* Process both **speech and song** audio files to classify emotions.
* Build a **user-friendly web app** where users can upload `.wav` files and instantly see the predicted emotion.
* Improve classification accuracy by carefully **removing confusing emotion classes** (like surprise and neutral) that reduce the model’s performance.
* Visualize audio properties (waveform, spectrogram, MFCC) to better understand how different emotions sound.

---

##  Project Objectives

* Extract **emotion-specific features** from audio files using MFCC.
* Classify audio into **one of several emotion categories** (angry, calm, disgust, happy, sad, unknown).
* Build a practical and simple **web interface using Streamlit**.
* Deploy the model for real-time audio emotion classification.

---

##  Key Features

*  **High Accuracy:** Achieves around **78% accuracy** after optimizing class selection.
*  **Speech and Song Compatible:** Works with both types of audio.
*  **.wav Audio Support:** Designed to process `.wav` files.
*  **MFCC-Based Feature Extraction:** Extracts the most emotion-relevant features.
*  **XGBoost Model:** Fast, powerful, and highly efficient.
*  **Streamlit App:** Clean, browser-based user interface.
*  **Ready for Deployment:** Easily run the app and get predictions.

---

##  Dataset

* The dataset consists of **speech and song audio files** stored in the folder:

```text
Audio_song_speech
```

* Audio files are labeled with **different emotions based on their file names.**
* Emotions covered: *angry, calm, disgust, fearful, happy, sad, surprise, neutral*
  (Later optimized to drop surprise and neutral for better accuracy.)

---

## Installation & Running Instructions

### Step 1: Install Required Libraries

```bash
pip install streamlit librosa scikit-learn xgboost numpy seaborn matplotlib
```

### Step 2: Run the Streamlit Web App

```bash
streamlit run Deployed_code.py
```

---

##  How to Use the Web App

1. Open the web app in your browser.
2. Upload a `.wav` audio file.
3. The system will **extract features automatically**.
4. It will display the predicted emotion on the screen.

---

##  Model Details

| Feature         | Details                               |
| --------------- | ------------------------------------- |
| Model           | XGBoostClassifier                     |
| Input Features  | 40 MFCC features                      |
| Dataset         | Audio\_song\_speech                   |
| Removed Classes | Surprise, Neutral (merged or dropped) |
| Model File      | xgb\_model\_audio.pkl                 |
| Label Encoder   | label\_encoder\_audio.pkl             |

---

##  Project Structure

```text
├──  aap.py              # Streamlit web app
├── xgb_model_audio.pkl           # Trained model file
├── label_encoder_audio.pkl       # Saved label encoder
├── audio.ipynb                   # Model training & evaluation notebook
└── README.md                     # Project documentation
```

---

##  Future Enhancements

* Support for **real-time audio streaming**.
* Integration of **deep learning models (CNN/RNN)** for improved accuracy.
* **Multilingual emotion detection** from audio.
* Add more advanced **visualizations (waveform, spectrograms)** in the web app.

---
 
 
