# 🔊 AI-Based Profanity Detection & Automatic Video Censoring System

## 📌 Project Overview

This project presents an intelligent multimedia processing system that automatically detects profanity in video content and replaces offensive words with a beep sound.

The system integrates:

* Automatic Speech Recognition (ASR)
* Natural Language Processing (NLP)
* Deep Learning
* Multimedia Processing
* Model Evaluation Metrics

It provides a comparative study of multiple algorithms across speech and text domains.

---

## 🎯 Objectives

* Convert speech from video into text using ASR models
* Detect profanity using multiple NLP techniques
* Automatically censor offensive words in the video
* Compare performance of different algorithms
* Display evaluation metrics and confusion matrix

---

## 🧠 System Architecture

Video Input
→ Audio Extraction
→ Speech Recognition
→ Word-Level Timestamp Extraction
→ Profanity Detection
→ Evaluation Metrics
→ Audio Replacement (Beep)
→ Censored Video Output

---

## 🗂 Project Structure

```
profanity_beeper/
│
├── app.py
│
├── speech_models/
│   ├── whisper_model.py
│   └── vosk_model.py
│
├── profanity_models/
│   ├── keyword_filter.py
│   ├── tfidf_model.py
│   ├── lstm_model.py
│   └── bert_model.py
│
├── evaluation/
│   └── metrics.py
│
├── censor.py
├── badwords.txt
└── temp/
```

---

## 🔊 Speech Recognition Models

### 1️⃣ Whisper (Transformer-Based ASR)

* Deep Learning model
* High accuracy
* Context-aware transcription
* Provides word-level timestamps

### 2️⃣ Vosk (HMM-DNN Hybrid Model)

* Lightweight and efficient
* Offline speech recognition
* Based on statistical acoustic modeling

Comparison Metric:

* Word Error Rate (WER)
* Processing Time

---

## 🧾 Profanity Detection Models

### 1️⃣ Keyword Matching

* Rule-based approach
* Uses predefined abusive word dictionary
* Fast baseline model

### 2️⃣ TF-IDF + Logistic Regression

* Classical Machine Learning model
* Text vectorization using TF-IDF
* Statistical classification

### 3️⃣ LSTM (Recurrent Neural Network)

* Deep learning sequence model
* Implemented using PyTorch
* Captures contextual dependencies

### 4️⃣ BERT (Transformer-Based Model)

* Pretrained Toxic-BERT model
* Context-aware detection
* Highest semantic understanding

Comparison Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 📊 Evaluation

The system computes:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix Visualization

Evaluation is performed on detected words per transcription.

---

## 💻 User Interface (Streamlit)

Features:

* Upload video file
* Select Speech Recognition model
* Select Profanity Detection model
* View full transcript with highlighted profanity
* View evaluation metrics
* View confusion matrix
* Download censored video

---

## 🛠 Technologies Used

| Component        | Technology                                     |
| ---------------- | ---------------------------------------------- |
| Frontend         | Streamlit                                      |
| ASR              | Whisper, Vosk                                  |
| NLP              | Keyword Matching, TF-IDF, LSTM (PyTorch), BERT |
| Deep Learning    | PyTorch, Transformers                          |
| Video Processing | MoviePy                                        |
| Audio Processing | FFmpeg, PyDub                                  |
| Evaluation       | Scikit-learn                                   |
| Language         | Python                                         |

---

## 🚀 Installation

### Install Dependencies

```
pip install streamlit
pip install openai-whisper
pip install vosk
pip install torch transformers
pip install scikit-learn matplotlib seaborn
pip install moviepy pydub imageio imageio-ffmpeg
```

### Install FFmpeg

Download from:
https://www.gyan.dev/ffmpeg/builds/

Add `C:\ffmpeg\bin` to system PATH.

Verify:

```
ffmpeg -version
```

Download Vosk model from:
https://alphacephei.com/vosk/models
Extract into:
models/vosk-model-small-en-us-0.15


---

## ▶️ Run Application

```
streamlit run app.py
```

---

## 📈 Academic Contribution

This project demonstrates comparative evaluation of:

* Transformer-based models
* Hybrid statistical models
* Classical machine learning techniques
* Deep learning sequence models

It highlights trade-offs between:

* Accuracy
* Computational complexity
* Context awareness
* Processing speed

---

## 🔮 Future Enhancements

* Real-time live audio censorship
* Multilingual profanity detection
* Large-scale dataset evaluation
* Subtitle (.srt) generation
* Deployment as web application

---

## 👨‍💻 Authors

Developed as part of academic coursework in Artificial Intelligence and Machine Learning.

---

## 📜 License

For educational and research purposes only.
