# 🔊 AI Profanity Detection & Video Censoring System

An AI-powered application that automatically detects profanity in videos and replaces abusive words with a beep sound.

The system processes multimedia content using **speech recognition + natural language processing + audio editing** to generate a clean censored video.

---

## 🎯 Objective

To automatically identify offensive language in video content and censor it without manual editing by:

1. Extracting audio from video
2. Converting speech → text with timestamps
3. Detecting profanity
4. Replacing offensive segments with beep sound
5. Generating a censored video output

---

## 🧠 System Pipeline

Video Input
→ Audio Extraction
→ Speech Recognition (Whisper)
→ Word Timestamp Detection
→ Profanity Identification
→ Audio Replacement (Beep)
→ Censored Video Output

---

## 🚀 Features

* Upload any video file
* Automatic speech transcription
* Detects abusive words
* Replaces them with beep sound
* Download censored video
* Streamlit interactive UI

---

## 🛠 Technologies Used

| Module             | Technology              |
| ------------------ | ----------------------- |
| Frontend           | Streamlit               |
| Speech Recognition | OpenAI Whisper          |
| Audio Processing   | FFmpeg, PyDub           |
| Video Processing   | MoviePy                 |
| NLP                | Keyword-based filtering |
| Language           | Python                  |

---

## 📁 Project Structure

```
profanity_beeper/
│
├── app.py                  # Streamlit UI
├── whisper_transcribe.py   # Speech recognition
├── censor.py               # Beep replacement logic
├── badwords.txt            # List of abusive words
├── temp/                   # Temporary files
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```
git clone <your-repo-link>
cd profanity_beeper
```

### 2️⃣ Install Dependencies

```
pip install streamlit
pip install openai-whisper
pip install moviepy==1.0.3
pip install pydub
pip install numpy pandas
pip install imageio imageio-ffmpeg
```

### 3️⃣ Install FFmpeg (Important)

Download: https://www.gyan.dev/ffmpeg/builds/

Extract and add to PATH:

```
C:\ffmpeg\bin
```

Verify:

```
ffmpeg -version
```

---

## ▶️ Run Application

```
streamlit run app.py
```

Upload a video and click **Censor Video**

---

## 📊 Current Algorithm

Speech Recognition → Whisper (Transformer Model)
Profanity Detection → Keyword Matching

---

## 📌 Future Enhancements

* Add Vosk speech recognition comparison
* Add BERT toxicity classifier
* Display detected words in UI
* Calculate accuracy metrics (WER, Precision, Recall, F1)
* Support multiple languages
* Real-time live censoring

---

## 📚 Academic Contribution

This project demonstrates integration of:

* Automatic Speech Recognition (ASR)
* Natural Language Processing (NLP)
* Multimedia Processing
* Human-Computer Interaction

It provides an automated alternative to manual video censorship systems used in broadcasting and streaming platforms.

---

## 👨‍💻 Authors

Project developed as part of academic coursework in Artificial Intelligence / Machine Learning.

---

## 📜 License

This project is for educational purposes only.
