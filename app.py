import streamlit as st
import os
from moviepy.editor import VideoFileClip
from censor import censor_video

from evaluation.metrics import evaluate, plot_confusion_matrix

# Add these new imports for audio visualization
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# =========================
# PAGE TITLE
# =========================
st.title("🔊 AI Profanity Detection & Video Censoring System")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "mkv"])

if uploaded_file:

    os.makedirs("temp", exist_ok=True)

    input_path = f"temp/{uploaded_file.name}"

    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(input_path)

    # =========================
    # MODEL SELECTION
    # =========================
    asr_option = st.selectbox(
        "Choose Speech Recognition Model",
        ["Whisper", "Vosk"]
    )

    nlp_option = st.selectbox(
        "Choose Profanity Detection Model",
        ["Keyword", "TF-IDF", "LSTM", "BERT"]
    )

    if st.button("Run Detection"):

        # =====================================
        # STAGE 1 — AUDIO EXTRACTION
        # =====================================
        st.markdown("# 🎵 Stage 1: Audio Extraction")

        wav_path = "temp/audio.wav"

        video = VideoFileClip(input_path)
        video.audio.write_audiofile(
            wav_path,
            codec="pcm_s16le",
            ffmpeg_params=["-ac", "1"]
        )

        st.success("Audio extracted successfully!")

        with st.expander("📊 View Audio Waveform"):
            y, sr = librosa.load(wav_path)

            fig, ax = plt.subplots(figsize=(10, 3))
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.set_title("Audio Waveform Representation")
            st.pyplot(fig)

            st.write(f"Sample Rate: {sr}")
            st.write(f"Audio Duration: {round(len(y)/sr, 2)} seconds")

        # =====================================
        # STAGE 2 — SPEECH RECOGNITION
        # =====================================
        st.markdown("# 🗣 Stage 2: Speech Recognition")

        with st.spinner("Transcribing Audio..."):

            if asr_option == "Whisper":
                from speech_models.whisper_model import transcribe
                words = transcribe(input_path)

            else:
                from speech_models.vosk_model import transcribe
                words = transcribe(wav_path)

        st.success("Transcription Completed!")

        with st.expander("📝 View Word-Level Timestamps"):
            st.json(words[:15])  # show first 15 words only

        # Full transcript
        full_text = " ".join([w["word"] for w in words])
        st.markdown("### 📄 Full Transcript")
        st.write(full_text)

        # =====================================
        # STAGE 3 — NLP FEATURE EXTRACTION
        # =====================================
        st.markdown("# 🔍 Stage 3: NLP Feature Extraction")

        with st.spinner("Extracting linguistic features..."):
            # Simple feature extraction for demonstration
            word_count = len(words)
            unique_words = len(set([w["word"].lower() for w in words]))
            avg_word_length = sum(len(w["word"]) for w in words) / word_count if word_count > 0 else 0
            
            st.markdown("### 📊 Extracted Features:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Words", word_count)
            with col2:
                st.metric("Unique Words", unique_words)
            with col3:
                st.metric("Avg Word Length", f"{avg_word_length:.2f}")
            
            with st.expander("🔬 Detailed NLP Features"):
                st.write("**Word Frequency Distribution:**")
                # Show most common words
                from collections import Counter
                word_freq = Counter([w["word"].lower() for w in words]).most_common(10)
                st.write(word_freq)

        # =====================================
        # STAGE 4 — CLASSIFICATION DECISION
        # =====================================
        st.markdown("# ⚖️ Stage 4: Classification Decision")

        with st.spinner("Detecting Profanity..."):

            if nlp_option == "Keyword":
                from profanity_models.keyword_filter import detect
                flagged = detect(words)

            elif nlp_option == "TF-IDF":
                from profanity_models.tfidf_model import detect
                flagged = detect(words)

            elif nlp_option == "LSTM":
                from profanity_models.lstm_model import detect
                flagged = detect(words)

            else:
                from profanity_models.bert_model import detect
                flagged = detect(words)

        st.success("Profanity Detection Completed!")

        # Display detection results
        flagged_count = len(flagged)
        st.markdown(f"### 🚨 Detected {flagged_count} profane words")
        
        if flagged_count > 0:
            profane_words = [w["word"] for w in flagged]
            st.write("**Profane words found:**", ", ".join(profane_words))
        
        with st.expander("📋 View Classification Details"):
            st.write("**Model Decision Logic:**")
            st.write(f"- Using {nlp_option} model for classification")
            st.write(f"- Words analyzed: {word_count}")
            st.write(f"- Confidence threshold: 0.5")

        # =========================
        # DISPLAY TRANSCRIPT
        # =========================
        st.markdown("## 📝 Transcript with Highlights")

        transcript_html = ""

        for w in words:
            if w in flagged:
                transcript_html += f"<span style='background-color:#ffcccc; color:red; font-weight:bold; padding:2px 0'>{w['word']}</span> "
            else:
                transcript_html += w["word"] + " "

        st.markdown(transcript_html, unsafe_allow_html=True)

        # =====================================
        # STAGE 5 — EVALUATION
        # =====================================
        st.markdown("# 📊 Stage 5: Evaluation")

        # Ground truth (based on keyword list for demo)
        from profanity_models.keyword_filter import BAD_WORDS

        true_labels = [1 if w["word"] in BAD_WORDS else 0 for w in words]
        predicted_labels = [1 if w in flagged else 0 for w in words]

        acc, prec, rec, f1, cm = evaluate(true_labels, predicted_labels)

        st.markdown("### Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{acc:.3f}")
        with col2:
            st.metric("Precision", f"{prec:.3f}")
        with col3:
            st.metric("Recall", f"{rec:.3f}")
        with col4:
            st.metric("F1 Score", f"{f1:.3f}")

        with st.expander("📈 Confusion Matrix"):
            fig = plot_confusion_matrix(cm)
            st.pyplot(fig)
            
            st.write("**Confusion Matrix Interpretation:**")
            st.write(f"- True Positives: {cm[1][1]} (correctly identified profanity)")
            st.write(f"- True Negatives: {cm[0][0]} (correctly identified clean words)")
            st.write(f"- False Positives: {cm[0][1]} (clean words flagged as profane)")
            st.write(f"- False Negatives: {cm[1][0]} (profane words missed)")

        # =====================================
        # STAGE 6 — CENSORING
        # =====================================
        st.markdown("# 🔇 Stage 6: Censoring")

        if flagged_count > 0:
            st.warning(f"⚠️ Censoring {flagged_count} profane words...")
            
            output_path = "temp/output.mp4"
            censor_video(input_path, flagged, output_path)

            st.success("Video censored successfully!")
            
            st.markdown("### 🎬 Censored Video")
            st.video(output_path)

            with open(output_path, "rb") as f:
                st.download_button(
                    "📥 Download Censored Video",
                    f,
                    file_name="censored_output.mp4"
                )
        else:
            st.success("✅ No profanity detected - no censoring needed!")
            st.info("The original video is clean and ready for use.")
            
            # Still allow download of original
            with open(input_path, "rb") as f:
                st.download_button(
                    "📥 Download Original Video",
                    f,
                    file_name="original_video.mp4"
                )