import streamlit as st
import os
from moviepy.editor import VideoFileClip
from censor import censor_video

from evaluation.metrics import evaluate, plot_confusion_matrix

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

        # =========================
        # AUDIO CONVERSION (FOR VOSK)
        # =========================
        wav_path = "temp/audio.wav"

        video = VideoFileClip(input_path)
        video.audio.write_audiofile(
            wav_path,
            codec="pcm_s16le",
            ffmpeg_params=["-ac", "1"]
        )

        # =========================
        # LOAD ASR MODEL
        # =========================
        with st.spinner("Transcribing Audio..."):

            if asr_option == "Whisper":
                from speech_models.whisper_model import transcribe
                words = transcribe(input_path)

            else:
                from speech_models.vosk_model import transcribe
                words = transcribe(wav_path)

        st.success("Transcription Completed!")

        # =========================
        # LOAD NLP MODEL
        # =========================
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

        # =========================
        # DISPLAY TRANSCRIPT
        # =========================
        st.markdown("## 📝 Transcript")

        transcript_html = ""

        for w in words:
            if w in flagged:
                transcript_html += f"<span style='color:red; font-weight:bold'>{w['word']}</span> "
            else:
                transcript_html += w["word"] + " "

        st.markdown(transcript_html, unsafe_allow_html=True)

        # =========================
        # EVALUATION METRICS
        # =========================
        st.markdown("## 📊 Evaluation Metrics")

        # Ground truth (based on keyword list for demo)
        from profanity_models.keyword_filter import BAD_WORDS

        true_labels = [1 if w["word"] in BAD_WORDS else 0 for w in words]
        predicted_labels = [1 if w in flagged else 0 for w in words]

        acc, prec, rec, f1, cm = evaluate(true_labels, predicted_labels)

        st.write(f"Accuracy: {acc:.3f}")
        st.write(f"Precision: {prec:.3f}")
        st.write(f"Recall: {rec:.3f}")
        st.write(f"F1 Score: {f1:.3f}")

        fig = plot_confusion_matrix(cm)
        st.pyplot(fig)

        # =========================
        # CENSOR VIDEO
        # =========================
        st.markdown("## 🎬 Censored Video")

        output_path = "temp/output.mp4"
        censor_video(input_path, flagged, output_path)

        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button(
                "Download Censored Video",
                f,
                file_name="censored_output.mp4"
            )
