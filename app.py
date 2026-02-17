import streamlit as st
import os
from whisper_transcibe import transcribe_audio
from censor import censor_video

st.title("🔊 AI Profanity Video Censor")

uploaded_file = st.file_uploader("Upload Video", type=["mp4","mov","mkv"])

if uploaded_file:
    input_path = f"temp/{uploaded_file.name}"
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(input_path)

    if st.button("Censor Video"):

        with st.spinner("Transcribing audio..."):
            words = transcribe_audio(input_path)

        st.success("Transcription Done!")

        with st.spinner("Beeping bad words..."):
            output_path = "temp/output.mp4"
            censor_video(input_path, words, output_path)

        st.success("Done!")

        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button("Download Censored Video", f, file_name="censored.mp4")
