import streamlit as st
import os

from speech_models.whisper_model import transcribe
from profanity_models.keyword_filter import detect as keyword_detect
from profanity_models.tfidf_model import detect as tfidf_detect
from censor import censor_video

st.title("🔊 AI Profanity Video Censor")

uploaded_file = st.file_uploader("Upload Video", type=["mp4","mov","mkv"])

if uploaded_file:
    input_path = f"temp/{uploaded_file.name}"
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(input_path)

    nlp_option = st.selectbox(
        "Choose Profanity Detection Algorithm",
        ["Keyword Matching", "TF-IDF + Logistic Regression"]
    )

    if st.button("Censor Video"):

        with st.spinner("Transcribing..."):
            words = transcribe(input_path)

        if nlp_option == "Keyword Matching":
            flagged = keyword_detect(words)
        else:
            flagged = tfidf_detect(words)

        st.write("Detected Profanity Words:")
        st.write(flagged)

        with st.spinner("Censoring..."):
            output_path = "temp/output.mp4"
            censor_video(input_path, flagged, output_path)

        st.video(output_path)
