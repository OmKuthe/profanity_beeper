import streamlit as st
import os
from moviepy.editor import VideoFileClip
from censor import censor_video

from evaluation.metrics import evaluate, plot_confusion_matrix

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# =========================
# PAGE TITLE
# =========================
st.title("🔊 AI Profanity Detection & Video Censoring System")
st.markdown("### *Understanding Machine Learning Stages through Video Censoring*")

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

    if st.button("Run Detection - See Full ML Pipeline"):

        # =====================================
        # STAGE 1 — AUDIO EXTRACTION
        # =====================================
        st.markdown("---")
        st.markdown("# 🎵 Stage 1: Data Acquisition")
        st.markdown("**ML Concept:** Raw data collection - getting the initial input")
        
        wav_path = "temp/audio.wav"

        with st.spinner("Extracting audio from video..."):
            video = VideoFileClip(input_path)
            video.audio.write_audiofile(
                wav_path,
                codec="pcm_s16le",
                ffmpeg_params=["-ac", "1"]
            )

        st.success("✅ Audio extracted successfully!")

        with st.expander("📊 View Raw Audio Data (Waveform)"):
            y, sr = librosa.load(wav_path)

            fig, ax = plt.subplots(figsize=(10, 3))
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.set_title("Raw Audio Signal - Unprocessed Data")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)

            st.markdown("**📝 What's happening here?**")
            st.markdown("- We're collecting raw audio data from the video file")
            st.markdown("- This is the **first stage of any ML pipeline**: getting your data")
            st.markdown(f"- Sample Rate: {sr} Hz (quality of audio)")
            st.markdown(f"- Duration: {round(len(y)/sr, 2)} seconds")

        # =====================================
        # STAGE 2 — SPEECH RECOGNITION
        # =====================================
        st.markdown("---")
        st.markdown("# 🗣 Stage 2: Data Preprocessing")
        st.markdown("**ML Concept:** Converting raw data into a usable format")

        with st.spinner("Converting speech to text..."):

            if asr_option == "Whisper":
                from speech_models.whisper_model import transcribe
                words = transcribe(input_path)
            else:
                from speech_models.vosk_model import transcribe
                words = transcribe(wav_path)

        st.success("✅ Audio converted to text!")

        with st.expander("📝 View Raw Transcription Output"):
            st.json(words[:10])  # show first 10 words only
            st.markdown("**📝 What's happening here?**")
            st.markdown("- Converting audio waveforms to text is a form of preprocessing")
            st.markdown("- The speech recognition model transforms non-structured data (audio) into structured data (text)")

        # Full transcript
        full_text = " ".join([w["word"] for w in words])
        st.markdown("### 📄 Raw Transcript")
        st.write(full_text)

        # =====================================
        # STAGE 3 — TEXT PREPROCESSING (NEW - Tokenization & Lemmatization)
        # =====================================
        st.markdown("---")
        st.markdown("# ✂️ Stage 3: Text Preprocessing - Tokenization & Lemmatization")
        st.markdown("**ML Concept:** Cleaning and preparing text data for the model")

        with st.expander("📚 Understanding Tokenization and Lemmatization"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔪 Tokenization")
                st.markdown("**What it is:** Splitting text into individual words (tokens)")
                st.markdown("**Why it matters:**")
                st.markdown("- Computers need discrete units to process")
                st.markdown("- Creates the building blocks for analysis")
                st.markdown("- Example: 'I love movies' → ['I', 'love', 'movies']")
            
            with col2:
                st.markdown("### 🌿 Lemmatization")
                st.markdown("**What it is:** Reducing words to their base/dictionary form")
                st.markdown("**Why it matters:**")
                st.markdown("- Groups related words together")
                st.markdown("- Reduces vocabulary size")
                st.markdown("- Example: 'running', 'ran', 'runs' → 'run'")

        with st.spinner("Applying tokenization and lemmatization..."):
            
            # Step 3.1: Basic Tokenization (already have words from ASR)
            st.markdown("### Step 3.1: Tokenization")
            st.markdown("*Splitting the transcript into individual word tokens*")
            
            # Display tokenization process
            sample_text = full_text[:200] + "..." if len(full_text) > 200 else full_text
            st.markdown(f"**Input text:** \"{sample_text}\"")
            
            tokens = [w["word"] for w in words]
            st.markdown(f"**Tokens generated:** {len(tokens)} individual words")
            
            with st.expander("🔍 View first 20 tokens"):
                st.write(tokens[:20])
            
            # Step 3.2: Lemmatization
            st.markdown("### Step 3.2: Lemmatization")
            st.markdown("*Reducing words to their base form*")
            
            # Import NLTK for lemmatization
            import nltk
            from nltk.stem import WordNetLemmatizer
            from nltk.corpus import wordnet
            
            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                with st.spinner("Downloading NLTK data (first time only)..."):
                    nltk.download('punkt')
                    nltk.download('wordnet')
                    nltk.download('averaged_perceptron_tagger')
            
            # Initialize lemmatizer
            lemmatizer = WordNetLemmatizer()
            
            # Function to get wordnet POS tag
            def get_wordnet_pos(word):
                """Map POS tag to first character lemmatize() accepts"""
                tag = nltk.pos_tag([word])[0][1][0].upper()
                tag_dict = {"J": wordnet.ADJ,
                           "N": wordnet.NOUN,
                           "V": wordnet.VERB,
                           "R": wordnet.ADV}
                return tag_dict.get(tag, wordnet.NOUN)
            
            # Apply lemmatization to sample words
            st.markdown("**Lemmatization Examples:**")
            
            # Create sample words that demonstrate lemmatization
            example_words = [
                ("running", "running → run (verb base form)"),
                ("better", "better → good (adjective base form)"),
                ("mice", "mice → mouse (noun base form)"),
                ("was", "was → be (verb base form)"),
                ("studies", "studies → study (noun/verb base form)")
            ]
            
            example_df = []
            for word, explanation in example_words:
                pos = get_wordnet_pos(word)
                lemma = lemmatizer.lemmatize(word, pos)
                example_df.append({
                    "Original Word": word,
                    "POS Tag": pos,
                    "Lemmatized Form": lemma,
                    "Explanation": explanation
                })
            
            st.table(example_df)
            
            # Apply lemmatization to actual tokens from video
            st.markdown("**Applying to your video transcript:**")
            
            # Take first 20 words for demonstration
            sample_tokens = tokens[:20]
            lemmatized_samples = []
            
            for token in sample_tokens:
                if token.isalpha():  # Only lemmatize alphabetic words
                    pos = get_wordnet_pos(token.lower())
                    lemma = lemmatizer.lemmatize(token.lower(), pos)
                    if token.lower() != lemma:  # Only show changed words
                        lemmatized_samples.append({
                            "Original": token,
                            "Lemmatized": lemma,
                            "Change": "Yes" if token.lower() != lemma else "No"
                        })
            
            if lemmatized_samples:
                st.table(lemmatized_samples)
                st.markdown(f"**Found {len(lemmatized_samples)} words that were reduced to their base form**")
            else:
                st.info("No words in this sample needed lemmatization (most were already in base form)")
            
            # Step 3.3: Create preprocessed corpus
            st.markdown("### Step 3.3: Creating Preprocessed Corpus")
            st.markdown("*Applying lemmatization to all words for final processing*")
            
            # Create a dictionary of original -> lemmatized for all words
            lemmatized_words = []
            word_mapping = {}
            
            for w in words:
                original = w["word"].lower()
                if original.isalpha():
                    pos = get_wordnet_pos(original)
                    lemma = lemmatizer.lemmatize(original, pos)
                else:
                    lemma = original
                
                lemmatized_words.append({
                    "original": w["word"],
                    "lemma": lemma,
                    "start": w["start"],
                    "end": w["end"]
                })
                word_mapping[original] = lemma
            
            st.success(f"✅ Created preprocessed corpus with {len(lemmatized_words)} lemmatized tokens")

        # =====================================
        # STAGE 4 — NLP FEATURE EXTRACTION
        # =====================================
        st.markdown("---")
        st.markdown("# 🔍 Stage 4: Feature Extraction")
        st.markdown("**ML Concept:** Converting text into numerical features the model can understand")

        with st.spinner("Extracting linguistic features..."):
            
            # Using lemmatized words for feature extraction
            lemmatized_text = [w["lemma"] for w in lemmatized_words]
            
            st.markdown("### 📊 Feature Engineering Steps:")
            
            # Feature 1: Word Count
            word_count = len(lemmatized_text)
            st.markdown(f"**1. Basic Statistics:**")
            st.markdown(f"   - Total words: {word_count}")
            
            # Feature 2: Vocabulary Size
            unique_words = len(set(lemmatized_text))
            st.markdown(f"   - Unique words (after lemmatization): {unique_words}")
            st.markdown(f"   - Vocabulary reduction: {len(set(tokens)) - unique_words} words grouped")
            
            # Feature 3: Average word length
            avg_word_length = sum(len(w) for w in lemmatized_text) / word_count if word_count > 0 else 0
            st.markdown(f"   - Average word length: {avg_word_length:.2f} characters")
            
            # Feature 4: TF-IDF explanation
            st.markdown("**2. Advanced Features (TF-IDF):**")
            st.markdown("   - **Term Frequency (TF):** How often a word appears in this document")
            st.markdown("   - **Inverse Document Frequency (IDF):** How rare a word is across all documents")
            st.markdown("   - **TF-IDF Score = TF × IDF** (higher score = more important to this specific document)")
            
            # Show word frequency distribution
            with st.expander("📈 Word Frequency Distribution"):
                from collections import Counter
                
                # Compare raw vs lemmatized frequency
                raw_freq = Counter([w["word"].lower() for w in words]).most_common(15)
                lemma_freq = Counter(lemmatized_text).most_common(15)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Raw Tokens:**")
                    for word, count in raw_freq:
                        st.markdown(f"- {word}: {count}")
                
                with col2:
                    st.markdown("**Lemmatized Tokens:**")
                    for word, count in lemma_freq:
                        st.markdown(f"- {word}: {count}")
                
                st.markdown("**💡 Notice:** Different forms of the same word are grouped together after lemmatization!")

        # =====================================
        # STAGE 5 — CLASSIFICATION DECISION
        # =====================================
        st.markdown("---")
        st.markdown("# ⚖️ Stage 5: Model Training & Classification")
        st.markdown("**ML Concept:** Using algorithms to make predictions based on features")

        with st.spinner("Running profanity detection model..."):

            # Pass lemmatized words to detection models
            if nlp_option == "Keyword":
                from profanity_models.keyword_filter import detect
                # Update keyword filter to work with lemmas
                flagged = detect(lemmatized_words, use_lemmas=True)
                
            elif nlp_option == "TF-IDF":
                from profanity_models.tfidf_model import detect
                flagged = detect(lemmatized_words)
                
            elif nlp_option == "LSTM":
                from profanity_models.lstm_model import detect
                flagged = detect(lemmatized_words)
                
            else:  # BERT
                from profanity_models.bert_model import detect
                flagged = detect(lemmatized_words)

        st.success("✅ Classification completed!")

        flagged_count = len(flagged)

        # Display detection results
        st.markdown(f"### 🚨 Detected {flagged_count} profane words")
        
        # Show which words were flagged (with their original form)
        if flagged_count > 0:
            profane_words = [f"{w['original']} → {w['lemma']}" if w['original'].lower() != w['lemma'] else w['original'] 
                           for w in flagged]
            st.write("**Profane words found (original → lemma):**")
            for word in profane_words:
                st.markdown(f"- {word}")

        with st.expander("📋 Model Decision Explanation"):
            st.markdown("**How the model makes decisions:**")
            st.markdown(f"1. **Input:** Preprocessed text with {len(lemmatized_text)} tokens")
            st.markdown(f"2. **Feature Vector:** Each word converted to numerical features")
            st.markdown(f"3. **Classification:** {nlp_option} model applies learned patterns")
            st.markdown(f"4. **Threshold:** Words with confidence > 0.5 are flagged")
            
            if nlp_option == "Keyword":
                st.markdown("**Keyword model:** Simple dictionary lookup of profane words")
            elif nlp_option == "TF-IDF":
                st.markdown("**TF-IDF model:** Weighs words by importance and checks against trained patterns")
            elif nlp_option == "LSTM":
                st.markdown("**LSTM model:** Uses recurrent neural network to understand word context")
            else:
                st.markdown("**BERT model:** Uses transformer architecture for deep contextual understanding")

        # =========================
        # DISPLAY TRANSCRIPT WITH HIGHLIGHTS
        # =========================
        st.markdown("## 📝 Transcript with Profanity Highlights")

        # Create HTML with original words but flag based on lemmas
        transcript_html = ""
        for w in lemmatized_words:
            if w in flagged:
                transcript_html += f"<span style='background-color:#ffcccc; color:red; font-weight:bold; padding:2px 4px; border-radius:3px' title='Lemma: {w['lemma']}'>{w['original']}</span> "
            else:
                transcript_html += w["original"] + " "

        st.markdown(transcript_html, unsafe_allow_html=True)
        st.caption("*Hover over highlighted words to see their lemmatized form*")

        # =====================================
        # STAGE 6 — MODEL EVALUATION
        # =====================================
        st.markdown("---")
        st.markdown("# 📊 Stage 6: Model Evaluation")
        st.markdown("**ML Concept:** Measuring how well our model performs")

        # Ground truth (based on keyword list for demo)
        from profanity_models.keyword_filter import BAD_WORDS
        
        # Compare lemmatized words against BAD_WORDS (also lemmatized for fair comparison)
        lemmatized_bad_words = set()
        for word in BAD_WORDS:
            if word.isalpha():
                pos = get_wordnet_pos(word)
                lemma = lemmatizer.lemmatize(word.lower(), pos)
                lemmatized_bad_words.add(lemma)
            else:
                lemmatized_bad_words.add(word.lower())

        true_labels = [1 if w["lemma"] in lemmatized_bad_words else 0 for w in lemmatized_words]
        predicted_labels = [1 if w in flagged else 0 for w in lemmatized_words]

        acc, prec, rec, f1, cm = evaluate(true_labels, predicted_labels)

        st.markdown("### Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{acc:.3f}", help="Overall correctness: (TP+TN)/(Total)")
        with col2:
            st.metric("Precision", f"{prec:.3f}", help="Of words flagged as profane, how many were actually profane?")
        with col3:
            st.metric("Recall", f"{rec:.3f}", help="Of all profane words, how many did we catch?")
        with col4:
            st.metric("F1 Score", f"{f1:.3f}", help="Harmonic mean of precision and recall")

        with st.expander("📈 Confusion Matrix Analysis"):
            fig = plot_confusion_matrix(cm)
            st.pyplot(fig)
            
            st.markdown("**What these numbers mean:**")
            st.markdown(f"- ✅ **True Positives: {cm[1][1]}** - Correctly identified profanity")
            st.markdown(f"- ✅ **True Negatives: {cm[0][0]}** - Correctly identified clean words")
            st.markdown(f"- ❌ **False Positives: {cm[0][1]}** - Clean words incorrectly flagged (Type I Error)")
            st.markdown(f"- ❌ **False Negatives: {cm[1][0]}** - Profane words missed (Type II Error)")
            
            st.markdown("**Why preprocessing matters for evaluation:**")
            st.markdown("- Without lemmatization: 'cursing' and 'cursed' would be treated as different words")
            st.markdown("- With lemmatization: Both map to 'curse', improving detection accuracy")

        # =====================================
        # STAGE 7 — DEPLOYMENT (CENSORING)
        # =====================================
        st.markdown("---")
        st.markdown("# 🚀 Stage 7: Deployment")
        st.markdown("**ML Concept:** Using the trained model in real-world applications")

        if flagged_count > 0:
            st.warning(f"⚠️ DEPLOYMENT ACTION: Censoring {flagged_count} profane words...")
            
            with st.spinner("Applying video censorship based on model predictions..."):
                output_path = "temp/output.mp4"
                censor_video(input_path, flagged, output_path)

            st.success("✅ Video censored successfully - Model deployed!")
            
            st.markdown("### 🎬 Final Output")
            st.video(output_path)
            
            st.markdown("**📋 Deployment Summary:**")
            st.markdown("1. **Input:** Raw video file")
            st.markdown("2. **Preprocessing:** Audio extraction → Transcription → Tokenization → Lemmatization")
            st.markdown("3. **Model Inference:** Profanity classification on 7 stages")
            st.markdown("4. **Output Action:** Video censorship applied")
            
            with open(output_path, "rb") as f:
                st.download_button(
                    "📥 Download Censored Video",
                    f,
                    file_name="censored_output.mp4"
                )
        else:
            st.success("✅ No profanity detected - deployment would output original video")
            st.markdown("**📋 Deployment Summary:**")
            st.markdown("- Model ran inference on all words")
            st.markdown("- No profanity detected, so no censorship needed")
            st.markdown("- Original video can be safely used")
            
            with open(input_path, "rb") as f:
                st.download_button(
                    "📥 Download Original Video",
                    f,
                    file_name="original_video.mp4"
                )

        # =====================================
        # COMPLETE ML PIPELINE SUMMARY
        # =====================================
        st.markdown("---")
        st.markdown("# 📚 Complete Machine Learning Pipeline Summary")
        
        pipeline_stages = {
            "1. Data Acquisition": ["Raw video input", "Audio extraction", "Waveform data"],
            "2. Data Preprocessing": ["Speech-to-text conversion", "Tokenization (word splitting)", "Lemmatization (base form reduction)"],
            "3. Feature Extraction": ["Word count", "Unique vocabulary", "Word frequency", "TF-IDF vectors"],
            "4. Model Training": ["Learning patterns from labeled data", "Building classification boundaries"],
            "5. Prediction/Classification": ["Applying model to new words", "Flagging profanity with confidence scores"],
            "6. Evaluation": ["Accuracy metrics", "Confusion matrix analysis", "Error analysis"],
            "7. Deployment": ["Video censorship", "Real-world application", "User-facing output"]
        }
        
        for stage, steps in pipeline_stages.items():
            with st.expander(f"**{stage}**"):
                for step in steps:
                    st.markdown(f"- {step}")
        
        st.success("🎉 Full ML Pipeline Completed! This demonstrates how machine learning transforms raw data into valuable real-world applications.")