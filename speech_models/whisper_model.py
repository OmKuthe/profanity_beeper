import whisper

model = whisper.load_model("base")

def transcribe(audio_path):
    result = model.transcribe(audio_path, word_timestamps=True)

    words = []
    for segment in result["segments"]:
        for word in segment["words"]:
            words.append({
                "word": word["word"].lower().strip(),
                "start": word["start"],
                "end": word["end"]
            })
    return words
