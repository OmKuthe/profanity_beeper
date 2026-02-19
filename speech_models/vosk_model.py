from vosk import Model, KaldiRecognizer
import json
import wave

MODEL_PATH = "models/vosk-model-small-en-us-0.15"
model = Model(MODEL_PATH)

def transcribe(audio_path):

    wf = wave.open(audio_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    words = []

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if "result" in result:
                for w in result["result"]:
                    words.append({
                        "word": w["word"].lower(),
                        "start": w["start"],
                        "end": w["end"]
                    })

    return words
