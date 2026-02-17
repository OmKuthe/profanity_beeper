from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment
from pydub.generators import Sine
import os

def load_bad_words():
    with open("badwords.txt") as f:
        return set(w.strip().lower() for w in f.readlines())

BAD_WORDS = load_bad_words()

def censor_video(video_path, word_timestamps, output_path):

    video = VideoFileClip(video_path)
    audio = AudioSegment.from_file(video_path)

    beep = Sine(1000).to_audio_segment(duration=300).apply_gain(-5)

    for w in word_timestamps:
        if w["word"] in BAD_WORDS:
            start = int(w["start"] * 1000)
            end = int(w["end"] * 1000)

            duration = end - start
            beep_sound = beep[:duration]

            audio = audio[:start] + beep_sound + audio[end:]

    temp_audio = "temp/censored.wav"
    audio.export(temp_audio, format="wav")

    final = video.set_audio(AudioFileClip(temp_audio))
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")
