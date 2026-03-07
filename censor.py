import subprocess
import os
import json

def censor_video(input_path, flagged_words, output_path):
    """
    Censor profane words in video by beeping the audio
    
    Args:
        input_path: Path to input video
        flagged_words: List of word dictionaries with timing information
        output_path: Path for output video
    """
    
    if not flagged_words:
        print("No words to censor")
        return
    
    # Prepare filter commands for ffmpeg
    filter_complex = ""
    
    # Create volume envelopes for each flagged word
    for i, word_info in enumerate(flagged_words):
        # Handle different dictionary structures
        if isinstance(word_info, dict):
            # Try different possible key names
            start = word_info.get('start', word_info.get('start_time', 0))
            end = word_info.get('end', word_info.get('end_time', start + 0.5))
            
            # Get word for logging
            word = (word_info.get('word') or 
                   word_info.get('original') or 
                   word_info.get('text') or 
                   f"word_{i}")
        else:
            # If it's not a dict, skip or use default
            print(f"Skipping non-dict word info: {word_info}")
            continue
        
        # Ensure we have valid times
        if start is None or end is None:
            print(f"Missing timing for word: {word}")
            continue
            
        # Create volume envelope (beep during profane words)
        # volume=0: silence during profanity, then return to normal
        if i > 0:
            filter_complex += ","
        
        # Add silence during profane word
        filter_complex += f"volume=enable='between(t,{start},{end})':volume=0"
        
        print(f"Censoring '{word}' from {start}s to {end}s")
    
    if not filter_complex:
        print("No valid timings found for censorship")
        return
    
    # Complete ffmpeg command
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-af', filter_complex,
        '-c:v', 'copy',  # Copy video stream without re-encoding
        '-y',  # Overwrite output file
        output_path
    ]
    
    try:
        # Run ffmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            raise Exception("FFmpeg processing failed")
        
        print(f"Video censored successfully: {output_path}")
        
    except Exception as e:
        print(f"Error during censoring: {e}")
        raise

# Alternative simpler version using pydub (if you prefer)
def censor_video_simple(input_path, flagged_words, output_path):
    """
    Simpler censoring using pydub (requires pydub installation)
    """
    try:
        from pydub import AudioSegment
        from moviepy.editor import VideoFileClip
    except ImportError:
        print("pydub not installed, falling back to ffmpeg method")
        censor_video(input_path, flagged_words, output_path)
        return
    
    # Load video
    video = VideoFileClip(input_path)
    audio = video.audio
    
    # Create beep sound
    beep_duration = 100  # milliseconds
    beep_freq = 1000  # Hz
    beep = AudioSegment.sine(beep_freq, duration=beep_duration)
    
    # Export audio temporarily
    temp_audio = "temp/temp_audio.wav"
    audio.write_audiofile(temp_audio)
    
    # Load audio with pydub
    audio_segment = AudioSegment.from_wav(temp_audio)
    
    # Overlay beeps at flagged positions
    for word_info in flagged_words:
        # Get timing information
        if isinstance(word_info, dict):
            start_ms = word_info.get('start', 0) * 1000  # Convert to milliseconds
            end_ms = word_info.get('end', start_ms/1000 + 0.5) * 1000
        else:
            continue
        
        # Overlay beep
        beep_position = int((start_ms + end_ms) / 2) - beep_duration//2
        audio_segment = audio_segment.overlay(beep, position=beep_position)
    
    # Export modified audio
    modified_audio = "temp/modified_audio.wav"
    audio_segment.export(modified_audio, format="wav")
    
    # Combine with video
    from moviepy.editor import VideoFileClip, AudioFileClip
    video = VideoFileClip(input_path)
    new_audio = AudioFileClip(modified_audio)
    final_video = video.set_audio(new_audio)
    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
    
    # Cleanup
    os.remove(temp_audio)
    os.remove(modified_audio)
    
    print(f"Video censored successfully: {output_path}")

# For backward compatibility
def censor_video_compat(input_path, flagged_words, output_path):
    """Compatibility wrapper for different word dict formats"""
    
    # Normalize the flagged words
    normalized_words = []
    for w in flagged_words:
        if isinstance(w, dict):
            # Create a standardized word dict
            std_word = {
                'word': w.get('word') or w.get('original') or 'unknown',
                'start': w.get('start') or w.get('start_time') or 0,
                'end': w.get('end') or w.get('end_time') or (w.get('start') or 0) + 0.5
            }
            normalized_words.append(std_word)
        else:
            print(f"Skipping non-dict word: {w}")
    
    # Call the main censoring function
    censor_video(input_path, normalized_words, output_path)