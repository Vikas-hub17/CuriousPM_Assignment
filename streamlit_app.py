import streamlit as st
import openai
import whisper
from gtts import gTTS
import pyttsx3
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import io
import requests
import os
import tempfile
from moviepy.editor import VideoFileClip

# Initialize Whisper model (small model to balance speed and accuracy)
whisper_model = whisper.load_model("small")

# Azure OpenAI GPT-4o credentials
openai.api_key = "22ec84421ec24230a3638d1b51e3a7dc"  # Replace with your actual API key
azure_openai_endpoint = "https://internshala.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"

# Function to transcribe audio using Whisper
def transcribe_audio(video_file):
    # Save the uploaded video file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        temp_video_path = temp_video.name

    # Use MoviePy to read the video from the temporary file
    video = VideoFileClip(temp_video_path)

    # Extract audio and save it as a temporary audio file
    audio_path = os.path.join(tempfile.gettempdir(), "audio.wav")
    video.audio.write_audiofile(audio_path)

    # Use Whisper (or any other ASR) to transcribe the audio
    transcription = whisper_model.transcribe(audio_path)
    
    # Delete the temporary files after processing
    os.remove(temp_video_path)
    os.remove(audio_path)

    return transcription['text']

# Function to correct transcription using Azure OpenAI GPT-4o
def correct_transcription(transcription):
    prompt = f"Correct the following transcription by removing grammatical errors, filler words (umm, hmm), and improve the overall quality:\n\n{transcription}"
    response = openai.Completion.create(
        engine="gpt-4o",
        prompt=prompt,
        max_tokens=500
    )
    return response.choices[0].text.strip()

# Function to synthesize corrected text into speech using gTTS
def synthesize_audio_gtts(corrected_text):
    tts = gTTS(text=corrected_text, lang='en')
    audio_path = "corrected_audio.mp3"
    tts.save(audio_path)
    return audio_path

# Alternative function to synthesize corrected text into speech using pyttsx3 (Offline)
def synthesize_audio_pyttsx(corrected_text):
    engine = pyttsx3.init()
    audio_path = "corrected_audio.mp3"
    engine.save_to_file(corrected_text, audio_path)
    engine.runAndWait()
    return audio_path

# Function to replace the original video's audio with the AI-generated audio
def replace_audio(video_file, new_audio_path):
    video = VideoFileClip(video_file.name)
    
    # Load new audio
    audio = AudioSegment.from_file(new_audio_path, format="mp3")
    
    # Replace video audio with new audio
    video = video.set_audio(audio)
    
    # Save the new video
    output_path = "output_video.mp4"
    video.write_videofile(output_path, codec="libx264")
    return output_path

# Streamlit UI
st.title("AI-Powered Video Audio Replacement (Without Google Cloud)")

video_file = st.file_uploader("Upload Video for Audio Replacement", type=["mp4", "mov", "avi"])

if video_file and st.button("Process Video"):
    with st.spinner("Processing..."):
        # Step 1: Transcribe the video using Whisper
        transcription = transcribe_audio(video_file)
        st.write("Original Transcription:", transcription)

        # Step 2: Correct transcription using GPT-4o
        corrected_text = correct_transcription(transcription)
        st.write("Corrected Transcription:", corrected_text)

        # Step 3: Synthesize AI audio from corrected transcription
        # Option 1: Using gTTS (online)
        new_audio_path = synthesize_audio_gtts(corrected_text)

        # Option 2: Using pyttsx3 (offline)
        # new_audio_path = synthesize_audio_pyttsx(corrected_text)

        # Step 4: Replace original video audio with generated audio
        output_video = replace_audio(video_file, new_audio_path)

        # Step 5: Display the output video with new audio
        st.video(output_video)
        st.success("Audio replaced successfully!")
