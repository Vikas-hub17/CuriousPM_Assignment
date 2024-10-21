import streamlit as st
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech
import openai
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import io
import requests
import json

# Azure OpenAI GPT-4o credentials
openai.api_key = "22ec84421ec24230a3638d1b51e3a7dc"  # Replace with your actual API key
azure_openai_endpoint = "https://internshala.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"

# Function to connect to Azure OpenAI GPT-4o and get a basic response
def test_openai_connection():
    headers = {
        "Content-Type": "application/json",
        "api-key": openai.api_key
    }
    data = {
        "messages": [{"role": "user", "content": "Hello, Azure OpenAI!"}],
        "max_tokens": 50
    }
    response = requests.post(azure_openai_endpoint, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    else:
        return f"Failed: {response.status_code} - {response.text}"

# Function to transcribe video audio using Google Speech-to-Text
def transcribe_audio(audio_file):
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=audio_file.read())
    config = speech.RecognitionConfig(language_code="en-US")

    response = client.recognize(config=config, audio=audio)
    transcription = "".join([result.alternatives[0].transcript for result in response.results])
    return transcription

# Function to correct transcription using Azure OpenAI GPT-4o
def correct_transcription(transcription):
    prompt = f"Correct the following transcription by removing grammatical errors, filler words (umm, hmm), and improve the overall quality:\n\n{transcription}"
    response = openai.Completion.create(
        engine="gpt-4o",
        prompt=prompt,
        max_tokens=500
    )
    return response.choices[0].text.strip()

# Function to generate audio from corrected transcription using Google Text-to-Speech
def synthesize_audio(corrected_text):
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=corrected_text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", name="en-US-Journey")
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
    return response.audio_content

# Function to replace the original video's audio with the AI-generated audio
def replace_audio(video_file, new_audio):
    video = VideoFileClip(video_file.name)
    audio = AudioSegment.from_file(io.BytesIO(new_audio), format="mp3")
    video = video.set_audio(audio)
    output_path = "output_video.mp4"
    video.write_videofile(output_path, codec="libx264")
    return output_path

# Streamlit UI
st.title("AI-Powered Video Audio Replacement with Azure OpenAI GPT-4o")

# Step 1: Test Azure OpenAI Connection
if st.button("Test Azure OpenAI Connection"):
    st.write("Connecting to Azure OpenAI GPT-4o...")
    result = test_openai_connection()
    st.write(f"Azure OpenAI Response: {result}")

# Step 2: Upload Video File for Audio Replacement
video_file = st.file_uploader("Upload Video for Audio Replacement", type=["mp4", "mov", "avi"])

if video_file and st.button("Process Video"):
    with st.spinner("Processing..."):
        # Step 3: Extract and transcribe audio
        transcription = transcribe_audio(video_file)
        st.write("Original Transcription:", transcription)

        # Step 4: Correct transcription using GPT-4o
        corrected_text = correct_transcription(transcription)
        st.write("Corrected Transcription:", corrected_text)

        # Step 5: Generate AI audio from corrected transcription
        new_audio = synthesize_audio(corrected_text)

        # Step 6: Replace original video audio with generated audio
        output_video = replace_audio(video_file, new_audio)

        # Step 7: Display the output video with new audio
        st.video(output_video)
        st.success("Audio replaced successfully!")
