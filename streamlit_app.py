import os
import tempfile
import streamlit as st
import openai
import requests
import json
from moviepy.editor import VideoFileClip
import whisper

# Load Whisper model for transcription
whisper_model = whisper.load_model("small")

# Azure OpenAI GPT-4o connection details
azure_openai_key = "22ec84421ec24230a3638d1b51e3a7dc"  # Replace with your actual API key
azure_openai_endpoint = "https://internshala.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"

# Function to transcribe audio from a video file using Whisper
def transcribe_audio(video_file):
    # Save the uploaded video file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        temp_video_path = temp_video.name

    try:
        # Load video using MoviePy from the temporary file
        video = VideoFileClip(temp_video_path)

        # Extract the audio and save it as a temporary file
        audio_path = os.path.join(tempfile.gettempdir(), "audio.wav")
        video.audio.write_audiofile(audio_path)

        # Transcribe the audio using Whisper
        transcription = whisper_model.transcribe(audio_path)

    finally:
        # Ensure proper cleanup by closing the video file and deleting temp files
        video.reader.close()  # Close the video reader
        video.audio.reader.close_proc()  # Close the audio reader process
        os.remove(temp_video_path)  # Now it is safe to delete the video file
        os.remove(audio_path)  # Remove the audio file

    return transcription['text']

# Function to correct transcription using Azure OpenAI GPT-4o
def correct_transcription(transcription):
    prompt = f"Correct the following transcription by removing grammatical errors, filler words (umm, hmm), and improve the overall quality:\n\n{transcription}"
    
    try:
        # Setting up headers for the API request
        headers = {
            "Content-Type": "application/json",
            "api-key": azure_openai_key
        }
        
        # Data to be sent to Azure OpenAI
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500
        }

        # Making the POST request to Azure OpenAI
        response = requests.post(azure_openai_endpoint, headers=headers, json=data)

        if response.status_code == 200:
            result = response.json()  # Parse the JSON response
            return result['choices'][0]['message']['content'].strip()
        else:
            return f"Failed to connect: {response.status_code} - {response.text}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit UI
def main():
    st.title("AI-Powered Video Audio Replacement and Azure OpenAI GPT-4o")

    # Step 1: Test Azure OpenAI Connection
    if st.button("Test Azure OpenAI GPT-4o Connection"):
        st.write("Testing connection...")
        prompt = "Hello, Azure OpenAI!"
        response = correct_transcription(prompt)
        st.write(f"Azure OpenAI Response: {response}")

    # Step 2: Upload Video File for Audio Transcription and Correction
    video_file = st.file_uploader("Upload Video for Transcription and Audio Replacement", type=["mp4", "mov", "avi"])

    if video_file and st.button("Process Video"):
        with st.spinner("Processing..."):
            # Step 3: Extract and transcribe audio
            transcription = transcribe_audio(video_file)
            st.write("Original Transcription:", transcription)

            # Step 4: Correct transcription using GPT-4o
            corrected_text = correct_transcription(transcription)
            st.write("Corrected Transcription:", corrected_text)

            # Step 5: Display success message (video processing can be added here)
            st.success("Transcription corrected and displayed successfully!")

if __name__ == "__main__":
    main()

