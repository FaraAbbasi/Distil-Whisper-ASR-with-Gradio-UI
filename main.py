"""
Automatic Speech Recognition (ASR) with Distil-Whisper
A web interface for speech-to-text transcription using distil-whisper/distil-small.en
"""
# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

"""
This code sets up an automatic speech recognition (ASR) system using the Hugging Face Transformers library.

If you would like to run this code on your own machine, you can install the following:
"""

# !pip install transformers
# !pip install gradio

"""
Creates an ASR pipeline with two key parameters:

task="automatic-speech-recognition" - Specifies this is for converting speech to text

model="distil-whisper/distil-small.en" - Uses a distilled version of OpenAI's Whisper model optimized for English
"""

from transformers import pipeline

asr = pipeline(task="automatic-speech-recognition",     # Specifies this is for converting speech to text
               model="distil-whisper/distil-small.en")  # Uses a distilled version of OpenAI's Whisper model optimized for English

"""
Transcribing The Audio
Transcribe audio file to text using speech recognition model.

Arguments: filepath: Path to audio file (e.g., .mp3, .wav)

Returns: string: Transcribed text or empty string if error occurs
"""

import os

def transcribe_speech(filepath):  
    
    if filepath is None:            # Check if audio file was provided
        gr.Warning("No audio found, please retry.")   
        return ""                   # Return empty string on error
        
    output = asr(filepath,          #  Gets the transcribed text from the model's output.
      batch_size=8,                 # Process 8 chunks simultaneously (GPU optimization)
      chunk_length_s=30,            # Split long files into 30-second segments
      max_new_tokens=256)           # Limit output to ~200-300 words per chunk
    return output["text"]           #  Returns the transcribed text as a string

"""
Build a shareable app with Gradio
This code creates a Gradio web interface for live microphone speech transcription.
"""

import gradio as gr       #  Imports Gradio - a library for creating web UIs for ML models

demo = gr.Blocks()  

"""
Microphone Transcription Interface
"""

mic_transcribe = gr.Interface(    
    fn=transcribe_speech,     
    inputs=gr.Audio(sources="microphone",  
                    type="filepath"),
    outputs=gr.Textbox(label="Transcription",
                       lines=9),
    allow_flagging="never")

"""
File Upload Transcription Interface
"""

file_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources="upload",
                    type="filepath"),
    outputs=gr.Textbox(label="Transcription",
                       lines=3),
    allow_flagging="never",
)

"""
Launching The UI
"""

with demo:
    gr.TabbedInterface(
        [mic_transcribe,
         file_transcribe],
        ["Transcribe Microphone",
         "Transcribe Audio File"],
    )

demo.launch(debug = True)
