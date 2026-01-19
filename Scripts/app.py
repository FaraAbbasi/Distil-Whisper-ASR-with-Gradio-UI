import os
import gradio as gr
from transformers import pipeline

# Initialize the ASR pipeline

print("Loading Distil-Whisper model...")
asr = pipeline(
    "automatic-speech-recognition",
    model="distil-whisper/distil-small.en"
)
print("Model loaded successfully!")

# Transcribe audio file to text using Distil-Whisper model.

def transcribe_speech(filepath):  
    
    if filepath is None:           
        gr.Warning("No audio found, please retry.")   
        return ""                   
        
    output = asr(filepath,          
      batch_size=8,                 
      chunk_length_s=30,            
      max_new_tokens=256)           
    return output["text"]    

# Create Gradio interface
demo = gr.Blocks(title="Transcribing Chatbot", theme="soft")

# Build the interface
with demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            <div style="text-align: center;">
                <h1 style="font-size: 2.5em; color: #4a6fa5; margin-bottom: 10px;">🎙️ UI Transcribing Chatbot</h1>
                <h3 style="color: #666; font-weight: normal;">Convert speech to text instantly</h3>
                <hr style="margin: 20px 0; border: 1px solid #e0e0e0;">
            </div>
            """)
    
    # Tabs for different input methods
    with gr.Tabs():
        # Tab 1: Microphone
        with gr.Tab("🎤 Microphone"):
            audio_input = gr.Audio(sources=["microphone"], label="Record", type = 'filepath')
            transcribe_btn = gr.Button("Transcribe", variant="primary")
            output = gr.Textbox(label="Transcript", lines=5)
            
            transcribe_btn.click(
                fn=transcribe_speech,
                inputs=audio_input,
                outputs=output
            )
        
        # Tab 2: File Upload
        with gr.Tab("📁 File Upload"):
            file_input = gr.File(label="Upload audio file", file_types=["audio"])
            transcribe_file_btn = gr.Button("Transcribe", variant="primary")
            file_output = gr.Textbox(label="Transcript", lines=5)
            
            transcribe_file_btn.click(
                fn=transcribe_speech,
                inputs=file_input,
                outputs=file_output
            )

# Launch the app
if __name__ == "__main__":
    demo.launch(debug=True)