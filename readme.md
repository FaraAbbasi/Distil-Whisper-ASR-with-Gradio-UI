# Distil Whisper ASR with Gradio UI
A lightweight web interface for speech-to-text transcription using the Distil-Whisper model, optimized for English speech recognition. This project uses Gradio to provide an interactive UI for real-time microphone input and file upload transcription.

## Features

* Real-time Microphone Transcription: Speak into your microphone and get instant text output.

* File Upload Support: Upload audio files (.wav, .mp3, .aac) for transcription.

* Distil-Whisper Model: Uses the efficient distil-whisper/distil-small.en, a distilled version of OpenAI's Whisper.

* Gradio Web UI: Easy-to-use interface for both live and file-based transcription.

* Error Handling: Includes file validation and graceful error responses.

## 🚀Quick Start

### 1st Method (Recommended)

#### Installation by creating a python virtual enviroment

To setup a python virtual environment to run the application follow these steps:

```bash
# create a new virtual environment
python -m venv .venv 

# Activate the virtual environment in windows using gitbash
source .venv/Scripts/activate

# Install required packages 
pip install -r requirements.txt

# Running the application
python app.py

# Then open the provided public URL in your browser.
```

### 2nd Method

```bash
# Clone the repository
git clone https://github.com/FaraAbbasi/Distil-Whisper-ASR-with-Gradio-UI.git

# Change the current directory
cd Distil-Whisper-ASR-with-Gradio-UI

# Install dependencies
pip install -r requirements.txt

# Running the App
python app.py

# Then open the provided public URL in your browser.
```

## How It Works

### Model Initialization

```python
from transformers import pipeline
# Initialize the ASR pipeline with distil-whisper
asr = pipeline(
    task="automatic-speech-recognition",
    model="distil-whisper/distil-small.en"  # Distilled version for efficiency
)
```

### Audio Processing

The system:

1. Accepts audio input (microphone or file)

1. Splits long audio into 30-second chunks

2. Processes chunks in batches (size=8) for GPU optimization

3. Limits output tokens to 256 per chunk for focused transcription

4. Combines results into a complete transcript

### Transcription Function

```python
def transcribe_speech(filepath):
    output = asr(filepath,
                 batch_size=8,           # Process 8 chunks simultaneously
                 chunk_length_s=30,      # 30-second audio segments
                 max_new_tokens=256)     # Limit output length
    return output["text"]

```

### Parameters Explained

Parameter | Value | Purpose
--------- | ----- | ------- |
batch_size | 8 | Number of audio chunks processed simultaneously (GPU optimization)
chunk_length_s | 30 | Split long audio into 30-second segments for processing
max_new_tokens | 256 | Limits transcription to ~200-300 words per chunk

## Web Interface

#### Features

* Dual Interface: Separate tabs for microphone and file upload

* Real-time Feedback: Visual indicators during processing

* Error Handling: Clear warnings for invalid inputs

* Responsive Design: Works on desktop and mobile browsers

## How to use

1. ### Microphone Input

   * Click the "Microphone" button to start recording.

   * Speak into your microphone.

   * View the transcribed text in the output box.

2. ### File Upload

   * Upload an audio file using the file uploader.

   * Click "Transcribe" to process the file.

   * Download or view the transcribed text.
  
## Code Structure

* app.py – Main script containing the Gradio interface and transcription logic.

* transcribe_speech() – Function to transcribe audio files using the Distil-Whisper pipeline.

* Model is loaded via pipeline("automatic-speech-recognition", model="distil-whisper/distil-multimix").

## Model Information

* Model: distil-whisper/distil-small.en

* Task: Automatic Speech Recognition (ASR)

* Language: Optimized for English

* Framework: Hugging Face Transformers

* This is a distilled version of OpenAI's Whisper model, offering faster inference with minimal accuracy loss.

## Example

```python
from transformers import pipeline

asr_pipeline = pipeline("automatic-speech-recognition", model="distil-whisper/distil-multimix")
result = asr_pipeline("audio.wav")
print(result["text"])

```

## Error Handling

* If an audio file is not found, the function returns an empty string.

* Invalid file formats may cause transcription errors.

* Microphone transcription requires browser microphone permissions.

## License

This project is open-source and available under the MIT License.

## Acknowledgements

* Hugging Face for the Transformers library and model hosting.

* OpenAI for the original Whisper model.

* Gradio for the UI framework.

## Contributing

Contributions are warmly welcomed! This project thrives on community input and collaboration. We encourage you to participate in making it better, whether through bug reports, feature suggestions, or direct code contributions.

## Support

For issues and questions:

1. Check the [FAQ section](#frequently-asked-questions)

1. Open an Issue on GitHub

2. Provide detailed information about your problem

----
<div align="center"> Made with ❤️ for the open-source community
Give a ⭐ if you find this project useful!

</div>

## ❓Frequently Asked Questions

**Q: How accurate is the transcription?**\
A: The distil-small.en model achieves near-whisper-small accuracy (~97%) on English benchmarks.


**Q: Can I use this for languages other than English?**\
A: This specific model is optimized for English. For multilingual support, use the base Whisper model.

**Q: Does this work offline?**\
A: After the initial download, the model runs locally, so internet is not required for transcription.

**Q: What audio formats are supported?**\
A: Most common formats: MP3, WAV, FLAC, M4A, OGG, and more via FFmpeg.

**Q: How long can the audio be?**\
A: There's no strict limit - long audio is automatically chunked and processed.

**Q: Is GPU required?**\
A: No, but GPU acceleration significantly improves speed. The system works on CPU as well.

