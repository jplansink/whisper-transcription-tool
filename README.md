
# 🎙️ Whisper Transcription Tool

This tool automatically splits `.m4a` audio files into 2-minute chunks and transcribes them using OpenAI's Whisper. It's built for batch-processing audio and saving accurate transcriptions in clean `.txt` files.

---

## ⚙️ Requirements

- Python 3.8 or newer
- [FFmpeg](https://ffmpeg.org/download.html) installed and in your system `PATH`
- Internet connection (for downloading Whisper models the first time)

---

## 🧰 Setup

1. Clone or download this repository.
2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
```

3. Install required Python packages:

```bash
pip install -r requirements.txt
```

---

## 📁 Folder Structure

```
whisper_app/
├── whisper_app.py           # Main transcription script
├── requirements.txt         # Python dependencies
├── README.md                # This documentation
├── input/                   # Drop your .m4a files here
├── chunks/                  # Auto-generated audio chunks
├── transcriptions/          # Output .txt transcription files
└── processed/               # Moved files after transcription
```

> You can create the folders manually or let the script auto-generate them.

---

## ▶️ How to Use

1. Put one or more `.m4a` files in the `input/` folder.
2. Run the script:

```bash
python whisper_app.py
```

3. Transcriptions will be saved in `transcriptions/`, and processed audio files will be moved to `processed/`.

---

## ⚙️ Default Settings

- **Model**: `large`
- **Language**: `en` (English)
- **Chunk Duration**: 120 seconds (2 minutes)

To change these, open `whisper_app.py` and modify the values near the top of the `main()` function.

---

## 🧠 Full Source Code

```python
import whisper
from tqdm import tqdm
from datetime import timedelta
import os
import subprocess
import shutil

# Function to split audio into chunks
def split_audio(file_path, output_dir, chunk_duration):
    os.makedirs(output_dir, exist_ok=True)
    split_command = [
        "ffmpeg",
        "-i", file_path,
        "-f", "segment",
        "-segment_time", str(chunk_duration),
        "-c", "copy",
        os.path.join(output_dir, "chunk_%03d.m4a")
    ]
    subprocess.run(split_command, check=True)

# Function to convert seconds to hh:mm:ss
def format_timestamp(seconds):
    return str(timedelta(seconds=round(seconds)))

# Transcribe an audio file
def transcribe_audio(file_path, model, output_file):
    print(f"Starting transcription for {file_path}...")
    result = model.transcribe(file_path, task="transcribe", language="en")

    if not result.get("segments"):
        print(f"No transcription segments found for {file_path}. Skipping...")
        return

    with open(output_file, "a") as file:
        for segment in tqdm(result["segments"], desc=f"Processing {os.path.basename(file_path)}", unit="segment"):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"]
            file.write(f"[{start} - {end}]: {text}\n")

# Main program
def main():
    INPUT_FOLDER = "./input"
    PROCESSED_FOLDER = "./processed"
    CHUNKS_FOLDER = "./chunks"
    DEFAULT_MODEL = "large"
    DEFAULT_CHUNK_DURATION = 120
    DEFAULT_LANGUAGE = "en"

    print("Welcome to the Whisper Transcription App!")
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    os.makedirs(CHUNKS_FOLDER, exist_ok=True)

    audio_files = [os.path.join(INPUT_FOLDER, f) for f in os.listdir(INPUT_FOLDER) if f.endswith(".m4a")]
    if not audio_files:
        print("No `.m4a` files found in the input folder. Exiting...")
        return

    model_size = DEFAULT_MODEL
    chunk_duration = DEFAULT_CHUNK_DURATION
    language = DEFAULT_LANGUAGE

    print(f"Loading Whisper model '{model_size}'...")
    model = whisper.load_model(model_size)

    for audio_file in audio_files:
        print(f"\nProcessing file: {audio_file}")
        chunks_dir = os.path.join(CHUNKS_FOLDER, os.path.basename(audio_file).replace(".m4a", ""))
        os.makedirs(chunks_dir, exist_ok=True)

        transcription_filename = f"transcript_{os.path.splitext(os.path.basename(audio_file))[0]}.txt"
        output_file = os.path.join("./transcriptions", transcription_filename)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        print(f"Splitting audio into {chunk_duration}-second chunks...")
        split_audio(audio_file, chunks_dir, chunk_duration)

        chunk_files = sorted([os.path.join(chunks_dir, f) for f in os.listdir(chunks_dir) if f.endswith(".m4a")])
        with tqdm(total=len(chunk_files), desc="Processing all chunks", unit="chunk") as chunk_bar:
            for chunk_file in chunk_files:
                transcribe_audio(chunk_file, model, output_file)
                chunk_bar.update(1)

        shutil.move(audio_file, os.path.join(PROCESSED_FOLDER, os.path.basename(audio_file)))
        print(f"File processed and moved to {PROCESSED_FOLDER}")

    print("All files have been processed!")

if __name__ == "__main__":
    main()
```
