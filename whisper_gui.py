
import gradio as gr
import whisper
import os
import subprocess
import tempfile
from pathlib import Path
from datetime import timedelta
import time
import numpy as np
from scipy.io.wavfile import write as wav_write

def format_time(seconds):
    seconds = int(seconds)
    return str(timedelta(seconds=seconds))

def save_recorded_audio(audio_np, sr):
    if audio_np is None or len(audio_np) == 0:
        raise ValueError("No audio data received from microphone.")
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "recorded.wav")
    wav_write(temp_path, sr, audio_np)
    return temp_path

def split_audio(file_path, chunk_duration):
    tmp_dir = tempfile.mkdtemp()
    chunk_pattern = os.path.join(tmp_dir, "chunk_%03d.wav")
    command = [
        "ffmpeg", "-i", file_path,
        "-f", "segment", "-segment_time", str(chunk_duration),
        "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
        chunk_pattern
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return sorted([os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.endswith(".wav")])

def transcribe(audio_input, model_size="base", language="en", chunk_duration=120):
    model = whisper.load_model(model_size)
    os.makedirs("transcriptions", exist_ok=True)

    try:
        if isinstance(audio_input, tuple):  # (sample_rate, np.ndarray)
            sr, audio_np = audio_input
            audio_file = save_recorded_audio(audio_np, sr)
            file_name = "recorded_audio"
        else:
            audio_file = audio_input
            file_name = Path(audio_file).stem
    except Exception as e:
        yield f"❌ Recording failed: {e}", "", None
        return

    output_txt_path = f"transcriptions/{file_name}.txt"
    transcript_lines = []

    chunks = [audio_file]
    if chunk_duration > 0:
        chunks = split_audio(audio_file, chunk_duration)

    total_chunks = len(chunks)
    start_time = time.time()

    for i, chunk in enumerate(chunks):
        result = model.transcribe(chunk, language=language)

        for seg in result.get("segments", []):
            start = format_time(seg["start"])
            end = format_time(seg["end"])
            text = seg["text"].strip()
            transcript_lines.append(f"[{start} - {end}]: {text}")

        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        eta = avg_time * (total_chunks - i - 1)

        progress_status = f"🔄 Chunk {i+1}/{total_chunks} | ⏱ Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}"
        current_transcript = "\n".join(transcript_lines[-10:])
        yield progress_status, current_transcript, None

    # Save full result
    with open(output_txt_path, "w") as f:
        f.write("\n".join(transcript_lines))

    full_transcript = "\n".join(transcript_lines)
    total_time = time.time() - start_time
    yield f"✅ Done in {format_time(total_time)}", full_transcript, output_txt_path

# Build UI
with gr.Blocks(title="Whisper Advanced GUI") as demo:
    gr.Markdown("## 🎙️ Whisper Transcription")

    with gr.Row():
        audio_input = gr.Audio(label="🎧 Upload or Record Audio", type="numpy", scale=2)
        with gr.Column():
            model_dropdown = gr.Dropdown(choices=["tiny", "base", "small", "medium", "large"], value="base", label="Model")
            language_input = gr.Textbox(value="en", label="Language")
            chunk_slider = gr.Slider(minimum=0, maximum=600, step=10, value=120, label="Chunk Duration (sec)")

    with gr.Row():
        transcribe_btn = gr.Button("🎬 Start Transcription")
        status = gr.Textbox(label="Progress", lines=1, interactive=False)
        transcript = gr.Textbox(label="📝 Transcript (last 10 lines)", lines=14)
        download = gr.File(label="⬇️ Download Full Transcript")

    transcribe_btn.click(
        fn=transcribe,
        inputs=[audio_input, model_dropdown, language_input, chunk_slider],
        outputs=[status, transcript, download],
        show_progress=True
    )

demo.launch(share=False, inbrowser=True)
