import os
import sys
import uuid
import base64
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Include src in path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Load the model once per container
model = ChatterboxTTS.from_pretrained(device="cuda")

def handler(event):
    input_data = event.get("input", {})
    text = input_data.get("text")
    voice_mode = input_data.get("voice_mode", "default")
    reference_audio_b64 = input_data.get("reference_audio")

    if not text:
        return {"error": "Missing 'text' in input."}

    # Handle reference audio (if cloning)
    if voice_mode == "clone" and reference_audio_b64:
        ref_path = f"/tmp/{uuid.uuid4()}.wav"
        with open(ref_path, "wb") as f:
            f.write(base64.b64decode(reference_audio_b64))
        wav = model.generate(text, audio_prompt_path=ref_path)
    else:
        wav = model.generate(text)

    # Save output audio
    out_path = f"/tmp/{uuid.uuid4()}.wav"
    ta.save(out_path, wav, model.sr)

    # Read and encode as base64
    with open(out_path, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    return {
        "output": {
            "audio_b64": audio_b64
        }
    }
