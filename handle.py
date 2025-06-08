from chatterbox.tts import ChatterboxTTS
import torchaudio as ta
import os
import uuid
import base64

# Load the model only once when the container starts
model = ChatterboxTTS.from_pretrained(device="cuda")

def handler(event):
    text = event["input"].get("text")
    reference_audio = event["input"].get("reference_audio")  # optional

    if not text:
        return {"error": "Missing 'text' in input."}

    # Optional: save reference audio to temp path
    if reference_audio:
        ref_path = f"/tmp/{uuid.uuid4()}.wav"
        with open(ref_path, "wb") as f:
            f.write(bytes.fromhex(reference_audio))
        wav = model.generate(text, audio_prompt_path=ref_path)
    else:
        wav = model.generate(text)

    # Save output wav
    out_path = f"/tmp/{uuid.uuid4()}.wav"
    ta.save(out_path, wav, model.sr)

    with open(out_path, "rb") as f:
        audio_bytes = f.read()

    # Return audio as hex string
    return {
        "output": {
            "audio_hex": audio_bytes.hex()
        }
    }
