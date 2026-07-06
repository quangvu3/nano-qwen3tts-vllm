"""
Create a voice-clone .pkl file for use as a `speaker` in examples/server.py.

The server's /v1/audio/speech endpoint checks examples/voices/<speaker>.pkl
(see is_voice_clone/load_voice_clone_prompt in server.py) and, if found, uses it
as a voice-clone prompt instead of a built-in CustomVoice speaker. This script
builds that pickle from a reference audio clip (+ optional transcript) using a
Base model, the same way examples/voices/Anna.pkl was made.

Usage:
    # ICL mode (recommended): needs the exact transcript of ref-audio
    python examples/create_voice_clone.py \
        --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --ref-audio examples/speakers/storyteller_1.wav \
        --ref-text "Hoa cuc dai vang tuoi no ro day dac ven bo suoi trong veo mat lanh." \
        --name Storyteller

    # x_vector_only mode: no transcript needed, weaker cloning
    python examples/create_voice_clone.py \
        --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --ref-audio examples/speakers/storyteller_1.wav \
        --x-vector-only \
        --name Storyteller

Then request it from the server with: {"speaker": "Storyteller", ...}
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from nano_qwen3tts_vllm.interface import Qwen3TTSInterface


def main():
    parser = argparse.ArgumentParser(description="Create a voice-clone .pkl for examples/server.py")
    parser.add_argument("--model-path", type=str, required=True,
                         help="Base model path or HuggingFace ID (e.g. Qwen/Qwen3-TTS-12Hz-1.7B-Base)")
    parser.add_argument("--ref-audio", type=str, required=True, help="Path to reference audio file")
    parser.add_argument("--ref-text", type=str, default="",
                         help="Exact transcript of ref-audio (required unless --x-vector-only)")
    parser.add_argument("--x-vector-only", action="store_true",
                         help="Speaker-embedding-only mode (no transcript needed, weaker cloning)")
    parser.add_argument("--name", type=str, required=True, help="Voice name, e.g. 'Anna' -> voices/Anna.pkl")
    parser.add_argument("--voices-dir", type=str, default=str(Path(__file__).parent / "voices"),
                         help="Output directory (default: examples/voices)")
    args = parser.parse_args()

    if not args.x_vector_only and not args.ref_text:
        parser.error("--ref-text is required unless --x-vector-only is set")

    voices_dir = Path(args.voices_dir)
    voices_dir.mkdir(parents=True, exist_ok=True)
    out_path = voices_dir / f"{args.name}.pkl"

    print(f"Loading model from: {args.model_path}")
    if os.path.isdir(args.model_path) or os.path.isfile(args.model_path):
        interface = Qwen3TTSInterface(model_path=args.model_path, enforce_eager=False, tensor_parallel_size=1)
    else:
        interface = Qwen3TTSInterface.from_pretrained(
            pretrained_model_name_or_path=args.model_path, enforce_eager=False, tensor_parallel_size=1,
        )
    print("Model loaded.")

    ref_audio, ref_sr = sf.read(args.ref_audio)
    if ref_audio.ndim > 1:
        ref_audio = np.mean(ref_audio, axis=-1)

    voice_clone_prompt = interface.create_voice_clone_prompt(
        ref_audio=(ref_audio, ref_sr),
        ref_text=args.ref_text if args.ref_text else None,
        x_vector_only_mode=args.x_vector_only,
    )

    with open(out_path, "wb") as f:
        pickle.dump(voice_clone_prompt, f)

    print(f"Saved voice clone to: {out_path.absolute()}")
    print(f"Use it via: {{\"speaker\": \"{args.name}\", ...}} in POST /v1/audio/speech")


if __name__ == "__main__":
    main()
