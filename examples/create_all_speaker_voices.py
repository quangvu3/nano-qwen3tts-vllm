"""
Batch-create voice-clone .pkl files for every reference pair in examples/speakers/.

For each `<stem>.wav` in --speakers-dir (skipping `.wav.bak` backups), reads the
sibling `<stem>.txt` as the exact transcript and builds an ICL-mode voice-clone
prompt via Qwen3TTSInterface.create_voice_clone_prompt, then pickles it to
--voices-dir/<stem>.pkl -- the same format examples/create_voice_clone.py produces
and that examples/server.py's is_voice_clone()/load_voice_clone_prompt() expect.

The model is loaded once and reused across all speaker files.

Usage:
    python examples/create_all_speaker_voices.py \
        --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base

    # Re-create pkls that already exist:
    python examples/create_all_speaker_voices.py \
        --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base --overwrite
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
    parser = argparse.ArgumentParser(description="Batch-create voice-clone .pkl files from examples/speakers/")
    parser.add_argument("--model-path", type=str, required=True,
                         help="Base model path or HuggingFace ID (e.g. Qwen/Qwen3-TTS-12Hz-1.7B-Base)")
    parser.add_argument("--speakers-dir", type=str, default=str(Path(__file__).parent / "speakers"))
    parser.add_argument("--voices-dir", type=str, default=str(Path(__file__).parent / "voices"))
    parser.add_argument("--overwrite", action="store_true", help="Recreate .pkl even if it already exists")
    args = parser.parse_args()

    speakers_dir = Path(args.speakers_dir)
    voices_dir = Path(args.voices_dir)
    voices_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    for wav_path in sorted(speakers_dir.glob("*.wav")):
        stem = wav_path.stem
        txt_path = speakers_dir / f"{stem}.txt"
        if not txt_path.exists():
            print(f"[skip] {stem}: no matching .txt found")
            continue
        pairs.append((stem, wav_path, txt_path))

    if not pairs:
        print(f"No wav/txt pairs found in {speakers_dir}")
        return

    print(f"Found {len(pairs)} speaker pairs in {speakers_dir}")

    print(f"Loading model from: {args.model_path}")
    if os.path.isdir(args.model_path) or os.path.isfile(args.model_path):
        interface = Qwen3TTSInterface(model_path=args.model_path, enforce_eager=False, tensor_parallel_size=1)
    else:
        interface = Qwen3TTSInterface.from_pretrained(
            pretrained_model_name_or_path=args.model_path, enforce_eager=False, tensor_parallel_size=1,
        )
    print("Model loaded.\n")

    created, skipped, failed = 0, 0, 0
    for stem, wav_path, txt_path in pairs:
        out_path = voices_dir / f"{stem}.pkl"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {stem}: {out_path.name} already exists")
            skipped += 1
            continue

        try:
            ref_text = txt_path.read_text(encoding="utf-8").strip()
            ref_audio, ref_sr = sf.read(str(wav_path))
            if ref_audio.ndim > 1:
                ref_audio = np.mean(ref_audio, axis=-1)

            voice_clone_prompt = interface.create_voice_clone_prompt(
                ref_audio=(ref_audio, ref_sr),
                ref_text=ref_text,
                x_vector_only_mode=False,
            )

            with open(out_path, "wb") as f:
                pickle.dump(voice_clone_prompt, f)

            print(f"[ok] {stem}: saved {out_path.name}")
            created += 1
        except Exception as e:
            print(f"[fail] {stem}: {e}")
            failed += 1

    print(f"\nDone. created={created} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
