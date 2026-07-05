# coding=utf-8
"""Qwen3-TTS Gradio Demo for nano-qwen3tts-vllm.

Ported from Qwen3-TTS-Demo/app2.py (the original qwen_tts.Qwen3TTSModel-based HF Spaces
demo) to this repo's async multiprocess/ZMQ engine (nano_qwen3tts_vllm.interface.Qwen3TTSInterface).

Supports: Voice Design, Voice Clone (Base), TTS (CustomVoice)
Lazy loading - models are loaded on-demand when clicking "Load Model" button.
Both 0.6B and 1.7B models are available.

Note: each loaded model spawns its own Talker + Predictor worker processes
(nano-qwen3tts-vllm/workers/). Loading multiple tabs/sizes at once runs multiple
process pairs concurrently on the GPU - heavier than a single-model demo, but this
mirrors the lazy-load-per-tab UX of the original app2.py.

Usage:
    python examples/gradio_app.py
"""

import numpy as np

import gradio as gr

from nano_qwen3tts_vllm.interface import Qwen3TTSInterface

# Speaker and language choices for CustomVoice model
SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"
]
LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian", "Italian", "Dutch", "Turkish", "Arabic"]
# g-group-ai-lab/gwen-tts-0.6B adds Vietnamese support; only offered once that model is loaded.
LANGUAGES_VN = LANGUAGES + ["Vietnamese"]

VN_MODEL_ID = "g-group-ai-lab/gwen-tts-0.6B"


# ============================================================================
# LAZY MODEL LOADING - Models loaded on-demand when "Load Model" is clicked
# ============================================================================

# Global interface references (loaded lazily); each one owns its own worker processes.
voice_design_interface_0_6b = None
voice_design_interface_1_7b = None
voice_design_interface_vn = None
base_interface_0_6b = None
base_interface_1_7b = None
base_interface_vn = None
custom_voice_interface_0_6b = None
custom_voice_interface_1_7b = None
custom_voice_interface_vn = None


async def _load_interface(label: str, progress: gr.Progress, *, model_id: str) -> tuple:
    """Download/load a Qwen3TTSInterface and start its ZMQ worker processes."""
    progress(0, desc=f"Loading {label} model...")
    interface = Qwen3TTSInterface.from_pretrained(
        pretrained_model_name_or_path=model_id,
        enforce_eager=False,
        tensor_parallel_size=1,
    )
    await interface.start_zmq_tasks()
    progress(1.0, desc="Model loaded successfully!")
    return interface


async def load_voice_design_model_0_6b(progress=gr.Progress(track_tqdm=True)):
    """Load VoiceDesign 0.6B model on-demand."""
    global voice_design_interface_0_6b
    if voice_design_interface_0_6b is not None:
        return "Model already loaded."
    try:
        voice_design_interface_0_6b = await _load_interface(
            "VoiceDesign 0.6B", progress, model_id="Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign"
        )
        return "VoiceDesign 0.6B model loaded successfully!"
    except Exception as e:
        return f"Error loading model: {type(e).__name__}: {e}"


async def load_voice_design_model_1_7b(progress=gr.Progress(track_tqdm=True)):
    """Load VoiceDesign 1.7B model on-demand."""
    global voice_design_interface_1_7b
    if voice_design_interface_1_7b is not None:
        return "Model already loaded."
    try:
        voice_design_interface_1_7b = await _load_interface(
            "VoiceDesign 1.7B", progress, model_id="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        )
        return "VoiceDesign 1.7B model loaded successfully!"
    except Exception as e:
        return f"Error loading model: {type(e).__name__}: {e}"


async def load_voice_design_model_vn(progress=gr.Progress(track_tqdm=True)):
    """Load VoiceDesign 0.6BVN model on-demand."""
    global voice_design_interface_vn
    if voice_design_interface_vn is not None:
        return "Model already loaded."
    try:
        voice_design_interface_vn = await _load_interface(
            "VoiceDesign 0.6BVN", progress, model_id=VN_MODEL_ID
        )
        return "VoiceDesign 0.6BVN model loaded successfully!"
    except Exception as e:
        return f"Error loading model: {type(e).__name__}: {e}"


async def load_base_model_0_6b(progress=gr.Progress(track_tqdm=True)):
    """Load Base 0.6B model on-demand."""
    global base_interface_0_6b
    if base_interface_0_6b is not None:
        return "Model already loaded."
    try:
        base_interface_0_6b = await _load_interface(
            "Base 0.6B", progress, model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        )
        return "Base 0.6B model loaded successfully!"
    except Exception as e:
        return f"Error loading model: {type(e).__name__}: {e}"


async def load_base_model_1_7b(progress=gr.Progress(track_tqdm=True)):
    """Load Base 1.7B model on-demand."""
    global base_interface_1_7b
    if base_interface_1_7b is not None:
        return "Model already loaded."
    try:
        base_interface_1_7b = await _load_interface(
            "Base 1.7B", progress, model_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )
        return "Base 1.7B model loaded successfully!"
    except Exception as e:
        return f"Error loading model: {type(e).__name__}: {e}"


async def load_base_model_vn(progress=gr.Progress(track_tqdm=True)):
    """Load Base 0.6BVN model on-demand."""
    global base_interface_vn
    if base_interface_vn is not None:
        return "Model already loaded."
    try:
        base_interface_vn = await _load_interface("Base 0.6BVN", progress, model_id=VN_MODEL_ID)
        return "Base 0.6BVN model loaded successfully!"
    except Exception as e:
        return f"Error loading model: {type(e).__name__}: {e}"


async def load_custom_voice_model_0_6b(progress=gr.Progress(track_tqdm=True)):
    """Load CustomVoice 0.6B model on-demand."""
    global custom_voice_interface_0_6b
    if custom_voice_interface_0_6b is not None:
        return "Model already loaded."
    try:
        custom_voice_interface_0_6b = await _load_interface(
            "CustomVoice 0.6B", progress, model_id="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        )
        return "CustomVoice 0.6B model loaded successfully!"
    except Exception as e:
        return f"Error loading model: {type(e).__name__}: {e}"


async def load_custom_voice_model_1_7b(progress=gr.Progress(track_tqdm=True)):
    """Load CustomVoice 1.7B model on-demand."""
    global custom_voice_interface_1_7b
    if custom_voice_interface_1_7b is not None:
        return "Model already loaded."
    try:
        custom_voice_interface_1_7b = await _load_interface(
            "CustomVoice 1.7B", progress, model_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        )
        return "CustomVoice 1.7B model loaded successfully!"
    except Exception as e:
        return f"Error loading model: {type(e).__name__}: {e}"


async def load_custom_voice_model_vn(progress=gr.Progress(track_tqdm=True)):
    """Load CustomVoice 0.6BVN model on-demand."""
    global custom_voice_interface_vn
    if custom_voice_interface_vn is not None:
        return "Model already loaded."
    try:
        custom_voice_interface_vn = await _load_interface(
            "CustomVoice 0.6BVN", progress, model_id=VN_MODEL_ID
        )
        return "CustomVoice 0.6BVN model loaded successfully!"
    except Exception as e:
        return f"Error loading model: {type(e).__name__}: {e}"


# ============================================================================


def _normalize_audio(wav, eps=1e-12, clip=True):
    """Normalize audio to float32 in [-1, 1] range."""
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def _audio_to_tuple(audio):
    """Convert Gradio audio input to (wav, sr) tuple."""
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    return None


async def generate_voice_design(text, language, voice_description, progress=gr.Progress(track_tqdm=True)):
    """Generate speech using Voice Design model."""
    global voice_design_interface_0_6b, voice_design_interface_1_7b, voice_design_interface_vn

    interface = voice_design_interface_0_6b or voice_design_interface_1_7b or voice_design_interface_vn

    if interface is None:
        return None, "Error: Model not loaded. Please click 'Load Model' first."

    if not text or not text.strip():
        return None, "Error: Text is required."
    if not voice_description or not voice_description.strip():
        return None, "Error: Voice description is required."

    try:
        audio_codes = [
            chunk
            async for chunk in interface.generate_voice_design_async(
                text=text.strip(),
                instruct=voice_description.strip(),
                language=language,
            )
        ]
        wavs, sr = interface.speech_tokenizer.decode([{"audio_codes": audio_codes}])
        return (sr, wavs[0]), "Voice design generation completed successfully!"
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"


async def generate_voice_clone(ref_audio, ref_text, target_text, language, use_xvector_only, progress=gr.Progress(track_tqdm=True)):
    """Generate speech using Base (Voice Clone) model."""
    global base_interface_0_6b, base_interface_1_7b, base_interface_vn

    interface = base_interface_0_6b or base_interface_1_7b or base_interface_vn

    if interface is None:
        return None, "Error: Model not loaded. Please click 'Load Model' first."

    if not target_text or not target_text.strip():
        return None, "Error: Target text is required."

    audio_tuple = _audio_to_tuple(ref_audio)
    if audio_tuple is None:
        return None, "Error: Reference audio is required."

    if not use_xvector_only and (not ref_text or not ref_text.strip()):
        return None, "Error: Reference text is required when 'Use x-vector only' is not enabled."

    try:
        voice_clone_prompt = interface.create_voice_clone_prompt(
            ref_audio=audio_tuple,
            ref_text=ref_text.strip() if ref_text else None,
            x_vector_only_mode=use_xvector_only,
        )
        audio_codes = [
            chunk
            async for chunk in interface.generate_voice_clone_async(
                text=target_text.strip(),
                language=language,
                voice_clone_prompt=voice_clone_prompt,
            )
        ]
        wavs, sr = interface.speech_tokenizer.decode([{"audio_codes": audio_codes}])
        return (sr, wavs[0]), "Voice clone generation completed successfully!"
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"


async def generate_custom_voice(text, language, speaker, progress=gr.Progress(track_tqdm=True)):
    """Generate speech using CustomVoice model."""
    global custom_voice_interface_0_6b, custom_voice_interface_1_7b, custom_voice_interface_vn

    interface = custom_voice_interface_0_6b or custom_voice_interface_1_7b or custom_voice_interface_vn

    if interface is None:
        return None, "Error: Model not loaded. Please click 'Load Model' first."

    if not text or not text.strip():
        return None, "Error: Text is required."
    if not speaker:
        return None, "Error: Speaker is required."

    try:
        audio_codes = [
            chunk
            async for chunk in interface.generate_custom_voice_async(
                text=text.strip(),
                language=language,
                speaker=speaker.lower(),
            )
        ]
        wavs, sr = interface.speech_tokenizer.decode([{"audio_codes": audio_codes}])
        return (sr, wavs[0]), "Generation completed successfully!"
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"


# Build Gradio UI
def build_ui():
    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
    )

    css = """
    .gradio-container {max-width: none !important;}
    .tab-content {padding: 20px;}
    """

    with gr.Blocks(theme=theme, css=css, title="Qwen3-TTS Demo (nano-qwen3tts-vllm)") as demo:
        gr.Markdown(
            """
# Qwen3-TTS Demo (nano-qwen3tts-vllm)
A unified Text-to-Speech demo featuring three powerful modes:
- **Voice Design**: Create custom voices using natural language descriptions
- **Voice Clone (Base)**: Clone any voice from a reference audio
- **TTS (CustomVoice)**: Generate speech with predefined speakers
Built on [nano-qwen3tts-vllm](https://github.com/tsdocode/nano-qwen3tts-vllm), a nano-vllm-style serving engine for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Qwen Team.

**Note:** Each loaded model spawns its own Talker + Predictor worker processes. Loading several
tabs/sizes at once runs multiple process pairs concurrently on the GPU.
"""
        )

        with gr.Tabs():
            # Tab 1: Voice Design
            with gr.Tab("Voice Design"):
                gr.Markdown("### Create Custom Voice with Natural Language")
                gr.Markdown("**Note:** Click 'Load Model' before generating audio.")
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            design_load_btn_0_6b = gr.Button("Load 0.6B", variant="secondary")
                            design_load_btn_1_7b = gr.Button("Load 1.7B", variant="secondary")
                            design_load_btn_0_6b_vn = gr.Button("Load 0.6BVN", variant="secondary")
                        design_load_status = gr.Textbox(label="Load Status", lines=1, interactive=False)
                        design_text = gr.Textbox(
                            label="Text to Synthesize",
                            lines=4,
                            placeholder="Enter the text you want to convert to speech...",
                            value="It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!",
                            interactive=False,
                        )
                        design_language = gr.Dropdown(
                            label="Language",
                            choices=LANGUAGES,
                            value="Auto",
                            interactive=False,
                        )
                        design_instruct = gr.Textbox(
                            label="Voice Description",
                            lines=3,
                            placeholder="Describe the voice characteristics you want...",
                            value="Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice.",
                            interactive=False,
                        )
                        design_btn = gr.Button("Generate with Custom Voice", variant="primary", interactive=False)

                    with gr.Column(scale=2):
                        design_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                        design_status = gr.Textbox(label="Status", lines=2, interactive=False)

                design_load_btn_0_6b.click(
                    load_voice_design_model_0_6b,
                    outputs=[design_load_status],
                ).then(
                    lambda: (
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                    ),
                    outputs=[design_text, design_language, design_instruct, design_btn],
                )

                design_load_btn_1_7b.click(
                    load_voice_design_model_1_7b,
                    outputs=[design_load_status],
                ).then(
                    lambda: (
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                    ),
                    outputs=[design_text, design_language, design_instruct, design_btn],
                )

                design_load_btn_0_6b_vn.click(
                    load_voice_design_model_vn,
                    outputs=[design_load_status],
                ).then(
                    lambda: (
                        gr.update(interactive=True),
                        gr.update(choices=LANGUAGES_VN, value="Vietnamese", interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                    ),
                    outputs=[design_text, design_language, design_instruct, design_btn],
                )

                design_btn.click(
                    generate_voice_design,
                    inputs=[design_text, design_language, design_instruct],
                    outputs=[design_audio_out, design_status],
                )

            # Tab 2: Voice Clone (Base)
            with gr.Tab("Voice Clone (Base)"):
                gr.Markdown("### Clone Voice from Reference Audio")
                gr.Markdown("**Note:** Click 'Load Model' before generating audio.")
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            clone_load_btn_0_6b = gr.Button("Load 0.6B", variant="secondary")
                            clone_load_btn_1_7b = gr.Button("Load 1.7B", variant="secondary")
                            clone_load_btn_0_6b_vn = gr.Button("Load 0.6BVN", variant="secondary")
                        clone_load_status = gr.Textbox(label="Load Status", lines=1, interactive=False)
                        clone_ref_audio = gr.Audio(
                            label="Reference Audio (Upload a voice sample to clone)",
                            type="numpy",
                            interactive=False,
                        )
                        clone_ref_text = gr.Textbox(
                            label="Reference Text (Transcript of the reference audio)",
                            lines=2,
                            placeholder="Enter the exact text spoken in the reference audio...",
                            interactive=False,
                        )
                        clone_xvector = gr.Checkbox(
                            label="Use x-vector only (No reference text needed, but lower quality)",
                            value=False,
                            interactive=False,
                        )

                    with gr.Column(scale=2):
                        clone_target_text = gr.Textbox(
                            label="Target Text (Text to synthesize with cloned voice)",
                            lines=4,
                            placeholder="Enter the text you want the cloned voice to speak...",
                            interactive=False,
                        )
                        clone_language = gr.Dropdown(
                            label="Language",
                            choices=LANGUAGES,
                            value="Auto",
                            interactive=False,
                        )
                        clone_btn = gr.Button("Clone & Generate", variant="primary", interactive=False)

                with gr.Row():
                    clone_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                    clone_status = gr.Textbox(label="Status", lines=2, interactive=False)

                clone_load_btn_0_6b.click(
                    load_base_model_0_6b,
                    outputs=[clone_load_status],
                ).then(
                    lambda: (
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                    ),
                    outputs=[clone_ref_audio, clone_ref_text, clone_xvector, clone_target_text, clone_language, clone_btn],
                )

                clone_load_btn_1_7b.click(
                    load_base_model_1_7b,
                    outputs=[clone_load_status],
                ).then(
                    lambda: (
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                    ),
                    outputs=[clone_ref_audio, clone_ref_text, clone_xvector, clone_target_text, clone_language, clone_btn],
                )

                clone_load_btn_0_6b_vn.click(
                    load_base_model_vn,
                    outputs=[clone_load_status],
                ).then(
                    lambda: (
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(choices=LANGUAGES_VN, value="Vietnamese", interactive=True),
                        gr.update(interactive=True),
                    ),
                    outputs=[clone_ref_audio, clone_ref_text, clone_xvector, clone_target_text, clone_language, clone_btn],
                )

                clone_btn.click(
                    generate_voice_clone,
                    inputs=[clone_ref_audio, clone_ref_text, clone_target_text, clone_language, clone_xvector],
                    outputs=[clone_audio_out, clone_status],
                )

            # Tab 3: TTS (CustomVoice)
            with gr.Tab("TTS (CustomVoice)"):
                gr.Markdown("### Text-to-Speech with Predefined Speakers")
                gr.Markdown(
                    "**Note:** Click 'Load Model' before generating audio. "
                    "Style instructions aren't supported by nano-qwen3tts-vllm's CustomVoice path yet."
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            tts_load_btn_0_6b = gr.Button("Load 0.6B", variant="secondary")
                            tts_load_btn_1_7b = gr.Button("Load 1.7B", variant="secondary")
                            tts_load_btn_0_6b_vn = gr.Button("Load 0.6BVN", variant="secondary")
                        tts_load_status = gr.Textbox(label="Load Status", lines=1, interactive=False)
                        tts_text = gr.Textbox(
                            label="Text to Synthesize",
                            lines=4,
                            placeholder="Enter the text you want to convert to speech...",
                            value="Hello! Welcome to Text-to-Speech system. This is a demo of our TTS capabilities.",
                            interactive=False,
                        )
                        tts_language = gr.Dropdown(
                            label="Language",
                            choices=LANGUAGES,
                            value="English",
                            interactive=False,
                        )
                        tts_speaker = gr.Dropdown(
                            label="Speaker",
                            choices=SPEAKERS,
                            value="Ryan",
                            interactive=False,
                        )
                        tts_btn = gr.Button("Generate Speech", variant="primary", interactive=False)

                    with gr.Column(scale=2):
                        tts_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                        tts_status = gr.Textbox(label="Status", lines=2, interactive=False)

                tts_load_btn_0_6b.click(
                    load_custom_voice_model_0_6b,
                    outputs=[tts_load_status],
                ).then(
                    lambda: (
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                    ),
                    outputs=[tts_text, tts_language, tts_speaker, tts_btn],
                )

                tts_load_btn_1_7b.click(
                    load_custom_voice_model_1_7b,
                    outputs=[tts_load_status],
                ).then(
                    lambda: (
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                    ),
                    outputs=[tts_text, tts_language, tts_speaker, tts_btn],
                )

                tts_load_btn_0_6b_vn.click(
                    load_custom_voice_model_vn,
                    outputs=[tts_load_status],
                ).then(
                    lambda: (
                        gr.update(interactive=True),
                        gr.update(choices=LANGUAGES_VN, value="Vietnamese", interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                    ),
                    outputs=[tts_text, tts_language, tts_speaker, tts_btn],
                )

                tts_btn.click(
                    generate_custom_voice,
                    inputs=[tts_text, tts_language, tts_speaker],
                    outputs=[tts_audio_out, tts_status],
                )

        gr.Markdown(
            """
---
**Note**: Models are loaded on-demand when you click the "Load Model" button. Each model
runs its own Talker + Predictor worker processes for the lifetime of the app.
"""
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
