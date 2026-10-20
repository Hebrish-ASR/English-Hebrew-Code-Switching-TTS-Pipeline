"""Stage 6: voice the segments from text_stages.py with Chatterbox Multilingual, one call per
segment, same voice prompt throughout, language set per call; then join with short padding.

    uv run --with chatterbox-tts --with torchaudio python pipeline/speak_chatterbox.py \
        examples/bituach-leumi.json ref.wav out.wav

NOT RUN YET (2026-10-20): written from the chatterbox-tts 0.1.7 source (src/chatterbox/mtl_tts.py
and models/tokenizers/tokenizer.py), not executed. Needs a GPU for reasonable speed. Facts it
relies on, all read in the source:
  - ChatterboxMultilingualTTS.from_pretrained(device, t3_model="v3"); generate(text, language_id,
    audio_prompt_path, exaggeration, cfg_weight, ...) returns a (1, N) tensor at model.sr = 24000.
  - "he" is in SUPPORTED_LANGUAGES. language_id becomes a prefix token for the whole call: there is
    no inline switch, hence one call per segment.
  - For "he" the tokenizer tries to re-diacritize with dicta_onnx, stripping any niqqud you pass.
    If dicta_onnx is not installed it warns and keeps your text. Leave it uninstalled so the
    niqqud from stage 4 reaches the model unchanged.
  - The README recommends cfg_weight=0 when the reference clip's language differs from
    language_id (accent bleed). One English voice across both languages is exactly that case.
"""
import json, sys

import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

PAD_SECONDS = 0.06  # a starting guess: butt-joined clips read as cuts; tune by ear


def speak(segments, voice_wav, out_wav, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tts = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
    pad = torch.zeros(1, int(tts.sr * PAD_SECONDS))
    clips = []
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        wav = tts.generate(text, language_id=seg["lang"], audio_prompt_path=voice_wav,
                           cfg_weight=0.0 if seg["lang"] != "en" else 0.5)
        clips += [wav.cpu(), pad]
    ta.save(out_wav, torch.cat(clips[:-1], dim=1), tts.sr)


if __name__ == "__main__":
    contract, voice, out = sys.argv[1:4]
    speak(json.load(open(contract, encoding="utf-8"))["segments"], voice, out)
