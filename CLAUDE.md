# CLAUDE.md

Reference repository for the deterministic English+Hebrew code-switching TTS pipeline (detect,
transliterate, niqqud, tag, speak). Read `README.md`, then `docs/pipeline.md` (every fact there is
sourced and dated). Created 2026-10-20 under the Hebrish-ASR org; linked from
https://hebrish.app/models/token-classifier.

- `examples/*.json` are real output of `pipeline/text_stages.py`; regenerate them, never hand-edit.
- `pipeline/speak_chatterbox.py` has not been run. Say so until it has; then publish audio and
  update the README status, `docs/pipeline.md` "Not done yet", and the white paper.
- Figures: `figures/hero.png` is AI-generated and deliberately carries no Hebrew (image models
  misspelled it: לנומי for לאומי, 2026-10-20). Exact diagrams are `figures/src/diagrams.html`,
  rendered with the commands in its header; Hebrew in them is copied from `examples/` output.
- White paper: edit `whitepaper/code-switching-tts-pipeline.html`, render the PDF with the command
  in its header, and copy the PDF to Hebrish-App `whitepaper/` (the site serves it).
- Stage 4 needs `transformers<5` (DICTA tokenizer); keep the pin.
