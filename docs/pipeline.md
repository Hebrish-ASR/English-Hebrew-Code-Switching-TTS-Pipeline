# The pipeline in detail

Every fact below was read in the cited source on **2026-10-20** unless marked *inferred*.
"Run" means we executed it for this repository on that date (CPU, transformers 4.57.6).

## Stage 1: the script

Any text in English with Hebrew words in Latin letters. The worked example is the
[My Weird Prompts](https://myweirdprompts.com) podcast, whose LLM script agent writes English and
romanizes Hebrew. Nothing is required of the writer; the pipeline takes the text as it comes.

**Why this needs a pipeline at all:** script-based language detection cannot see the switch.
Hebrew letters have their own Unicode block (U+0590 to U+05FF), so Hebrew *script* inside English
is trivial to find. Romanized Hebrew is made of the same Basic Latin characters as the English
around it. GlotLID, fastText `lid.176` and `lid218e` have no `heb_Latn` label, and sentence-level
language ID would call the sentence English anyway. A TTS pipeline needs character offsets per
word: token classification.

## Stage 2: detect (run)

- Model: [`Hebrish/Hebrew-Latin-Token-Classifier`](https://huggingface.co/Hebrish/Hebrew-Latin-Token-Classifier),
  MIT. Same weights as the original [`danielrosehill/hebrew-latin-token-classifier`](https://huggingface.co/danielrosehill/hebrew-latin-token-classifier)
  (DOI [10.57967/hf/10997](https://doi.org/10.57967/hf/10997)).
- Labels `O`, `B-HE`, `I-HE`; multi-word terms come out as one span. Span F1 0.941 on a
  term-disjoint test set (no test term seen in training); 0 false positives on 58 hard negatives.
- Labelling policy: pronunciation, not etymology. *Shabbat*, *kosher*, *hummus* are not flagged,
  because English already says them.
- **Use the repository's `predict.py`**, as `text_stages.py` does (it downloads and imports it).
  The stock transformers pipeline produced stray one-letter spans here ("m" next to
  "mishpacha", "b" next to "bamba") because it labels sub-tokens; `predict.py` pre-splits words
  the way the training data was split and labels each word from its first sub-token.
- Precision matters more than recall downstream (see "Cost" below); raise `threshold` for fewer
  false positives.

## Stage 3: transliterate to Hebrew script (run)

- **Primary: a curated lexicon** of expected terms with their standard Hebrew spelling. The demo
  `LEXICON` in `text_stages.py` has four entries; in production use a real term list.
- **Fallback: [TaatikNet](https://huggingface.co/malper/taatiknet)** (`malper/taatiknet`, ByT5,
  code at [morrisalp/taatiknet](https://github.com/morrisalp/taatiknet)). Bidirectional with no
  prefix: the script of the input sets the direction (*inferred* from the README and the official
  Space; no prefix appears in either). Trained on about 15K single Wiktionary words, so it is
  **word-level**: the README says to split on whitespace. Its Hebrew output **already carries
  niqqud**. Licence: none declared on the model card; the repository's data is CC-BY-SA 3.0 / GFDL.
- **Why the lexicon comes first, from our run:** TaatikNet wrote *savta* as סַוְטָה (correct:
  סַבְתָּא) and *iriya* as אִירִיָּה (correct: עִירִיָּה). It guesses each word without context
  or spelling conventions. It got *mishpacha* (מִשְׁפָּחָה) and *bamba* (בַּמְבָּה) right.
- Trap: with `num_return_sequences=1` the pipeline returns one dict per word, with more it returns
  a list per word. `text_stages.py` handles both.

## Stage 4: niqqud (run)

- **[DICTA nakdan](https://huggingface.co/dicta-il/dictabert-large-char-menaked)**
  (`dicta-il/dictabert-large-char-menaked`, CC-BY-4.0, `trust_remote_code=True`). Used for
  forms without niqqud, i.e. lexicon entries; TaatikNet's output is passed through.
- Its default output drops the vowel letters (ktiv haser), the standard pointed spelling:
  מכולת → מַכֹּלֶת, ביטוח לאומי → בִּטּוּחַ לְאֻמִּי. With `mark_matres_lectionis` it keeps
  them, which gives forms such as לְאֻומִּי with a bare vav; we use the default.
- **Trap: needs `transformers<5`.** Under transformers 5.20.2 the tokenizer loads as a
  word-level `BertTokenizer` (one `[UNK]` per word) and `predict()` returns the input without
  any niqqud, and once with a word duplicated. Under 4.57.6 it loads as the character-level
  `BertTokenizerFast` and works.
- Alternative: [Phonikud](https://github.com/thewh1teagle/phonikud) (code CC-BY-4.0; weights
  `Phonikud/phonikud-onnx`, card says MIT). It adds niqqud plus non-standard marks for stress
  (U+05AB) and vocal shva (U+05BD), and its `phonemize()` turns that into IPA. Strip the
  non-standard marks before a TTS engine that takes ordinary pointed text; use Phonikud when the
  engine takes phonemes (e.g. [phonikud-tts](https://github.com/thewh1teagle/phonikud-tts), a
  Piper engine, CC-BY-NC 4.0: research and education only).

## Stage 5: tag (run)

String work. Walk the spans in order; text between spans is `en`, each span's pointed form is
`he`. Output: an ordered list of `{"text", "lang"}` segments (see `examples/`). Keep this
representation provider-neutral; only stage 6 knows about a particular engine.

## Stage 6: speak (written, not run)

**Chatterbox Multilingual** ([resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox),
`pip install chatterbox-tts`, 0.1.7, MIT):

- `ChatterboxMultilingualTTS.from_pretrained(device, t3_model="v3")`; the README recommends v3.
- `generate(text, language_id, audio_prompt_path=None, exaggeration=0.5, cfg_weight=0.5, ...)`
  returns a `(1, N)` tensor at `model.sr` = 24000 Hz, watermarked (Perth).
- `"he"` is in `SUPPORTED_LANGUAGES` (23 languages). `language_id` becomes a prefix token for the
  whole call. **There is no inline switch**, so voice each segment separately and join them.
- Use the same `audio_prompt_path` for every segment so the speaker does not change mid-sentence.
- The README says that when the reference clip's language differs from `language_id`, accent
  bleeds across, and `cfg_weight=0` reduces it. One English voice speaking Hebrew segments is
  that case.
- **Trap: your niqqud may be replaced.** For `he`, `MTLTokenizer.encode` calls
  `add_hebrew_diacritics()`, which runs `dicta_onnx.Dicta()` and would strip the input's niqqud
  and re-diacritize. If `dicta_onnx` is not installed it warns and passes the text through.
  Leave it uninstalled so stage 4's niqqud reaches the model. (*Inferred:* with current
  dicta-onnx, `Dicta()` without a model path raises, the error is caught, and the text passes
  through too. Do not rely on that.)
- Padding between clips: `PAD_SECONDS = 0.06` is a starting guess. Joins without padding sound
  like cuts; tune by ear.

**Hosted engines with SSML `<lang>`:**

| Engine | Hebrew inline? | Source |
| --- | --- | --- |
| Azure | **Yes, multilingual voices only.** `he-IL` is listed for `en-US-{Andrew,Ava,Brian,Emma}MultilingualNeural`; non-multilingual voices do not support `<lang>`. | [speech-synthesis-markup-voice](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice) |
| Google Cloud TTS | **No.** `<lang>` is best effort and Hebrew is documented as unsupported ("will result in silence"); switch voices with `<voice>` instead. | [SSML](https://docs.cloud.google.com/text-to-speech/docs/ssml) |
| Amazon Polly | **No Hebrew** in its language list. | [supported languages](https://docs.aws.amazon.com/polly/latest/dg/supported-languages.html) |

## Cost, and why the detector favours precision

Every detected span costs a sentence split, an extra TTS call and an audio seam. A missed Hebrew
word costs nothing: it is read as English, as today. A false positive pays the full cost to make
the output worse, an English word in a Hebrew voice. Hence a conservative labelling policy and a
`threshold`.

## Deterministic versus multilingual mid-sentence switching

The deterministic pipeline decides every switch in code, before the TTS sees the text. The
alternative is one multilingual model given the raw sentence and trusted to switch on its own.
Romanized Hebrew gives such a model no script signal, there is no handle to fix one word, and
its decisions cannot be tested. A TTS model trained on code-switched English-Hebrew speech could
change that; we have not found an open one (searched 2026-10-20, not exhaustively).

## Not done yet

- Stage 6 has not been run end to end; no audio is published here.
- Nobody has packaged the stages as one service (container, Replicate model, Space).
- The lexicon in `text_stages.py` is a demo. A real one would come from a curated term list.
- Open question from the planning notes: how well Chatterbox's `he` voice reads pointed versus
  unpointed Hebrew has not been listened to.
