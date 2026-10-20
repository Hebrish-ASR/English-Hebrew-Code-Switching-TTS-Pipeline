"""Stages 2-5 of the deterministic English+Hebrew code-switching TTS pipeline: everything before
the TTS call. Reference implementation, CPU-friendly. Run:

    uv run --with torch --with "transformers<5" --with sentencepiece python pipeline/text_stages.py \
        "I need to go to Bituach Leumi before the makolet closes."

Prints the intermediate result of every stage as JSON (the provider-neutral contract the TTS
stage consumes; see docs/pipeline.md). Models, all from the Hugging Face Hub:
  2 detect         Hebrish/Hebrew-Latin-Token-Classifier   (MIT)
  3 transliterate  LEXICON below, then malper/taatiknet    (weights: no licence stated; data CC-BY-SA 3.0)
  4 niqqud         dicta-il/dictabert-large-char-menaked   (CC-BY-4.0), for unpointed lexicon forms
Stage 5 (segment + tag) is plain code. Stage 6 (speak) is in pipeline/speak_chatterbox.py.
"""
import importlib.util, json, re, sys, unicodedata

from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer, pipeline

DETECTOR = "Hebrish/Hebrew-Latin-Token-Classifier"
TRANSLIT = "malper/taatiknet"
NAKDAN = "dicta-il/dictabert-large-char-menaked"

# Stage 3, primary: a curated lexicon of the terms you expect (unpointed Hebrew). TaatikNet is the
# fallback because it works one word at a time, without context, and does not know spelling
# conventions. In production, build this from a real term list (e.g. the Hebrish lexicon).
LEXICON = {
    "bituach leumi": "ביטוח לאומי",
    "makolet": "מכולת",
    "arnona": "ארנונה",
    "mas hachnasa": "מס הכנסה",
}
NIQQUD = re.compile(r"[֑-ׇ]")


def detect(text, threshold=0.5):
    """Stage 2: character spans of romanized Hebrew.

    Uses predict.py from the model repository, not the transformers pipeline: it pre-splits words
    the way the training data was split and labels each word from its first sub-token. The stock
    pipeline turns sub-tokens into stray one-letter spans ("m" + "mishpacha"; seen 2026-10-20).
    """
    spec = importlib.util.spec_from_file_location("predict", hf_hub_download(DETECTOR, "predict.py"))
    predict = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predict)
    return [{"start": h["start"], "end": h["end"], "surface": h["text"], "score": h["score"]}
            for h in predict.HebrewSpans(DETECTOR)(text, threshold=threshold)]


def transliterate(spans):
    """Stage 3: Hebrew script per span, lexicon first, TaatikNet per word as the fallback."""
    t2t = None
    for s in spans:
        key = s["surface"].lower()
        if key in LEXICON:
            s["hebrew"], s["source"] = LEXICON[key], "lexicon"
            continue
        t2t = t2t or pipeline("text2text-generation", model=TRANSLIT)
        words = [unicodedata.normalize("NFC", w) for w in s["surface"].split()]
        out = t2t(words, max_length=200, num_beams=5)
        # One result per word: a dict, or a list of dicts if num_return_sequences > 1.
        best = [(o[0] if isinstance(o, list) else o)["generated_text"] for o in out]
        s["hebrew"], s["source"] = " ".join(best), "taatiknet"
    return spans


def add_niqqud(spans):
    """Stage 4: vowel points for forms that lack them (TaatikNet output already has niqqud)."""
    todo = [s for s in spans if not NIQQUD.search(s["hebrew"])]
    if todo:
        tok = AutoTokenizer.from_pretrained(NAKDAN)
        model = AutoModel.from_pretrained(NAKDAN, trust_remote_code=True).eval()
        # Default output drops the vowel letters (ktiv haser): standard pointed spelling, e.g.
        # מכולת -> מַכֹּלֶת. Needs transformers<5: 5.x loads this char tokenizer as word-level and
        # predict() then returns the input unpointed (seen 2026-10-20 with 5.20.2).
        for s, pointed in zip(todo, model.predict([s["hebrew"] for s in todo], tok)):
            s["pointed"] = pointed
    for s in spans:
        s.setdefault("pointed", s["hebrew"])
    return spans


def segments(text, spans):
    """Stage 5: split at every language switch; one (text, language) pair per piece."""
    out, pos = [], 0
    for s in sorted(spans, key=lambda s: s["start"]):
        if s["start"] > pos:
            out.append({"text": text[pos:s["start"]], "lang": "en"})
        out.append({"text": s["pointed"], "lang": "he"})
        pos = s["end"]
    if pos < len(text):
        out.append({"text": text[pos:], "lang": "en"})
    return out


def run(text):
    spans = add_niqqud(transliterate(detect(text)))
    return {"text": text, "spans": spans, "segments": segments(text, spans)}


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "I need to go to Bituach Leumi before the makolet closes."
    print(json.dumps(run(text), ensure_ascii=False, indent=2))
