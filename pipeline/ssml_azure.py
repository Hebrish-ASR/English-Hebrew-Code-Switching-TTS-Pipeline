"""Stage 6, hosted alternative: one SSML document with inline Hebrew for an Azure multilingual
voice. Prints the SSML; send it with the Azure Speech SDK or REST API.

    python pipeline/ssml_azure.py examples/bituach-leumi.json

Checked 2026-10-20 in Microsoft's SSML docs (speech-synthesis-markup-voice): <lang xml:lang> works
only inside a multilingual voice; he-IL is listed for en-US-{Andrew,Ava,Brian,Emma}MultilingualNeural.
Not sent to Azure from here. Google Cloud TTS documents Hebrew in <lang> as unsupported (silence),
and Amazon Polly has no Hebrew, so neither works with this file.
"""
import json, sys
from xml.sax.saxutils import escape

VOICE = "en-US-AvaMultilingualNeural"


def ssml(segments, voice=VOICE):
    body = "".join(escape(s["text"]) if s["lang"] == "en" else f'<lang xml:lang="he-IL">{escape(s["text"])}</lang>'
                   for s in segments)
    return (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
            f'<voice name="{voice}">{body}</voice></speak>')


if __name__ == "__main__":
    print(ssml(json.load(open(sys.argv[1], encoding="utf-8"))["segments"]))
