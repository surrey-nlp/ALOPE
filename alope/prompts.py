"""
Prompt templates for QE scoring.
"""

from typing import Dict

LANGUAGE_MAP: Dict[str, str] = {
    "en": "English",
    "ta": "Tamil",
    "te": "Telugu",
    "hi": "Hindi",
    "gu": "Gujarati",
    "mr": "Marathi",
    "ne": "Nepali",
    "si": "Sinhala",
    "et": "Estonian",
    "de": "German",
    "zh": "Chinese",
    "fr": "French",
    "es": "Spanish",
}


def render_prompt(template: str, source: str, target: str, src_lang_code: str, tgt_lang_code: str) -> str:
    src_lang = LANGUAGE_MAP.get(src_lang_code.lower(), src_lang_code)
    tgt_lang = LANGUAGE_MAP.get(tgt_lang_code.lower(), tgt_lang_code)
    return template.format(
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        source=source,
        target=target,
    )

