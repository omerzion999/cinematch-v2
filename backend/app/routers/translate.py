"""POST /api/translate - batch-translate free-form chat text for the language toggle."""

import json
import re

from fastapi import APIRouter
from pydantic import BaseModel

from app.agent.llm import _call_llm, _get_client

router = APIRouter()


class TranslateRequest(BaseModel):
    texts: list[str]
    target_lang: str


class TranslateResponse(BaseModel):
    translations: list[str]


_TRANSLATE_SYSTEM = """\
You are a translator for a TV/movie recommendation chatbot UI.
Translate each string in the given JSON array into {target}.
Preserve TV/movie show titles, proper nouns, numbers, and emojis exactly as-is.
Keep the tone and formatting of each string.
Reply with ONLY a JSON array of strings, the same length and order as the input, with no extra commentary.
"""


@router.post("/api/translate", response_model=TranslateResponse)
def translate(payload: TranslateRequest) -> TranslateResponse:
    if not payload.texts:
        return TranslateResponse(translations=[])

    if not _get_client():
        return TranslateResponse(translations=payload.texts)

    target_name = "Hebrew" if payload.target_lang == "he" else "English"
    system = _TRANSLATE_SYSTEM.format(target=target_name)
    user = json.dumps(payload.texts, ensure_ascii=False)

    raw = _call_llm(system, user, max_tokens=1500)
    if not raw:
        return TranslateResponse(translations=payload.texts)

    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw)

    try:
        translations = json.loads(raw)
    except json.JSONDecodeError:
        return TranslateResponse(translations=payload.texts)

    if not isinstance(translations, list) or len(translations) != len(payload.texts):
        return TranslateResponse(translations=payload.texts)

    return TranslateResponse(translations=[str(item) for item in translations])
