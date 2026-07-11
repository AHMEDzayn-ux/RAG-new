"""Small text helpers shared across API/service layers."""

import re

# Emoji + pictographic symbols + variation selectors / ZWJ. Stripped from any
# text that gets spoken (TTS reads "👋" as "waving hand") or shown to customers.
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"   # symbols & pictographs, emoji, supplemental
    "\U00002600-\U000027BF"   # misc symbols + dingbats (✨ ✅ ❌ …)
    "\U0001F1E6-\U0001F1FF"   # regional indicators (flags)
    "\U00002190-\U000021FF"   # arrows
    "\U00002B00-\U00002BFF"   # misc symbols & arrows
    "\U0000FE00-\U0000FE0F"   # variation selectors
    "\U0000200D"              # zero-width joiner
    "\U000024C2\U00002122\U00003030"
    "]+",
    flags=re.UNICODE,
)


def strip_emojis(text: str) -> str:
    """Remove emoji/pictographs and tidy the whitespace they leave behind."""
    if not text:
        return text
    t = _EMOJI_RE.sub("", text)
    t = re.sub(r"[ \t]{2,}", " ", t)
    # Clean up " ! " / " ." / doubled spaces left where an emoji sat.
    t = re.sub(r"\s+([,.!?])", r"\1", t)
    return t.strip()
