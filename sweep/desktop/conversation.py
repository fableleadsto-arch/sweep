"""Small explicit companion replies that do not need model inference."""
from __future__ import annotations

import re


def quick_reply(text: str) -> str | None:
    value = re.sub(r"[.!?,]+$", "", text.strip().casefold()).strip()
    if re.fullmatch(r"(?:hi|hello|hey)(?: sweep)?", value):
        return "Hi, I'm here. Tell me what's on your mind, or ask me to open an app, read your screen, or work with a file."
    if value in {"thanks", "thank you", "thanks sweep", "thank you sweep"}:
        return "You're welcome. I'm here when you need me."
    if value in {"what can you do", "what can you help me with", "how can you help me", "help me get started"}:
        return ("You can chat with me here, or ask for a task:\n"
                "• Open YouTube in Brave, or open Calculator.\n"
                "• Read my screen, then ask about the text I found.\n"
                "• Search the web or read a public page.\n"
                "• Attach a document to summarize, an image to inspect, or a CSV to clean and convert.\n\n"
                "Chat and file processing stay on this device. I'll ask before screen capture, app control, or web access.")
    return None
