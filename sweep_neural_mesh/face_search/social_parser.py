"""Social/web platform parsing — classify result URLs.

openface-search's social_parser.py idea, extended: instead of a fixed social
list, every result is classified into a platform with a person-profile
heuristic so Sweep covers **all sources** (social + news + blogs + companies),
which is what a web-intelligence platform needs.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

# Platform registry: (display name, domain regex, profile-path regex)
# Profile-path regexes encode what a *person* profile URL looks like on each
# platform; everything else on the domain is still tagged with the platform
# but flagged is_profile=False (a company page, a post, a hashtag page...).
PLATFORMS: list[tuple[str, str, str]] = [
    ("linkedin", r"(^|\.)linkedin\.com$", r"^/in/[^/]+/?$"),
    ("instagram", r"(^|\.)instagram\.com$", r"^/[^/]+/?$"),
    ("facebook", r"(^|\.)(facebook|fb)\.com$", r"^/[^/]+/?$"),
    ("x", r"(^|\.)(x\.com|twitter\.com)$", r"^/[^/]+/?$"),
    ("tiktok", r"(^|\.)tiktok\.com$", r"^/@[^/]+/?$"),
    ("youtube", r"(^|\.)youtube\.com$", r"^/(@[^/]+|c/[^/]+|channel/[^/]+|user/[^/]+)/?$"),
    ("pinterest", r"(^|\.)pinterest\.[a-z.]+$", r"^/[^/]+/?$"),
    ("reddit", r"(^|\.)reddit\.com$", r"^/(user|u)/[^/]+/?$"),
    ("github", r"(^|\.)github\.com$", r"^/[^/]+/?$"),
    ("medium", r"(^|\.)medium\.com$", r"^/@[^/]+/?$"),
    ("threads", r"(^|\.)threads\.net$", r"^/@[^/]+/?$"),
    ("bluesky", r"(^|\.)bsky\.app$", r"^/profile/[^/]+/?$"),
    ("mastodon", r"(^|\.)(mastodon\.[a-z.]+|fosstodon\.org)$", r"^/@[^/]+/?$"),
    ("substack", r"(^|\.)substack\.com$", r"^/profile/[^/]+/?$"),
    ("aboutme", r"(^|\.)about\.me$", r"^/[^/]+/?$",
     ),
    ("gravatar", r"(^|\.)gravatar\.com$", r"^/[^/]+/?$"),
    ("flickr", r"(^|\.)flickr\.com$", r"^/people/[^/]+/?$"),
    ("vk", r"(^|\.)vk\.com$", r"^/id\d+/?$|^/[a-z0-9_]+/?$"),
    ("weibo", r"(^|\.)weibo\.com$", r"^/u/[^/]+/?$"),
    ("quora", r"(^|\.)quora\.com$", r"^/profile/[^/]+/?$"),
    ("twitch", r"(^|\.)twitch\.com$", r"^/[^/]+/?$"),
    ("keybase", r"(^|\.)keybase\.io$", r"^/[^/]+/?$"),
]

# Reserved paths that are never a person profile even on matching domains
_RESERVED = {
    "p": "instagram",  # /p/<post>
    "reel": "instagram",
    "explore": "instagram",
    "stories": "instagram",
    "hashtag": "instagram",
    "watch": "facebook",
    "groups": "facebook",
    "pages": "facebook",
    "events": "facebook",
    "search": None,
    "login": None,
    "share": None,
    "status": None,
    "photo": None,
    "post": None,
    "topic": None,
    "company": None,
    "jobs": None,
    "pulse": None,
    "feed": None,
    "home": None,
    "update": None,
    "public": None,
    "languages": None,
    "settings": None,
    "privacy": None,
    "about": None,
    "join": None,
    "intent": None,
}


def classify(url: str) -> tuple[str, bool]:
    """Classify a URL → (platform, is_profile).

    platform is a lowercase name from PLATFORMS, or "web" for everything else.
    is_profile means the URL looks like a *person's* profile page.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return "web", False
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return "web", False
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path or "/"
    segments = [s for s in path.split("/") if s]

    for name, domain_re, profile_re in PLATFORMS:
        if not re.search(domain_re, host):
            continue
        is_profile = bool(re.match(profile_re, path))
        if is_profile and segments and segments[0] in _RESERVED:
            is_profile = False
        return name, is_profile

    return "web", False


def domain(url: str) -> str:
    """Registrable-looking domain (www. stripped)."""
    try:
        host = urlparse(url).netloc.lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host
