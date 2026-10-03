# Face / Reverse-Image Search

Sweep can take a photo of a person (or any image) and search the public
internet for pages where it appears — social profiles across **all platforms**
(LinkedIn, Instagram, Facebook, X, TikTok, YouTube, Reddit, GitHub, Mastodon,
Bluesky, ...) plus news, blogs, wikis and company pages.

This is Sweep's implementation of the patterns proven by two open-source
projects, extended to all sources:

| Upstream | What was adopted |
|----------|------------------|
| [openface-search](https://github.com/jacquesmsfb/openface-search) | Face pipeline (detect → crop → ArcFace embed), social-profile parsing, orchestrator with pluggable sources |
| [selfwatch (reverse-image-search)](https://github.com/ryguyye/reverse-image-search) | Multi-provider fan-out with per-provider failure-as-data, canonical-URL dedupe + cross-provider merge, TinEye HMAC signing, honest provider enablement |

Extensions beyond both upstreams:
- **All sources, not just social** — every result URL is classified into a
  platform *and* flagged as person-profile vs. other page (news, blogs,
  companies, wikis).
- **Keyless mode** — with zero API keys, name-hint search (DuckDuckGo across
  platform site-filters) + Wikimedia Commons search still run.
- **Local face verification** — candidate thumbnails are pulled and compared
  against the query face with ArcFace cosine similarity before ranking.

## Quick start

```bash
# CLI
python -m sweep_neural_mesh.face_search.cli photo.jpg --name "jane dough"
python -m sweep_neural_mesh.face_search.cli https://example.com/pic.jpg --mode image
python -m sweep_neural_mesh.face_search.cli --providers     # provider status

# Chat (after `sweep chat`)
/face-search photo.jpg "jane dough"
/face-search meme.jpg --mode image --no-verify
```

Python API:

```python
from sweep_neural_mesh.face_search import FaceSearchOrchestrator

engine = FaceSearchOrchestrator()
report = engine.run_search("photo.jpg", name_hint="jane dough")
print(report.to_dict())          # full machine-readable result
for m in report.results:
    print(m.platform, m.url, m.confidence, m.verified)
```

## Providers

| Provider | Credentials needed | Input | Notes |
|----------|--------------------|-------|-------|
| `google_lens` | `SERPAPI_KEY` | public image URL | richest visual matches |
| `yandex_images` | `SERPAPI_KEY` | public image URL | strong for faces |
| `bing_reverse_image` | `SERPAPI_KEY` | public image URL | experimental (upstream API retired 2025) |
| `google_images` | `SERPAPI_KEY` | public image URL | classic reverse image |
| `tineye` | `TINEYE_API_KEY` + `TINEYE_PRIVATE_KEY` | URL **or** upload bytes | HMAC-signed direct API |
| `name_search` | none (keyless) | name hint | DDG text search across platform site-filters |
| `commons_search` | none (keyless) | name hint | Wikimedia Commons file usage |

Enablement is honest: `python -m sweep_neural_mesh.face_search.cli --providers`
shows exactly what will run and why anything is disabled.

### Public URL hosting (for URL-based providers)

SerpAPI engines need SerpAPI to fetch the image itself, so the query crop
must be reachable from the public internet. Set:

```
SWEEP_PUBLIC_BASE_URL=https://your-host.example.com
SWEEP_UPLOADS_DIR=/path/served/at/uploads/face_search   # optional
```

The crop is written to the uploads dir and referenced as
`<base>/uploads/face_search/<token>.jpg`; the caller is responsible for
static-serving that path (selfwatch's approach). Files older than one hour
are deleted automatically. Without a public base URL, URL-based providers
are skipped with an explanatory note — the run never fails because of it.

## Architecture

```
sweep_neural_mesh/face_search/
├── models.py            # Face / RawMatch / ProviderResult / ProfileMatch / SearchReport
├── face.py              # detect (insightface→YuNet→Haar) + ArcFace 512-d embed
├── social_parser.py     # URL → (platform, is_profile) across all sources
├── dedupe.py            # canonical-URL dedupe + cross-provider merge + confidence
├── orchestrator.py      # fan-out, hosting, verification, ranking, cleanup
├── cli.py               # standalone CLI
└── providers/
    ├── base.py          # Provider contract (failure-as-data)
    ├── serpapi_sources.py  # Lens / Yandex / Bing / Google Images
    ├── tineye.py        # HMAC-signed TinEye REST API
    └── keyless.py       # DDG name search + Commons + thumbnail fetcher
```

Face backends (tried in order, degrading gracefully):
1. **insightface buffalo_l** — detection + 512-d embeddings; weights ship in
   `models/face/insightface/models/buffalo_l` (already downloaded by
   `services/intelligence/model_downloader.py`), CPU ONNX runtime.
2. **OpenCV YuNet** — detection only, embedding via the raw w600k_r50 ONNX.
3. **Haar cascade** — detection only, no embeddings (verification disabled).

## Confidence & ranking

Each merged result gets a 0..1 confidence:
- base 0.35
- +0.15 per additional provider that saw the same URL (corroboration, capped +0.3)
- +0.2 when the URL is a person-profile
- provider-native score (0..1) can raise the floor

Face verification adds a hard signal: `verified=True` (cosine ≥ 0.45) floats
to the top; `verified=False` sinks. `verified=None` means it wasn't attempted
(no thumbnail, no embedding, or `--no-verify`).

## Privacy & ethics

- Query images and face crops live in temp files only and are deleted at the
  end of every run (including on failure paths, via `finally`).
- No face database is built; nothing persists between runs.
- Only publicly reachable content is queried — the same pages any search
  engine indexes.
- Results are *leads*, not verdicts: confidence is a ranking prior, and only
  the local face-verification step makes a same-person judgement, on the
  user's own query image.

Companion-brain integration: the capability is registered as `face-search`
in `companion/capabilities.py`, so `POST /api/brain/compute` with a task like
"find this person's social profiles from a photo" auto-routes here.

Tests: `sweep_neural_mesh/tests/test_face_search.py` (offline, no network).
