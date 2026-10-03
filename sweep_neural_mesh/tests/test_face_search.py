"""Unit tests for the face/reverse-image search engine.

All tests run offline — no network access is required. Network-touching code
paths (provider search, image fetch) are exercised with fake transports or
stubbed at the client boundary.
"""
from __future__ import annotations

import pytest

from sweep_neural_mesh.face_search import social_parser
from sweep_neural_mesh.face_search.dedupe import _canonicalize, merge
from sweep_neural_mesh.face_search.models import ProviderResult, RawMatch


# ── social parser ─────────────────────────────────────────────────────


class TestClassify:
    @pytest.mark.parametrize(
        ("url", "platform", "is_profile"),
        [
            ("https://www.linkedin.com/in/jane-dough/", "linkedin", True),
            ("https://linkedin.com/in/janedough", "linkedin", True),
            ("https://www.linkedin.com/company/acme-corp", "linkedin", False),
            ("https://instagram.com/emilyrose_x/", "instagram", True),
            ("https://instagram.com/p/CxYz123/", "instagram", False),
            ("https://instagram.com/explore/tags/cats/", "instagram", False),
            ("https://facebook.com/emma.walters", "facebook", True),
            ("https://facebook.com/watch/?v=123", "facebook", False),
            ("https://x.com/janedough", "x", True),
            ("https://twitter.com/janedough/status/123", "x", False),
            ("https://tiktok.com/@creator", "tiktok", True),
            ("https://youtube.com/@channelname", "youtube", True),
            ("https://youtube.com/watch?v=abc", "youtube", False),
            ("https://reddit.com/user/somebody", "reddit", True),
            ("https://github.com/torvalds", "github", True),
            ("https://medium.com/@writer", "medium", True),
            ("https://mastodon.social/@someone", "mastodon", True),
            ("https://bsky.app/profile/someone.bsky.social", "bluesky", True),
            ("https://about.me/janedough", "aboutme", True),
            ("https://en.wikipedia.org/wiki/Face", "web", False),
            ("https://news.ycombinator.com/item?id=1", "web", False),
            ("not a url at all", "web", False),
            ("", "web", False),
        ],
    )
    def test_classification(self, url, platform, is_profile):
        got_platform, got_profile = social_parser.classify(url)
        assert got_platform == platform
        assert got_profile is is_profile


# ── dedupe ────────────────────────────────────────────────────────────


class TestCanonicalize:
    def test_strips_www_and_trailing_slash(self):
        assert (
            _canonicalize("https://WWW.Example.com/path/")
            == "https://example.com/path"
        )

    def test_strips_tracking_params(self):
        url = "https://instagram.com/em?utm_source=x&fbclid=1&igshid=2"
        assert _canonicalize(url) == "https://instagram.com/em"

    def test_keeps_meaningful_query(self):
        url = "https://example.com/page?id=7&utm_source=x"
        assert _canonicalize(url) == "https://example.com/page?id=7"

    def test_http_upgrades_to_https(self):
        assert _canonicalize("http://example.com/a") == "https://example.com/a"


class TestMerge:
    def test_cross_provider_merge_and_sorting(self):
        r1 = ProviderResult(
            provider="google_lens",
            matches=[RawMatch(url="https://instagram.com/em", title="Em", source="google_lens")],
        )
        r2 = ProviderResult(
            provider="yandex_images",
            matches=[RawMatch(url="https://www.instagram.com/em/", source="yandex_images")],
        )
        r3 = ProviderResult(
            provider="google_lens",
            matches=[RawMatch(url="https://example.com/article", source="google_lens")],
        )
        merged = merge([r1, r2, r3])
        assert len(merged) == 2
        top = merged[0]
        assert top.url == "https://instagram.com/em"
        assert sorted(top.sources) == ["google_lens", "yandex_images"]
        assert top.platform == "instagram"
        assert top.is_profile is True
        assert top.confidence > 0.5
        # article is not a profile → sorted after
        assert merged[1].url == "https://example.com/article"

    def test_title_and_thumbnail_filled_from_later_sources(self):
        r1 = ProviderResult(provider="a", matches=[RawMatch(url="https://x.com/j")])
        r2 = ProviderResult(
            provider="b",
            matches=[RawMatch(url="https://x.com/j", title="J", thumbnail_url="t.jpg")],
        )
        merged = merge([r1, r2])
        assert len(merged) == 1
        assert merged[0].title == "J"
        assert merged[0].thumbnail_url == "t.jpg"


# ── provider base parsing ─────────────────────────────────────────────


class TestProviderResultParsing:
    def test_serpapi_result_parsing(self):
        from sweep_neural_mesh.face_search.providers.serpapi_sources import GoogleLensProvider

        p = GoogleLensProvider()
        res = p._result(
            "google_lens",
            [
                {"link": "https://instagram.com/em", "title": "Em", "thumbnail": "t.jpg"},
                {"no_link": True},
                {"url": "https://example.com/2", "title": "Two"},
            ],
            max_results=10,
        )
        assert res.ok
        assert len(res.matches) == 2
        assert res.matches[0].url == "https://instagram.com/em"
        assert res.matches[0].thumbnail_url == "t.jpg"
        assert res.matches[1].url == "https://example.com/2"

    def test_tineye_signing_is_deterministic(self):
        from sweep_neural_mesh.face_search.providers.tineye import (
            _hmac_sha256,
            _sign_get,
        )

        sig1 = _sign_get(private_key="k", api_key="a", date=1000, nonce="n")
        sig2 = _sign_get(private_key="k", api_key="a", date=1000, nonce="n")
        assert sig1 == sig2
        assert len(sig1) == 64
        assert _hmac_sha256("k", "m") == _hmac_sha256("k", "m")

    def test_enablement_flags(self):
        from sweep_neural_mesh.face_search.providers.serpapi_sources import (
            BingReverseImageProvider,
            GoogleLensProvider,
        )
        from sweep_neural_mesh.face_search.providers.tineye import TinEyeProvider

        # without env keys nothing is enabled (tests must not require keys)
        assert GoogleLensProvider().is_enabled() == bool(__import__("os").environ.get("SERPAPI_KEY"))
        assert BingReverseImageProvider().is_enabled() == bool(__import__("os").environ.get("SERPAPI_KEY"))
        tineye = TinEyeProvider()
        assert tineye.is_enabled() == bool(
            __import__("os").environ.get("TINEYE_API_KEY")
            and __import__("os").environ.get("TINEYE_PRIVATE_KEY")
        )


# ── keyless providers ─────────────────────────────────────────────────


class TestKeyless:
    def test_name_hint_provider_disabled_without_hint(self):
        from sweep_neural_mesh.face_search.providers.keyless import NameHintSearchProvider

        p = NameHintSearchProvider()
        assert p.is_enabled() is False
        assert p.note() is not None

    def test_name_hint_provider_enabled_with_hint(self):
        from sweep_neural_mesh.face_search.providers.keyless import NameHintSearchProvider

        p = NameHintSearchProvider(name_hint="jane dough")
        assert p.is_enabled() is True

    def test_ddg_parser_handles_wrapped_links(self):
        from sweep_neural_mesh.face_search.providers.keyless import _ddg_results

        class FakeResp:
            def raise_for_status(self):
                return None

            text = (
                '<a rel="nofollow" class="result__a" '
                'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Flinkedin.com%2Fin%2Fj&amp;rut=x">Jane</a>'
            )

        class FakeClient:
            def post(self, *a, **kw):
                return FakeResp()

        out = _ddg_results(FakeClient(), "q", 5)
        assert out == [{"link": "https://linkedin.com/in/j", "title": "Jane"}]


# ── face pipeline (no model loads) ────────────────────────────────────


class TestFace:
    def test_cosine(self):
        from sweep_neural_mesh.face_search.face import cosine

        assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
        assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
        assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
        assert cosine([], [1.0]) is None
        assert cosine(None, None) is None
        assert cosine([1.0], [1.0, 2.0]) is None

    def test_availability_report(self):
        from sweep_neural_mesh.face_search.face import availability

        rep = availability()
        assert "insightface" in rep and "opencv" in rep
        assert isinstance(rep["opencv"]["installed"], bool)


# ── orchestrator (offline paths) ──────────────────────────────────────


class TestOrchestrator:
    def test_missing_image_is_reported_not_raised(self, tmp_path):
        from sweep_neural_mesh.face_search.orchestrator import FaceSearchOrchestrator

        report = FaceSearchOrchestrator().run_search(str(tmp_path / "missing.jpg"))
        assert report.results == []
        assert any("could not read" in n for n in report.notes)

    def test_fleet_builds_from_registry(self):
        from sweep_neural_mesh.face_search.orchestrator import build_providers

        fleet = build_providers(name_hint="jane")
        names = [p.name for p in fleet]
        # keyless upload engines are ALWAYS present (zero-configuration)
        for expected in (
            "bing_upload", "yandex_upload", "google_lens_upload", "tineye_upload",
            "name_search", "commons_search",
        ):
            assert expected in names
        # keyed accelerators appear only when their env keys are set
        if not __import__("os").environ.get("SERPAPI_KEY"):
            assert "google_lens" not in names
            assert "yandex_images" not in names
        if not (
            __import__("os").environ.get("TINEYE_API_KEY")
            and __import__("os").environ.get("TINEYE_PRIVATE_KEY")
        ):
            assert "tineye" not in names

    def test_keyless_engines_are_keyless(self):
        from sweep_neural_mesh.face_search.providers.keyless_engines import (
            KEYLESS_ENGINE_PROVIDERS,
        )

        for cls in KEYLESS_ENGINE_PROVIDERS:
            p = cls()
            assert p.is_enabled() is True, f"{p.name} must be keyless-enabled"
            assert p.accepts_upload is True

    def test_render_text(self):
        from sweep_neural_mesh.face_search.models import ProfileMatch, SearchReport
        from sweep_neural_mesh.face_search.orchestrator import FaceSearchOrchestrator

        report = SearchReport(query_image="x.jpg", mode="face")
        report.results = [
            ProfileMatch(
                url="https://linkedin.com/in/j",
                domain="linkedin.com",
                platform="linkedin",
                is_profile=True,
                confidence=0.7,
                sources=["a", "b"],
            )
        ]
        text = FaceSearchOrchestrator.render_text(report)
        assert "linkedin.com/in/j" in text
        assert "profiles (1)" in text

    def test_fan_out_records_provider_failures_as_data(self, monkeypatch):
        from sweep_neural_mesh.face_search import orchestrator
        from sweep_neural_mesh.face_search.models import ProviderResult
        from sweep_neural_mesh.face_search.providers.base import Provider

        class Boom(Provider):
            name = "boom"

            def search(self, client, **kw):
                raise RuntimeError("kaboom")

        class Ok(Provider):
            name = "ok"
            accepts_url = True

            def search(self, client, *, image_url, **kw):
                return ProviderResult(
                    provider=self.name,
                    matches=[RawMatch(url="https://example.com/a", source=self.name)],
                )

        orch = orchestrator.FaceSearchOrchestrator()
        results = orch._fan_out(
            [Boom(), Ok()],
            image_url="https://host/crop.jpg",
            image_bytes=None,
            image_filename=None,
            max_results=10,
        )
        by_name = {r.provider: r for r in results}
        assert by_name["boom"].error and "kaboom" in by_name["boom"].error
        assert by_name["ok"].ok and by_name["ok"].matches

    def test_fan_out_client_is_live_while_providers_run(self):
        """Regression: the httpx.Client context used to close before futures
        ran. Fake providers now prove the client is usable mid-flight."""
        import httpx as _httpx

        from sweep_neural_mesh.face_search import orchestrator
        from sweep_neural_mesh.face_search.models import ProviderResult
        from sweep_neural_mesh.face_search.providers.base import Provider

        class UsesClient(Provider):
            name = "uses_client"
            accepts_url = True

            def search(self, client, *, image_url, **kw):
                # a closed client raises RuntimeError('Pool after close')
                assert not client.is_closed, "httpx client closed before providers finished"
                return ProviderResult(provider=self.name, matches=[])

        orch = orchestrator.FaceSearchOrchestrator()
        results = orch._fan_out(
            [UsesClient()],
            image_url="https://host/crop.jpg",
            image_bytes=None,
            image_filename=None,
            max_results=5,
        )
        assert results[0].ok, results[0].error


# ── keyless upload engines (offline parser + flow tests) ─────────────


class TestKeylessEngines:
    def test_bing_iusc_parsing(self):
        from sweep_neural_mesh.face_search.providers.keyless_engines import (
            BingUploadProvider,
        )

        html = (
            '<a class="iusc" m="{&quot;murl&quot;:&quot;https://pics.example/a.jpg&quot;,'
            '&quot;purl&quot;:&quot;https://instagram.com/em&quot;,'
            '&quot;turl&quot;:&quot;https://tse.mm.bing.net/th?id=1&quot;,'
            '&quot;t&quot;:&quot;Em on Insta&quot;}"></a>'
            '<a class="iusc" m="{&quot;purl&quot;:&quot;&quot;}"></a>'  # no purl → skipped
        )
        out = BingUploadProvider()._parse(html, max_results=10)
        assert len(out) == 1
        assert out[0]["link"] == "https://instagram.com/em"
        assert out[0]["title"] == "Em on Insta"
        assert out[0]["thumbnail"].startswith("https://tse")

    def test_yandex_cbir_sites_parsing(self):
        from sweep_neural_mesh.face_search.providers.keyless_engines import (
            YandexUploadProvider,
        )

        html = (
            '{"CbirSites":{"Domains":["example.com"],'
            '"Sites":[{"Url":"https://example.com/page1","Title":"Page 1",'
            '"Thumb":"https://avatars.mds.yandex.net/t.jpg"},'
            '{"Url":"https://example.com/page1"}]}}'
        )
        out = YandexUploadProvider()._parse(html, max_results=10)
        assert len(out) == 1  # duplicate URL dropped
        assert out[0]["link"] == "https://example.com/page1"
        assert out[0]["title"] == "Page 1"

    def test_yandex_results_url_extraction(self):
        from sweep_neural_mesh.face_search.providers.keyless_engines import (
            YandexUploadProvider,
        )

        body = 'ok https://yandex.com/images/search?rpt=imageview&url=abc&amp;lr=1 end'
        url = YandexUploadProvider._extract_results_url(body)
        assert url and url.startswith("https://yandex.com/images/search")
        assert "&amp;" not in url

    def test_lens_page_parsing_skips_google_hosts(self):
        from sweep_neural_mesh.face_search.providers.keyless_engines import (
            GoogleLensUploadProvider,
        )

        html = (
            '["Em profile","https://instagram.com/em/x",null]'
            '["g static","https://lh3.googleusercontent.com/a.jpg",null]'
            '["wiki","https://en.wikipedia.org/wiki/Em",null]'
        )
        out = GoogleLensUploadProvider()._parse(html, max_results=10)
        links = [o["link"] for o in out]
        assert "https://instagram.com/em/x" in links
        assert all("googleusercontent" not in l for l in links)

    def test_tineye_match_parsing(self):
        from sweep_neural_mesh.face_search.providers.keyless_engines import (
            TinEyeUploadProvider,
        )

        html = (
            '<a class="match" href="https://example.com/a">Title A</a>'
            '<a class="match" href="https://example.com/a">dup</a>'
        )
        out = TinEyeUploadProvider()._parse(html, max_results=10)
        assert [o["link"] for o in out] == ["https://example.com/a"]
        assert out[0]["title"] == "Title A"

    def test_engines_report_missing_bytes_as_data(self):
        from sweep_neural_mesh.face_search.providers.keyless_engines import (
            BingUploadProvider,
        )

        res = BingUploadProvider().search(None, image_url=None,
                                          image_bytes=None, image_filename=None,
                                          max_results=10)
        assert res.error and "bytes" in res.error

    def test_challenge_detection(self):
        from sweep_neural_mesh.face_search.providers.keyless_engines import (
            _looks_like_challenge,
        )

        assert _looks_like_challenge(403, "")
        assert _looks_like_challenge(200, "unusual traffic from your computer")
        assert _looks_like_challenge(200, "<html>Just a moment...</html>")
        assert not _looks_like_challenge(200, "<html>normal page</html>")


# ── HTTP layer (offline) ─────────────────────────────────────────────


class TestHttpLayer:
    def test_backends_build(self):
        from sweep_neural_mesh.face_search.providers.http import HttpLike

        s = HttpLike(timeout=10)
        s._ensure()
        assert s.backend in ("curl", "httpx")
        s.close()

    def test_response_challenge_markers(self):
        from sweep_neural_mesh.face_search.providers.http import Response

        r = Response(status_code=403, text="", content=b"")
        assert r.looks_like_challenge()
        r2 = Response(status_code=200, text="verify you are human", content=b"")
        assert r2.looks_like_challenge()
        r3 = Response(status_code=200, text="hello", content=b"")
        assert not r3.looks_like_challenge()
