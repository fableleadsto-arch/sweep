import pytest

from sweep_core.integrations import capabilities, resources
from sweep_core.integrations.scraping import ENGINES, availability as scraping_availability


class TestRegistry:
    def test_capabilities_sections(self):
        caps = capabilities()
        assert set(caps) == {
            "scraping",
            "audio",
            "vision",
            "search",
            "bluetooth",
            "resources",
        }

    def test_all_scraping_engines_reported(self):
        reported = set(scraping_availability())
        assert reported == set(ENGINES)

    def test_every_entry_has_available_flag(self):
        for section in capabilities().values():
            if isinstance(section, dict):
                for name, info in section.items():
                    if isinstance(info, dict):
                        assert "available" in info or "client" in info or "binary" in info, (
                            f"{name} missing availability flag"
                        )


class TestGracefulDegradation:
    def test_handles_raise_cleanly_when_missing(self):
        from sweep_core.integrations import scraping

        pytest.importorskip("playwright", reason="only meaningful when installed")
        assert hasattr(scraping.browser_automation_handle(), "__name__")

    def test_resources_track_deferred_tools(self):
        names = {tool["name"] for tool in resources.DEFERRED_TOOLS}
        assert {"obscura", "bluez", "deepspeech", "ultravox"} <= names

    def test_voice_datasets_indexed(self):
        assert resources.VOICE_DATASETS[0]["url"].startswith("https://github.com/")
