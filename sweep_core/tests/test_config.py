from sweep_core.config import Settings


class TestSettings:
    def test_defaults(self):
        s = Settings(_env_file=None)
        assert s.env == "dev"
        assert s.host == "127.0.0.1"
        assert s.port == 8095
        assert s.log_level == "INFO"

    def test_env_prefix_override(self, monkeypatch):
        monkeypatch.setenv("SWEEP_ENV", "prod")
        monkeypatch.setenv("SWEEP_PORT", "9000")
        s = Settings(_env_file=None)
        assert s.env == "prod"
        assert s.port == 9000

    def test_extra_env_vars_ignored(self, monkeypatch):
        monkeypatch.setenv("SWEEP_UNRELATED_FUTURE_FLAG", "x")
        s = Settings(_env_file=None)
        assert not hasattr(s, "unrelated_future_flag")
