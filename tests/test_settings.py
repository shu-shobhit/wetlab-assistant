import pytest

from wetlab import settings


def test_load_reads_env_and_defaults(monkeypatch):
    monkeypatch.setenv("RIME_API_KEY", "r")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    monkeypatch.delenv("LIVEKIT_URL", raising=False)
    s = settings.load()
    assert s.rime_api_key == "r"
    assert s.livekit_url == "ws://127.0.0.1:7880"
    assert s.livekit_api_key == "devkey" and s.livekit_api_secret == "secret"
    # The fastest of the three models measured on the same spoken walkthrough,
    # and the one the timer announcement works on. The provider is pinned
    # because which host serves it moves the first token by about a second.
    assert s.llm_model == "deepseek/deepseek-v4-flash"
    assert s.llm_provider == "baidu/fp8,alibaba/fp8"
    # A gateway model name, not an OpenRouter one. The two namespaces look
    # alike and naming the wrong one fails at connect time, not at load.
    assert s.stt_model == "deepgram/nova-3"
    assert s.probe_interval_ms == 200 and s.status_dwell_s == 0.4


def test_reasoning_is_unset_rather_than_off_by_default(monkeypatch):
    """Saying nothing and saying no are different requests. A model with its own
    default must not be sent a field it was never configured with."""
    monkeypatch.setenv("RIME_API_KEY", "r")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    monkeypatch.delenv("LLM_REASONING", raising=False)
    assert settings.load().llm_reasoning is None


def test_a_reasoning_level_is_read_and_lowercased(monkeypatch):
    monkeypatch.setenv("RIME_API_KEY", "r")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    for raw, expected in (("minimal", "minimal"), (" OFF ", "off"), ("High", "high")):
        monkeypatch.setenv("LLM_REASONING", raw)
        assert settings.load().llm_reasoning == expected


def test_a_reasoning_level_that_is_not_one_of_the_choices_is_refused(monkeypatch):
    """It reaches OpenRouter otherwise, and a rejected reasoning field fails the
    turn rather than the startup: the agent registers, greets, and falls over
    the first time anybody speaks to it."""
    monkeypatch.setenv("RIME_API_KEY", "r")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    monkeypatch.setenv("LLM_REASONING", "none")
    with pytest.raises(settings.SettingsError, match="minimal"):
        settings.load()


def test_missing_required_key_names_it(monkeypatch):
    # load() falls back to .env, which is right in production and would defeat
    # this test on any machine that has one. Isolate from the file.
    monkeypatch.setattr(settings, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("RIME_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    with pytest.raises(settings.SettingsError, match="RIME_API_KEY"):
        settings.load()
