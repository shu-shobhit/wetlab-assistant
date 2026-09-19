import dataclasses

import jwt
import pytest
from aiohttp.test_utils import TestClient, TestServer

from wetlab import settings, webserver


@pytest.fixture
def s(monkeypatch):
    monkeypatch.setenv("RIME_API_KEY", "r")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    monkeypatch.delenv("LIVEKIT_URL", raising=False)
    return settings.load()


async def test_token_carries_room_join_grant(s, tmp_path):
    (tmp_path / "index.html").write_text("<html>ok</html>")
    app = webserver.make_app(s, tmp_path)
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/token", params={"room": "bench", "identity": "scientist"})
        assert resp.status == 200
        body = await resp.json()

        claims = jwt.decode(
            body["token"], "secret", algorithms=["HS256"], options={"verify_aud": False}
        )
        assert claims["video"]["roomJoin"] is True
        assert claims["video"]["room"] == "bench"
        assert claims["sub"] == "scientist"
        assert body["url"] == "ws://127.0.0.1:7880"


async def test_serves_the_bench_page(s, tmp_path):
    (tmp_path / "index.html").write_text("<html>ok</html>")
    app = webserver.make_app(s, tmp_path)
    async with TestClient(TestServer(app)) as client:
        page = await client.get("/")
        assert page.status == 200
        assert await page.text() == "<html>ok</html>"


async def test_protocols_lists_what_the_picker_can_offer(s, tmp_path, monkeypatch):
    """The picker is built from this, and the agent checks the id that comes
    back against the same corpus rather than trusting the browser."""
    monkeypatch.setenv("PROTOCOL_ID", "neb_q5_m0492")
    (tmp_path / "index.html").write_text("<html>ok</html>")
    app = webserver.make_app(s, tmp_path)
    async with TestClient(TestServer(app)) as client:
        body = await (await client.get("/protocols")).json()

    assert body["default"] == "neb_q5_m0492"
    by_id = {p["id"]: p for p in body["protocols"]}
    assert "neb_q5_m0492" in by_id
    neb = by_id["neb_q5_m0492"]
    assert neb["ready"] is True
    assert neb["steps"] == 17
    assert neb["title"] and neb["citation"]


async def test_protocols_is_read_off_disk_on_every_request(s, tmp_path):
    """A protocol ingested while the server is up shows up on the next reload,
    which is what happens during a build."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (tmp_path / "index.html").write_text("<html>ok</html>")
    app = webserver.make_app(dataclasses.replace(s, corpus_dir=corpus), tmp_path)
    async with TestClient(TestServer(app)) as client:
        first = await (await client.get("/protocols")).json()
        assert first["protocols"] == []

        added = corpus / "later"
        added.mkdir()
        (added / "parsed.yaml").write_text("id: later\ntitle: t\nsteps:\n  - id: s1\n    text: x\n")
        (added / "CITATION").write_text("test")
        second = await (await client.get("/protocols")).json()

    assert [p["id"] for p in second["protocols"]] == ["later"]
    assert second["protocols"][0]["ready"] is False


async def test_token_defaults_to_the_configured_room(s, tmp_path):
    (tmp_path / "index.html").write_text("<html>ok</html>")
    app = webserver.make_app(s, tmp_path)
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/token")
        body = await resp.json()
        claims = jwt.decode(
            body["token"], "secret", algorithms=["HS256"], options={"verify_aud": False}
        )
        assert claims["video"]["room"] == "bench"
        assert claims["sub"] == "scientist"
