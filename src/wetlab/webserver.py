"""Serves the bench page and mints room join tokens.

A separate process from the agent worker. The worker's process model is owned
by cli.run_app (forkserver on Linux), so an HTTP server inside it is fragile.

The API secret signs the token here and never reaches the browser.

    python -m wetlab.webserver
"""

from __future__ import annotations

import dataclasses
import datetime
from pathlib import Path

from aiohttp import web
from livekit import api

from . import protocol as protocol_mod
from . import settings as settings_mod

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
TOKEN_TTL = datetime.timedelta(hours=6)


def mint_token(s: settings_mod.Settings, room: str, identity: str) -> str:
    """Sign a join token for one participant in one room."""
    return (
        api.AccessToken(s.livekit_api_key, s.livekit_api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .with_ttl(TOKEN_TTL)
        .to_jwt()
    )


@web.middleware
async def _no_store(request: web.Request, handler):
    """Never let the browser cache the bench page.

    This is a page being edited while it runs. A cached room.js that predates
    the edit looks exactly like the edit not working, and the time lost to that
    is not recoverable during a two-day build.
    """
    response = await handler(request)
    response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response


def make_app(s: settings_mod.Settings, web_dir: Path) -> web.Application:
    app = web.Application(middlewares=[_no_store])

    async def token(request: web.Request) -> web.Response:
        room = request.query.get("room") or s.room_name
        identity = request.query.get("identity") or "scientist"
        return web.json_response({"token": mint_token(s, room, identity), "url": s.livekit_url})

    async def protocols(_request: web.Request) -> web.Response:
        """What the picker offers, read off disk on every request.

        Not cached and not baked into the page. A protocol ingested while the
        server is up shows up on the next reload, which is what happens during a
        build, and the corpus is three directories to stat.

        The page sends the chosen id back inside the room name. The agent checks
        it against this same listing rather than trusting it, so this endpoint
        decides what is offered and never what is loaded.
        """
        return web.json_response(
            {
                "default": s.protocol_id,
                "protocols": [
                    dataclasses.asdict(p) for p in protocol_mod.available(s.corpus_dir)
                ],
            }
        )

    async def index(_request: web.Request) -> web.FileResponse:
        return web.FileResponse(web_dir / "index.html")

    app.router.add_get("/token", token)
    app.router.add_get("/protocols", protocols)
    app.router.add_get("/", index)
    app.router.add_static("/static", web_dir, show_index=False)
    return app


def main() -> None:
    s = settings_mod.load()
    web.run_app(make_app(s, WEB_DIR), host="127.0.0.1", port=8080)


if __name__ == "__main__":
    main()
