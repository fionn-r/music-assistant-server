"""Tests that the YouTube provider holds the stream back until YouTube releases it."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from music_assistant_models.enums import MediaType

from music_assistant.providers.youtube import YoutubeProvider

STREAM_URL = "https://rr1---sn-test.googlevideo.com/videoplayback?expire=9999999999&itag=140"


@pytest.fixture
def provider() -> YoutubeProvider:
    """Return a YoutubeProvider instance with mocked dependencies."""
    manifest = MagicMock()
    manifest.domain = "youtube"
    config = MagicMock()
    config.get_value.return_value = "GLOBAL"
    return YoutubeProvider(AsyncMock(), manifest, config)


def _stream_format(**extra: Any) -> dict[str, Any]:
    """Build the yt-dlp format the provider picks for a video."""
    return {
        "format": "140 - audio only (medium)",
        "url": STREAM_URL,
        "audio_ext": "m4a",
        "audio_channels": 2,
        "asr": 44100,
        **extra,
    }


async def test_waits_until_a_pre_roll_ad_could_have_been_skipped(
    provider: YoutubeProvider,
) -> None:
    """A stream url YouTube only releases after an ad must not be handed out before then."""
    provider._get_stream_format = AsyncMock(  # type: ignore[method-assign]
        return_value=_stream_format(available_at=int(time.time()) + 5)
    )
    with patch("music_assistant.providers.youtube.asyncio.sleep") as sleep:
        streamdetails = await provider.get_stream_details("vid", MediaType.TRACK)
    sleep.assert_awaited_once()
    assert sleep.await_args is not None
    assert 3 < sleep.await_args.args[0] <= 5
    assert streamdetails.path == STREAM_URL


@pytest.mark.parametrize("available_at", [None, 0, int(time.time()) - 10])
async def test_a_released_stream_is_returned_right_away(
    provider: YoutubeProvider, available_at: int | None
) -> None:
    """Without an ad, or once it has passed, the stream details are returned without waiting."""
    provider._get_stream_format = AsyncMock(  # type: ignore[method-assign]
        return_value=_stream_format(available_at=available_at)
    )
    with patch("music_assistant.providers.youtube.asyncio.sleep") as sleep:
        await provider.get_stream_details("vid", MediaType.TRACK)
    sleep.assert_not_awaited()
