"""
Tests that a "play now" request which cannot start goes back to what was playing.

These run the real ``_enqueue_with_option``, ``load``, ``update_items`` and ``play_index`` against a
bare controller; only loading an item, the player and the resume/stop commands are stubbed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from music_assistant_models.enums import PlaybackState, QueueOption
from music_assistant_models.errors import AudioError, MediaNotFoundError
from music_assistant_models.player_queue import PlayerQueue
from music_assistant_models.queue_item import QueueItem

from music_assistant.controllers.player_queues import PlayerQueuesController
from music_assistant.controllers.player_queues.controller import _MAX_LOAD_ATTEMPTS
from music_assistant.controllers.player_queues.state import PlayerQueueData

QUEUE_ID = "q1"
# what an ffmpeg failure looks like: its log carries the provider's own stream url
STREAM_ERROR = "FFMpeg exited with code 8: HTTP error 403 https://rr2.googlevideo.com/videoplayback"


def _items(names: list[str]) -> list[QueueItem]:
    """Build simple queue items with the given names as their ids."""
    return [
        QueueItem(queue_id=QUEUE_ID, queue_item_id=name, name=name, duration=60) for name in names
    ]


def _controller(
    state: PlaybackState, *, shuffle: bool = False
) -> tuple[PlayerQueuesController, PlayerQueueData, list[str]]:
    """
    Build a controller whose queue holds e0, e1 and e2 and sits on e1.

    Items whose id starts with "y" fail to load. Returns the controller, the queue's record and the
    ids of the items a load was attempted for, in order.

    :param state: The playback state of the queue.
    :param shuffle: Whether shuffle is enabled on the queue.
    """
    ctrl = PlayerQueuesController.__new__(PlayerQueuesController)
    existing = _items(["e0", "e1", "e2"])
    queue = PlayerQueue(
        queue_id=QUEUE_ID,
        active=True,
        display_name="Q1",
        available=True,
        items=len(existing),
        state=state,
        current_index=1,
        index_in_buffer=1,
        shuffle_enabled=shuffle,
    )
    queue.current_item = existing[1]
    queue_data = PlayerQueueData(queue=queue, items=list(existing))
    ctrl._queue_data = {QUEUE_ID: queue_data}
    loaded: list[str] = []

    async def _load(queue_item: QueueItem, *_args: object, **_kwargs: object) -> None:
        loaded.append(queue_item.queue_item_id)
        if queue_item.queue_item_id.startswith("y"):
            raise AudioError(STREAM_ERROR)

    def _next_index(_queue_id: str, index: int, allow_repeat: bool = True) -> int | None:  # noqa: ARG001
        return index + 1 if index + 1 < len(queue_data.items) else None

    ctrl._load_item = AsyncMock(side_effect=_load)  # type: ignore[method-assign]
    ctrl._get_next_index = Mock(side_effect=_next_index)  # type: ignore[method-assign]
    ctrl._smart_shuffle = MagicMock()
    ctrl._smart_shuffle.is_enabled.return_value = False
    ctrl.resume = AsyncMock()  # type: ignore[method-assign]
    ctrl.stop = AsyncMock()  # type: ignore[method-assign]
    ctrl._handle_stop = AsyncMock()  # type: ignore[method-assign]
    ctrl.signal_update = Mock()  # type: ignore[method-assign]
    ctrl.on_player_update = Mock()  # type: ignore[method-assign]
    ctrl.update_next_item_on_player = Mock()  # type: ignore[method-assign]
    ctrl.get_next_item = Mock(return_value=None)  # type: ignore[method-assign]
    ctrl._check_player_permission = Mock()  # type: ignore[method-assign]
    ctrl._set_transitioning = Mock()  # type: ignore[method-assign]
    ctrl.player_media_from_queue_item = AsyncMock()  # type: ignore[method-assign]
    ctrl.mass = MagicMock()
    ctrl.mass.players.play_media = AsyncMock()
    ctrl.logger = MagicMock()
    return ctrl, queue_data, loaded


def _ids(queue_data: PlayerQueueData) -> list[str]:
    """Return the ids of the queue's items, in order."""
    return [item.queue_item_id for item in queue_data.items]


async def test_a_play_that_fails_goes_back_to_the_playing_item() -> None:
    """The item after the playing one must not start; the playing one resumes instead."""
    ctrl, queue_data, loaded = _controller(PlaybackState.PLAYING)

    with pytest.raises(MediaNotFoundError, match="Playback failed for y") as exc_info:
        await ctrl._enqueue_with_option(QUEUE_ID, _items(["y"]), QueueOption.PLAY)

    assert loaded == ["y"]
    assert _ids(queue_data) == ["e0", "e1", "e2"]
    assert queue_data.queue.index_in_buffer == 1
    assert queue_data.play_fallback is None
    assert isinstance(ctrl.resume, AsyncMock)
    ctrl.resume.assert_awaited_once_with(QUEUE_ID)
    assert isinstance(ctrl.mass.players.play_media, AsyncMock)
    ctrl.mass.players.play_media.assert_not_awaited()
    # the caller learns which item failed, not the provider's stream url from ffmpeg's log
    assert "googlevideo" not in str(exc_info.value)


async def test_a_paused_item_is_also_gone_back_to() -> None:
    """A request made while the queue is paused goes back to the paused item as well."""
    ctrl, queue_data, loaded = _controller(PlaybackState.PAUSED)

    with pytest.raises(MediaNotFoundError):
        await ctrl._enqueue_with_option(QUEUE_ID, _items(["y"]), QueueOption.PLAY)

    assert loaded == ["y"]
    assert _ids(queue_data) == ["e0", "e1", "e2"]
    assert isinstance(ctrl.resume, AsyncMock)
    ctrl.resume.assert_awaited_once_with(QUEUE_ID)


async def test_a_play_moves_on_within_its_own_items() -> None:
    """When one of the requested items fails, the next requested item still plays."""
    ctrl, queue_data, loaded = _controller(PlaybackState.PLAYING)
    requested = _items(["y1", "n1"])

    await ctrl._enqueue_with_option(QUEUE_ID, requested, QueueOption.PLAY)

    assert loaded == ["y1", "n1"]
    assert queue_data.queue.current_item is requested[1]
    assert _ids(queue_data) == ["e0", "e1", "y1", "n1", "e2"]
    assert isinstance(ctrl.resume, AsyncMock)
    ctrl.resume.assert_not_awaited()
    assert queue_data.play_fallback is None


async def test_a_play_that_fails_on_an_idle_queue_stops() -> None:
    """With nothing playing there is nothing to resume, and the next item must not start either."""
    ctrl, queue_data, loaded = _controller(PlaybackState.IDLE)

    with pytest.raises(MediaNotFoundError, match="Playback failed for y"):
        await ctrl._enqueue_with_option(QUEUE_ID, _items(["y"]), QueueOption.PLAY)

    assert loaded == ["y"]
    assert _ids(queue_data) == ["e0", "e1", "e2"]
    assert isinstance(ctrl._handle_stop, AsyncMock)
    ctrl._handle_stop.assert_awaited_once_with(QUEUE_ID)
    assert isinstance(ctrl.resume, AsyncMock)
    ctrl.resume.assert_not_awaited()


async def test_a_play_goes_back_once_its_load_attempts_are_used_up() -> None:
    """A large request that keeps failing goes back after the usual number of attempts."""
    ctrl, queue_data, loaded = _controller(PlaybackState.PLAYING)
    requested = _items([f"y{i}" for i in range(_MAX_LOAD_ATTEMPTS + 2)])

    with pytest.raises(MediaNotFoundError, match=f"Playback failed for y{_MAX_LOAD_ATTEMPTS - 1}"):
        await ctrl._enqueue_with_option(QUEUE_ID, requested, QueueOption.PLAY)

    assert len(loaded) == _MAX_LOAD_ATTEMPTS
    assert _ids(queue_data) == ["e0", "e1", "e2"]
    assert isinstance(ctrl.resume, AsyncMock)
    ctrl.resume.assert_awaited_once_with(QUEUE_ID)


async def test_a_shuffled_play_keeps_skipping_into_the_queue() -> None:
    """A shuffled batch is mixed into the queued items, so there is no own set to keep to."""
    ctrl, queue_data, _loaded = _controller(PlaybackState.PLAYING, shuffle=True)

    await ctrl._enqueue_with_option(QUEUE_ID, _items(["y1", "y2"]), QueueOption.PLAY)

    # e2 is the only item after the playing one that loads, wherever the shuffle put it
    assert queue_data.queue.current_item is not None
    assert queue_data.queue.current_item.queue_item_id == "e2"
    assert isinstance(ctrl.resume, AsyncMock)
    ctrl.resume.assert_not_awaited()
