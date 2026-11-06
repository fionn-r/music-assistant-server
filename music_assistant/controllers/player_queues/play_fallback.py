"""
Go back to what was playing when a "play now" request cannot start any of its own items.

Local patch on top of upstream. A PLAY request inserts its items after the playing one and starts
the first of them, and play_index moves on to the next item whenever one fails to load. A request
whose items all fail would then carry on into the items that were queued after them, so it takes
over the queue while nothing it asked for plays. With this mixin, playback only moves on within the
request's own items; when none of them starts, the queue gets its items back and the item that was
playing resumes where it was.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from music_assistant_models.enums import PlaybackState
from music_assistant_models.errors import MediaNotFoundError

from music_assistant.controllers.player_queues.base import _PlayerQueuesBase

if TYPE_CHECKING:
    from music_assistant_models.errors import AudioError
    from music_assistant_models.queue_item import QueueItem


@dataclass(frozen=True, slots=True)
class PlayFallback:
    """The queue as a play request found it, to go back to when none of its items start."""

    items: tuple[QueueItem, ...]
    index_in_buffer: int | None
    # the items the request inserted, the only ones playback may move on to
    request_item_ids: frozenset[str]
    # the item that was playing (or paused) when the request came in
    resume_item_id: str | None


class _PlayRequestExhaustedError(Exception):
    """None of a play request's own items could be started."""

    def __init__(self, item_name: str, cause: MediaNotFoundError | AudioError | None) -> None:
        """
        Initialize the error.

        :param item_name: Name of the last item of the request that failed to start.
        :param cause: Why it failed, None when it was already marked unavailable.
        """
        super().__init__(item_name)
        self.item_name = item_name
        self.cause = cause


class PlayFallbackMixin(_PlayerQueuesBase):
    """Go back to what was playing when a play request cannot start any of its own items."""

    if TYPE_CHECKING:
        # implemented on PlayerQueuesController
        def update_items(self, queue_id: str, queue_items: list[QueueItem]) -> None: ...  # noqa: D102
        async def resume(self, queue_id: str, fade_in: bool | None = None) -> None: ...  # noqa: D102

    def _capture_play_fallback(self, queue_id: str, request_items: list[QueueItem]) -> PlayFallback:
        """
        Capture the queue a play request is about to change, before its items are loaded.

        :param queue_id: The queue the request plays on.
        :param request_items: The items the request is about to insert.
        """
        queue = self._queue_data[queue_id].queue
        playing = (
            queue.current_item
            if queue.state in (PlaybackState.PLAYING, PlaybackState.PAUSED)
            else None
        )
        return PlayFallback(
            items=tuple(self._queue_data[queue_id].items),
            index_in_buffer=queue.index_in_buffer,
            request_item_ids=frozenset(item.queue_item_id for item in request_items),
            resume_item_id=playing.queue_item_id if playing else None,
        )

    async def _play_index_with_fallback(
        self, queue_id: str, index: int, fallback: PlayFallback | None
    ) -> None:
        """
        Start playback at the given index, going back to the captured queue when nothing starts.

        :param queue_id: The queue to start playback on.
        :param index: The index of the request's first item.
        :param fallback: What to go back to; None plays the index without a fallback.
        :raises MediaNotFoundError: When none of the request's items could be started.
        """
        if fallback is None:
            await self.play_index(queue_id, index)
            return
        queue_data = self._queue_data[queue_id]
        exhausted: _PlayRequestExhaustedError | None = None
        queue_data.play_fallback = fallback
        try:
            await self.play_index(queue_id, index)
        except _PlayRequestExhaustedError as err:
            exhausted = err
        finally:
            # disarmed before going back, so the item that resumes may skip on as usual
            queue_data.play_fallback = None
        if exhausted is None:
            return
        self.logger.warning(
            "Could not play %s, going back to what was queued before: %s",
            exhausted.item_name,
            exhausted.cause or "marked unavailable",
        )
        queue_data.queue.index_in_buffer = fallback.index_in_buffer
        self.update_items(queue_id, list(fallback.items))
        if fallback.resume_item_id is not None:
            await self.resume(queue_id)
        else:
            await self._handle_stop(queue_id)
        # the cause stays in the log: an ffmpeg error carries the provider's stream url
        raise MediaNotFoundError(f"Playback failed for {exhausted.item_name}")

    def _raise_if_play_request_exhausted(
        self,
        queue_id: str,
        failed_item: QueueItem,
        next_index: int | None,
        cause: MediaNotFoundError | AudioError | None,
        out_of_attempts: bool,
    ) -> None:
        """
        Keep a play request from moving on to an item it did not ask for.

        Called by play_index after an item failed to start; does nothing outside a play request.

        :param queue_id: The queue playback is being started on.
        :param failed_item: The item that could not be started.
        :param next_index: The index play_index would try next, None when there is none.
        :param cause: Why the item could not be started, None when it was marked unavailable.
        :param out_of_attempts: Whether play_index used up its load attempts.
        """
        if (fallback := self._queue_data[queue_id].play_fallback) is None:
            return
        next_item = self.get_item(queue_id, next_index)
        if (
            out_of_attempts
            or next_item is None
            or next_item.queue_item_id not in fallback.request_item_ids
        ):
            raise _PlayRequestExhaustedError(failed_item.name, cause)
