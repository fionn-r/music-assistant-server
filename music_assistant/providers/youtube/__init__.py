"""Plain YouTube (video audio) support for MusicAssistant."""

from __future__ import annotations

import asyncio
import importlib
import logging
import time
from http.cookies import SimpleCookie
from io import StringIO
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import parse_qs, urlparse

from music_assistant_models.config_entries import ConfigEntry
from music_assistant_models.enums import (
    ConfigEntryType,
    ContentType,
    ImageType,
    MediaType,
    ProviderFeature,
    StreamType,
)
from music_assistant_models.errors import (
    MediaNotFoundError,
    SetupFailedError,
    UnplayableMediaError,
)
from music_assistant_models.media_items import (
    Artist,
    AudioFormat,
    MediaItemImage,
    ProviderMapping,
    SearchResults,
    Track,
    UniqueList,
)
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.constants import CONF_ENTRY_UNOFFICIAL_PROVIDER, VERBOSE_LOG_LEVEL
from music_assistant.controllers.cache import use_cache
from music_assistant.helpers.util import import_module_in_thread, install_package
from music_assistant.models.music_provider import MusicProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant import MusicAssistant
    from music_assistant.models import ProviderInstanceType


CONF_COOKIE = "cookie"
CONF_PO_TOKEN_SERVER_URL = "po_token_server_url"

YT_DOMAIN = "https://www.youtube.com"
YT_COOKIE_DOMAIN = ".youtube.com"
WATCH_URL = f"{YT_DOMAIN}/watch?v={{0}}"
UNKNOWN_ARTIST_ID = "youtube_unknown_channel"
DEFAULT_STREAM_URL_EXPIRATION = 3600  # 1 hour
PACKAGE_TO_INSTALL = "yt-dlp[default]"
PO_TOKEN_PACKAGE_TO_INSTALL = "bgutil-ytdlp-pot-provider"

SUPPORTED_FEATURES = {ProviderFeature.SEARCH}


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return YoutubeProvider(mass, manifest, config, SUPPORTED_FEATURES)


class YoutubeProvider(MusicProvider):
    """Provider for plain YouTube videos, playing only their audio."""

    _yt_dlp_module = None
    _netscape_cookie: str | None = None
    _po_token_server_url: str | None = None

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Return Config entries to configure this provider."""
        return (
            CONF_ENTRY_UNOFFICIAL_PROVIDER,
            ConfigEntry(
                key=CONF_COOKIE,
                type=ConfigEntryType.SECURE_STRING,
                required=False,
                requires_reload=True,
            ),
            ConfigEntry(
                key=CONF_PO_TOKEN_SERVER_URL,
                type=ConfigEntryType.STRING,
                required=False,
                requires_reload=True,
            ),
        )

    async def handle_async_init(self) -> None:
        """Set up the YouTube provider."""
        logging.getLogger("yt_dlp").setLevel(self.logger.level + 10)
        self._po_token_server_url = self.get_config_value(CONF_PO_TOKEN_SERVER_URL, return_type=str)
        await self._install_packages()
        if raw_cookie := self.get_config_value(CONF_COOKIE, return_type=str):
            self._netscape_cookie = _convert_to_netscape(raw_cookie, YT_COOKIE_DOMAIN)

    @property
    def is_streaming_provider(self) -> bool:
        """Return True if the provider is a streaming provider."""
        return True

    @use_cache(3600 * 24)
    async def search(
        self, search_query: str, media_types: list[MediaType], limit: int = 5
    ) -> SearchResults:
        """
        Perform search on the provider.

        :param search_query: Search query.
        :param media_types: A list of media_types to include.
        :param limit: Number of items to return in the search (per type).
        """
        if MediaType.TRACK not in media_types:
            return SearchResults()
        try:
            info = await self._extract_info(f"ytsearch{limit}:{search_query}", flat=True)
        except MediaNotFoundError as err:
            self.logger.warning("Search failed for query '%s': %s", search_query, err)
            return SearchResults()
        tracks: list[Track] = []
        for entry in info.get("entries") or []:
            if not entry.get("id") or entry.get("live_status") == "is_live":
                continue
            tracks.append(self._parse_track(entry))
        return SearchResults(tracks=tracks)

    @use_cache(3600 * 24)
    async def get_track(self, prov_track_id: str) -> Track:
        """Get full track details by id."""
        info = await self._extract_info(WATCH_URL.format(prov_track_id))
        return self._parse_track(info)

    @use_cache(3600 * 24 * 30)
    async def get_artist(self, prov_artist_id: str) -> Artist:
        """Get full artist details by id."""
        if prov_artist_id != UNKNOWN_ARTIST_ID:
            channel_url = (
                f"{YT_DOMAIN}/channel/{prov_artist_id}"
                if prov_artist_id.startswith("UC")
                else f"{YT_DOMAIN}/{prov_artist_id}"
            )
            try:
                # playlist_items 0 gives us the channel details without any of its videos
                info = await self._extract_info(channel_url, flat=True, playlist_items="0")
            except MediaNotFoundError as err:
                self.logger.debug("Unable to fetch channel %s: %s", prov_artist_id, err)
            else:
                return self._parse_artist(info, fallback_id=prov_artist_id)
        return self._parse_artist({}, fallback_id=prov_artist_id)

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """Return the content details for the given track when it will be streamed."""
        stream_format = await self._get_stream_format(item_id)
        self.logger.debug("Found stream_format: %s for video %s", stream_format["format"], item_id)
        url: str = stream_format["url"]
        expiration = DEFAULT_STREAM_URL_EXPIRATION
        if parsed := parse_qs(urlparse(url).query):
            if expire_ts := parsed.get("expire", [None])[0]:
                expiration = int(expire_ts) - int(time.time())
        streamdetails = StreamDetails(
            provider=self.instance_id,
            item_id=item_id,
            audio_format=AudioFormat(
                content_type=ContentType.try_parse(stream_format["audio_ext"]),
            ),
            stream_type=StreamType.HTTP,
            path=url,
            can_seek=True,
            allow_seek=True,
            expiration=expiration,
        )
        if (audio_channels := stream_format.get("audio_channels")) and str(
            audio_channels
        ).isdigit():
            streamdetails.audio_format.channels = int(audio_channels)
        if asr := stream_format.get("asr"):
            streamdetails.audio_format.sample_rate = int(asr)
        return streamdetails

    async def _install_packages(self) -> None:
        """Install frequently changing packages dynamically."""
        # NOTE: Google breaks things quite often which requires us to update
        # yt-dlp very frequently. Installing it dynamically prevents us from
        # having to update MA to ensure this provider works. The requirements are
        # deliberately unpinned and installed with upgrade=True: without it an
        # already installed (outdated) yt-dlp is left alone and playback breaks as
        # soon as Google retires the player client that version falls back to.
        packages = [PACKAGE_TO_INSTALL]
        if self._po_token_server_url:
            packages.append(PO_TOKEN_PACKAGE_TO_INSTALL)
        for package in packages:
            try:
                await install_package(package, upgrade=True)
            except RuntimeError as err:
                # a failed upgrade (offline, PyPI hiccup) must not take down a provider
                # that still has a working version installed from an earlier run
                self.logger.warning("Failed to install/upgrade %s: %s", package, err)
        try:
            await import_module_in_thread("yt_dlp")
        except ImportError:
            raise SetupFailedError("Package yt_dlp failed to install")

    def _ydl_opts(self, *, flat: bool = False, **extra_opts: Any) -> dict[str, Any]:
        """Build the yt-dlp options, applying the optional cookie and PO token server."""
        extractor_args: dict[str, Any] = {"youtube": {"skip": ["translated_subs", "dash"]}}
        if self._po_token_server_url:
            extractor_args["youtubepot-bgutilhttp"] = {"base_url": [self._po_token_server_url]}
        ydl_opts: dict[str, Any] = {
            "quiet": self.logger.level > logging.DEBUG,
            "verbose": self.logger.level == VERBOSE_LOG_LEVEL,
            "extractor_args": extractor_args,
            **extra_opts,
        }
        if self._netscape_cookie:
            ydl_opts["cookiefile"] = StringIO(self._netscape_cookie)
        if flat:
            ydl_opts["extract_flat"] = "in_playlist"
        return ydl_opts

    async def _extract_info(
        self, url: str, *, flat: bool = False, **extra_opts: Any
    ) -> dict[str, Any]:
        """Retrieve the (metadata) info for the given URL with yt-dlp."""

        def _extract() -> dict[str, Any]:
            yt_dlp = self._get_yt_dlp_module()
            with yt_dlp.YoutubeDL(self._ydl_opts(flat=flat, **extra_opts)) as ydl:
                try:
                    info = ydl.extract_info(url, download=False)
                except yt_dlp.utils.DownloadError as err:
                    raise MediaNotFoundError(str(err)) from err
                if not info:
                    raise MediaNotFoundError(f"No info found for {url}")
                return cast("dict[str, Any]", info)

        return await asyncio.to_thread(_extract)

    async def _get_stream_format(self, item_id: str) -> dict[str, Any]:
        """Figure out the stream to use for the given video and return the best audio format."""

        def _extract_best_stream_url_format() -> dict[str, Any]:
            yt_dlp = self._get_yt_dlp_module()
            with yt_dlp.YoutubeDL(self._ydl_opts()) as ydl:
                try:
                    info = ydl.extract_info(WATCH_URL.format(item_id), download=False)
                except yt_dlp.utils.DownloadError as err:
                    raise UnplayableMediaError(str(err)) from err
                if not info:
                    raise UnplayableMediaError(f"No info found for video {item_id}")
                format_selector = ydl.build_format_selector("m4a/bestaudio")
                stream_format: dict[str, Any] | None = next(
                    format_selector({"formats": info["formats"]}), None
                )
                if not stream_format:
                    raise UnplayableMediaError("No stream formats found")
                return stream_format

        return await asyncio.to_thread(_extract_best_stream_url_format)

    def _get_yt_dlp_module(self) -> Any:
        """Return the (lazy imported) yt_dlp module."""
        if self._yt_dlp_module is None:
            self._yt_dlp_module = importlib.import_module("yt_dlp")
        return self._yt_dlp_module

    def _parse_track(self, track_obj: dict[str, Any]) -> Track:
        """Parse a yt-dlp video info dict into a MA Track."""
        track_id = str(track_obj["id"])
        track = Track(
            item_id=track_id,
            provider=self.instance_id,
            # video titles are freeform, so no attempt is made to split off a version
            name=track_obj.get("title") or track_id,
            duration=int(track_obj.get("duration") or 0),
            artists=UniqueList([self._parse_artist(track_obj)]),
            provider_mappings={
                ProviderMapping(
                    item_id=track_id,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                    url=WATCH_URL.format(track_id),
                    audio_format=AudioFormat(content_type=ContentType.M4A),
                )
            },
        )
        if description := track_obj.get("description"):
            track.metadata.description = description
        track.metadata.images = self._parse_thumbnails(track_obj.get("thumbnails") or [])
        return track

    def _parse_artist(self, obj: dict[str, Any], fallback_id: str | None = None) -> Artist:
        """Parse the channel/uploader of a yt-dlp info dict into a MA Artist."""
        artist_id = (
            obj.get("channel_id") or obj.get("uploader_id") or fallback_id or UNKNOWN_ARTIST_ID
        )
        artist = Artist(
            item_id=str(artist_id),
            provider=self.instance_id,
            name=obj.get("channel") or obj.get("uploader") or str(artist_id),
            provider_mappings={
                ProviderMapping(
                    item_id=str(artist_id),
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                    url=obj.get("channel_url") or obj.get("uploader_url"),
                )
            },
        )
        if thumbnails := obj.get("thumbnails"):
            artist.metadata.images = self._parse_thumbnails(thumbnails)
        return artist

    def _parse_thumbnails(self, thumbnails_obj: list[dict[str, Any]]) -> UniqueList[MediaItemImage]:
        """Parse yt-dlp thumbnails to MediaItemImage."""
        result: UniqueList[MediaItemImage] = UniqueList()
        processed_urls: set[str] = set()
        for img in sorted(thumbnails_obj, key=lambda w: w.get("width") or 0, reverse=True):
            if not (url := img.get("url")) or url in processed_urls:
                continue
            processed_urls.add(url)
            width = img.get("width") or 0
            height = img.get("height") or 0
            # video thumbs are 16:9, so they are registered as THUMB (the type the UI needs
            # to render lists) and the largest one additionally as LANDSCAPE. Channel avatars
            # are square and only ever become a THUMB.
            if not result and height and width / height > 1.5:
                result.append(
                    MediaItemImage(
                        type=ImageType.LANDSCAPE,
                        path=url,
                        provider=self.instance_id,
                        remotely_accessible=True,
                    )
                )
            result.append(
                MediaItemImage(
                    type=ImageType.THUMB,
                    path=url,
                    provider=self.instance_id,
                    remotely_accessible=True,
                )
            )
        return result


def _convert_to_netscape(raw_cookie_str: str, domain: str) -> str:
    """Convert a raw cookie into Netscape format, so yt-dlp can use it."""
    domain = domain.replace("https://", "")
    cookie = SimpleCookie()
    cookie.load(rawdata=raw_cookie_str)
    netscape_cookie = "# Netscape HTTP Cookie File\n"
    for morsel in cookie.values():
        netscape_cookie += f"{domain}\tTRUE\t/\tTRUE\t0\t{morsel.key}\t{morsel.value}\n"
    return netscape_cookie
