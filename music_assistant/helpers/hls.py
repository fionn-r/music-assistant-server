"""
RFC 8216-based HLS utilities.

For simple variant stream selection from master playlists, use helpers.playlists.parse_m3u.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urljoin

from music_assistant_models.errors import InvalidDataError

from music_assistant.helpers.datetime import from_iso_string

_URI_ATTRIBUTE_RE = re.compile(r'URI="([^"]*)"')


@dataclass
class HLSMediaSegment:
    """Single HLS media segment entry with associated metadata."""

    extinf_line: str = ""
    segment_url: str = ""
    key_line: str | None = None
    byterange_line: str | None = None
    discontinuity: bool = False
    map_line: str | None = None
    program_date_time: str | None = None

    @property
    def duration(self) -> float:
        """Extract duration in seconds from #EXTINF line."""
        try:
            duration_part = self.extinf_line.split("#EXTINF:")[1].split(",", 1)[0]
            return float(duration_part.strip())
        except IndexError, ValueError:
            return 0.0

    @property
    def start_time(self) -> datetime | None:
        """Return the wall-clock time from the #EXT-X-PROGRAM-DATE-TIME tag, if present."""
        if not self.program_date_time:
            return None
        try:
            return from_iso_string(self.program_date_time.split(":", 1)[1].strip())
        except IndexError, ValueError:
            return None

    @property
    def title(self) -> str | None:
        """Extract optional title from #EXTINF line."""
        try:
            parts = self.extinf_line.split("#EXTINF:")[1].split(",", 1)
            if len(parts) == 2:
                title = parts[1].strip()
                return title or None
            return None
        except IndexError:
            return None


@dataclass
class HLSMediaPlaylist:
    """
    HLS media playlist structure with headers, segments, and footers preserved.

    Note: header_lines excludes EXT-X-KEY and EXT-X-MAP tags. Per RFC 8216, these
    tags apply to subsequent segments until overridden, so they're stored per-segment
    for easier manipulation.
    """

    header_lines: list[str] = field(default_factory=list)
    segments: list[HLSMediaSegment] = field(default_factory=list)
    footer_lines: list[str] = field(default_factory=list)

    @property
    def media_sequence(self) -> int:
        """Return the media sequence number of the first segment (0 when not given)."""
        try:
            return int(self._header_value("#EXT-X-MEDIA-SEQUENCE:") or 0)
        except ValueError:
            return 0

    @property
    def target_duration(self) -> float | None:
        """Return the maximum segment duration in seconds, if given."""
        try:
            value = self._header_value("#EXT-X-TARGETDURATION:")
            return float(value) if value else None
        except ValueError:
            return None

    @property
    def ended(self) -> bool:
        """Return whether no more segments will be added to the playlist."""
        return "#EXT-X-ENDLIST" in self.footer_lines

    def segment_at(self, position: float) -> tuple[int, float]:
        """
        Return the index of the segment that holds a position, and the offset into it.

        A position past the end of the playlist lands at the start of its last segment.

        :param position: Position in seconds from the start of the playlist.
        """
        segment_start = 0.0
        for index, segment in enumerate(self.segments):
            if position < segment_start + segment.duration:
                return index, max(0.0, position - segment_start)
            segment_start += segment.duration
        return max(0, len(self.segments) - 1), 0.0

    def render(self, start_index: int = 0, base_url: str | None = None) -> str:
        """
        Return the playlist as text, leaving out the segments before start_index.

        The media sequence number follows the first segment that is kept, as the
        default IV of an encrypted segment derives from it.

        :param start_index: Index of the first segment to include.
        :param base_url: URL the playlist was fetched from, to make relative URIs absolute.
        """

        def _absolute(uri: str) -> str:
            return urljoin(base_url, uri) if base_url else uri

        def _absolute_attribute(line: str) -> str:
            return _URI_ATTRIBUTE_RE.sub(lambda match: f'URI="{_absolute(match[1])}"', line)

        sequence_line = f"#EXT-X-MEDIA-SEQUENCE:{self.media_sequence + start_index}"
        lines = [
            sequence_line if line.startswith("#EXT-X-MEDIA-SEQUENCE:") else line
            for line in self.header_lines
        ]
        if start_index and sequence_line not in lines:
            lines.append(sequence_line)
        # EXT-X-MAP and EXT-X-KEY apply until overridden, so each is written where it changes
        prev_map_line: str | None = None
        prev_key_line: str | None = None
        for segment in self.segments[start_index:]:
            if segment.discontinuity:
                lines.append("#EXT-X-DISCONTINUITY")
            if segment.program_date_time:
                lines.append(segment.program_date_time)
            if segment.map_line and segment.map_line != prev_map_line:
                lines.append(_absolute_attribute(segment.map_line))
                prev_map_line = segment.map_line
            if segment.key_line and segment.key_line != prev_key_line:
                lines.append(_absolute_attribute(segment.key_line))
                prev_key_line = segment.key_line
            lines.append(segment.extinf_line)
            if segment.byterange_line:
                lines.append(segment.byterange_line)
            lines.append(_absolute(segment.segment_url))
        lines.extend(self.footer_lines)
        return "\n".join(lines) + "\n"

    def _header_value(self, tag: str) -> str | None:
        """Return the value of a header tag, if present."""
        for line in self.header_lines:
            if line.startswith(tag):
                return line.removeprefix(tag).strip()
        return None


class HLSMediaPlaylistParser:
    """RFC 8216-based HLS media playlist parser."""

    def __init__(self, hls_playlist_text: str) -> None:
        """Initialize parser with playlist text."""
        self.hls_playlist_text = hls_playlist_text
        self.result = HLSMediaPlaylist()
        self.working_segment = HLSMediaSegment()
        self.segments_started = False

    def parse(self) -> HLSMediaPlaylist:
        """
        Parse HLS media playlist text into structured data.

        Returns:
            HLSMediaPlaylist object with extracted structure

        Raises:
            InvalidDataError: If playlist doesn't start with #EXTM3U or has invalid format
        """
        lines = [line.strip() for line in self.hls_playlist_text.split("\n") if line.strip()]

        if not lines or not lines[0].startswith("#EXTM3U"):
            msg = "Invalid HLS playlist: must start with #EXTM3U"
            raise InvalidDataError(msg)

        for line in lines:
            self.process_line(line)

        if not self.result.segments:
            msg = "Invalid HLS playlist: no segments found"
            raise InvalidDataError(msg)

        return self.result

    def process_line(self, line: str) -> None:
        """Process a single line from the playlist."""
        if line.startswith("#EXTINF:"):
            self._on_extinf(line)
        elif line.startswith("#EXT-X-KEY:"):
            self._on_key_line(line)
        elif line.startswith("#EXT-X-MAP:"):
            self._on_map_line(line)
        elif line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
            self._on_program_date_time(line)
        elif line.startswith("#EXT-X-BYTERANGE:"):
            self._on_byterange(line)
        elif line.startswith("#EXT-X-DISCONTINUITY"):
            self._on_discontinuity()
        elif line.startswith("#EXT"):
            self._on_ext_tag(line)
        elif line.startswith("#"):
            pass
        elif self.working_segment.extinf_line:
            self._on_segment_url(line)

    def _on_extinf(self, line: str) -> None:
        """Handle #EXTINF tag."""
        if self.working_segment.extinf_line:
            msg = (
                f"Malformed HLS playlist: #EXTINF '{line}' found without "
                f"preceding segment URL for '{self.working_segment.extinf_line}'"
            )
            raise InvalidDataError(msg)
        self.segments_started = True
        self.working_segment.extinf_line = line

    def _on_key_line(self, line: str) -> None:
        """Handle #EXT-X-KEY tag."""
        self.working_segment.key_line = line

    def _on_map_line(self, line: str) -> None:
        """Handle #EXT-X-MAP tag."""
        self.working_segment.map_line = line

    def _on_program_date_time(self, line: str) -> None:
        """Handle #EXT-X-PROGRAM-DATE-TIME tag."""
        self.working_segment.program_date_time = line

    def _on_byterange(self, line: str) -> None:
        """Handle #EXT-X-BYTERANGE tag."""
        self.working_segment.byterange_line = line

    def _on_discontinuity(self) -> None:
        """Handle #EXT-X-DISCONTINUITY tag."""
        self.working_segment.discontinuity = True

    def _on_ext_tag(self, line: str) -> None:
        """Handle other #EXT tags."""
        if self.segments_started:
            self.result.footer_lines.append(line)
        else:
            self.result.header_lines.append(line)

    def _on_segment_url(self, line: str) -> None:
        """Handle segment URL following #EXTINF."""
        self.working_segment.segment_url = line
        self.result.segments.append(self.working_segment)

        self.working_segment = HLSMediaSegment(
            key_line=self.working_segment.key_line,
            map_line=self.working_segment.map_line,
        )
