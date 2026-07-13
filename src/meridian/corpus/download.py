"""Resumable, checksummed downloader for PubMed baseline files.

The PubMed baseline is published as a numbered series of ``.xml.gz`` files, each
accompanied by a ``.md5`` companion. This module downloads them robustly:

- **Resumable at file granularity.** Each file is verified and kept, or re-fetched
  in full — the unit of progress is one baseline file. Re-running the downloader
  skips files already present and matching their stored checksum, so an
  interrupted run resumes without re-downloading completed files, using only local
  data (no network) to decide what to skip.
- **Checksummed.** Every downloaded file is verified against its ``.md5`` before it
  is trusted; a mismatch raises :class:`ChecksumError` rather than writing corrupt
  data.

Byte fetching is injected as a :class:`Fetcher`, so the download/resume/checksum
logic is exercised entirely offline in tests; only :class:`HttpFetcher` touches the
network.
"""

from __future__ import annotations

import hashlib
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

_MD5_HEX = re.compile(r"([a-fA-F0-9]{32})")


class ChecksumError(RuntimeError):
    """Raised when a downloaded file does not match its expected MD5."""


class Fetcher(Protocol):
    """Fetches the bytes at a URL, raising on any error."""

    def fetch(self, url: str) -> bytes: ...


class HttpFetcher:
    """A :class:`Fetcher` backed by ``urllib`` (the only networked component)."""

    def __init__(self, *, timeout: float = 60.0) -> None:
        self._timeout = timeout

    def fetch(self, url: str) -> bytes:  # pragma: no cover - network I/O
        with urllib.request.urlopen(url, timeout=self._timeout) as response:
            data: bytes = response.read()
            return data


def compute_md5(data: bytes) -> str:
    """Return the hex MD5 digest of ``data``."""
    return hashlib.md5(data).hexdigest()


def parse_md5(text: str) -> str:
    """Extract the 32-char hex digest from an NLM ``.md5`` file's contents.

    NLM writes lines like ``MD5(pubmed25n0001.xml.gz)= <hexdigest>``; a bare digest
    is also accepted.
    """
    match = _MD5_HEX.search(text)
    if match is None:
        raise ValueError("no MD5 digest found in checksum file")
    return match.group(1).lower()


def baseline_filename(number: int, *, release: str = "25", digits: int = 4) -> str:
    """Return the baseline filename for a file number, e.g. ``pubmed25n0001.xml.gz``."""
    return f"pubmed{release}n{number:0{digits}d}.xml.gz"


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Outcome of a single file download."""

    path: Path
    skipped: bool


def _md5_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".md5")


def _already_valid(dest: Path, verify: bool) -> bool:
    md5_path = _md5_path(dest)
    if not (dest.exists() and md5_path.exists()):
        return False
    if not verify:
        return True
    expected = parse_md5(md5_path.read_text())
    return compute_md5(dest.read_bytes()) == expected


def download_file(
    url: str,
    dest: Path,
    *,
    fetcher: Fetcher,
    verify: bool = True,
) -> DownloadResult:
    """Download ``url`` to ``dest`` (with its ``.md5``), skipping valid existing files.

    Raises :class:`ChecksumError` when ``verify`` is set and the fetched bytes do
    not match the fetched checksum.
    """
    if _already_valid(dest, verify):
        return DownloadResult(path=dest, skipped=True)

    data = fetcher.fetch(url)
    checksum_text = fetcher.fetch(url + ".md5")
    if verify:
        expected = parse_md5(checksum_text.decode("utf-8"))
        actual = compute_md5(data)
        if actual != expected:
            raise ChecksumError(f"{url}: expected MD5 {expected}, got {actual}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    _md5_path(dest).write_bytes(checksum_text)
    return DownloadResult(path=dest, skipped=False)


def download_baseline(
    base_url: str,
    filenames: list[str],
    dest_dir: Path,
    *,
    fetcher: Fetcher,
    verify: bool = True,
) -> list[DownloadResult]:
    """Download a set of baseline files into ``dest_dir``, resuming where possible."""
    base = base_url.rstrip("/")
    results: list[DownloadResult] = []
    for name in filenames:
        results.append(
            download_file(
                f"{base}/{name}",
                dest_dir / name,
                fetcher=fetcher,
                verify=verify,
            )
        )
    return results
