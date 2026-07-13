"""Offline tests for the resumable, checksummed downloader (injected fetcher)."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from meridian.corpus.download import (
    ChecksumError,
    DownloadResult,
    baseline_filename,
    compute_md5,
    download_baseline,
    download_file,
    parse_md5,
)


class DictFetcher:
    """A :class:`Fetcher` serving bytes from an in-memory URL map; counts calls."""

    def __init__(self, contents: dict[str, bytes]) -> None:
        self._contents = contents
        self.calls: list[str] = []

    def fetch(self, url: str) -> bytes:
        self.calls.append(url)
        try:
            return self._contents[url]
        except KeyError as exc:  # pragma: no cover - guards test misuse
            raise FileNotFoundError(url) from exc


def _md5_line(name: str, data: bytes) -> bytes:
    return f"MD5({name})= {compute_md5(data)}".encode()


def test_parse_md5_from_nlm_line() -> None:
    assert parse_md5("MD5(pubmed25n0001.xml.gz)= D41D8CD98F00B204E9800998ECF8427E") == (
        "d41d8cd98f00b204e9800998ecf8427e"
    )


def test_parse_md5_accepts_bare_digest() -> None:
    digest = "0" * 32
    assert parse_md5(digest) == digest


def test_parse_md5_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_md5("no digest here")


def test_baseline_filename_format() -> None:
    assert baseline_filename(1) == "pubmed25n0001.xml.gz"
    assert baseline_filename(1234, release="26") == "pubmed26n1234.xml.gz"


def test_download_file_writes_data_and_checksum(tmp_path: Path) -> None:
    data = gzip.compress(b"<PubmedArticleSet/>")
    url = "https://example/pubmed25n0001.xml.gz"
    fetcher = DictFetcher({url: data, url + ".md5": _md5_line("pubmed25n0001.xml.gz", data)})
    dest = tmp_path / "pubmed25n0001.xml.gz"

    result = download_file(url, dest, fetcher=fetcher)

    assert result == DownloadResult(path=dest, skipped=False)
    assert dest.read_bytes() == data
    assert (tmp_path / "pubmed25n0001.xml.gz.md5").exists()


def test_download_file_detects_corruption(tmp_path: Path) -> None:
    url = "https://example/f.xml.gz"
    wrong_md5 = "MD5(f.xml.gz)= " + "a" * 32
    fetcher = DictFetcher({url: b"payload", url + ".md5": wrong_md5.encode()})
    with pytest.raises(ChecksumError):
        download_file(url, tmp_path / "f.xml.gz", fetcher=fetcher)
    assert not (tmp_path / "f.xml.gz").exists()  # corrupt data never written


def test_resume_skips_valid_existing_file_without_refetching(tmp_path: Path) -> None:
    data = b"already here"
    url = "https://example/f.xml.gz"
    fetcher = DictFetcher({url: data, url + ".md5": _md5_line("f.xml.gz", data)})
    dest = tmp_path / "f.xml.gz"

    first = download_file(url, dest, fetcher=fetcher)
    calls_after_first = len(fetcher.calls)
    second = download_file(url, dest, fetcher=fetcher)

    assert first.skipped is False
    assert second.skipped is True
    assert len(fetcher.calls) == calls_after_first  # no network on resume


def test_refetches_when_local_file_corrupt(tmp_path: Path) -> None:
    data = b"good data"
    url = "https://example/f.xml.gz"
    fetcher = DictFetcher({url: data, url + ".md5": _md5_line("f.xml.gz", data)})
    dest = tmp_path / "f.xml.gz"
    (tmp_path / "f.xml.gz.md5").write_bytes(_md5_line("f.xml.gz", data))
    dest.write_bytes(b"corrupt")  # present but does not match checksum

    result = download_file(url, dest, fetcher=fetcher)

    assert result.skipped is False
    assert dest.read_bytes() == data


def test_verify_disabled_skips_present_file(tmp_path: Path) -> None:
    url = "https://example/f.xml.gz"
    fetcher = DictFetcher({})
    dest = tmp_path / "f.xml.gz"
    dest.write_bytes(b"whatever")
    (tmp_path / "f.xml.gz.md5").write_bytes(b"unused")

    result = download_file(url, dest, fetcher=fetcher, verify=False)

    assert result.skipped is True
    assert fetcher.calls == []


def test_download_baseline_iterates(tmp_path: Path) -> None:
    names = ["pubmed25n0001.xml.gz", "pubmed25n0002.xml.gz"]
    contents: dict[str, bytes] = {}
    for name in names:
        payload = f"data-{name}".encode()
        contents[f"https://base/{name}"] = payload
        contents[f"https://base/{name}.md5"] = _md5_line(name, payload)
    fetcher = DictFetcher(contents)

    results = download_baseline("https://base/", names, tmp_path, fetcher=fetcher)

    assert [r.path.name for r in results] == names
    assert all(not r.skipped for r in results)
