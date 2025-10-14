"""Utilities for extracting data from webpages."""


from __future__ import annotations
import asyncio
import re
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse

import markdown
import requests
import trafilatura
import yt_dlp
import fitz  # PyMuPDF
import pymupdf4llm as pf
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi


def _extract(raw_url: str) -> dict[str, Any]:
    url = prepare_url(raw_url)
    try:
        if "youtube.com" in url:
            extraction = _extract_youtube(url)
        elif url.endswith(".pdf"):
            extraction = _extract_pdf(url)
        else:
            extraction = _extract_webpage(url)

        if extraction is None:
            raise ValueError("Extraction failed: No content could be extracted from the webpage.")

        # Ensure we have a proper dict for meta
        raw_meta = extraction.get("meta", {})
        meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
        text_raw = extraction.get("text")
        text: str = text_raw if isinstance(text_raw, str) else ""
        markdown = extraction.get("markdown")
    except Exception as e:
        raise ValueError(f"Extraction failed with error: {str(e)}")

    if not meta.get("duration"): # get duration from word count
        word_count = len(text.split()) if text else 0
        meta["duration"] = int((word_count / 200) * 60)

    return {
        # Meta
        "url": meta.get("url"),
        "canonical_url": meta.get("canonical_url"),
        "title": meta.get("title"),
        "source_site": meta.get("source_site"),
        "format": meta.get("format"),
        "author": meta.get("author"),
        "publication_date": meta.get("publication_date"),
        "favicon_url": meta.get("favicon_url"),
        "duration": meta["duration"] if "duration" in meta else None,
        "server_status": "extracted",
        "server_status_at": datetime.now(),
        # Content
        "content_markdown": markdown,
        "content_text": text,
    }


async def extract_data(url: str) -> dict[str, Any]:
    return await asyncio.to_thread(_extract, url)


# === EXTRACTION FUNCTIONS ===


def _browser_headers(url: str) -> dict[str, str]:
    parsed = urlparse(url)
    referer = f"{parsed.scheme}://{parsed.netloc}"
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Referer": referer,
    }


def _extract_webpage(url: str) -> dict[str, Any] | None:
    """Extract text using trafilatura scraping"""
    downloaded = trafilatura.fetch_url(url)
    if not downloaded: # check for empty
        return None

    # Metadata
    raw_meta = trafilatura.metadata.extract_metadata(downloaded)
    if not raw_meta:
        raise ValueError("Extraction failed: No metadata could be extracted from the webpage.")
    canonical_raw = getattr(raw_meta, "canonical_url", None) or getattr(raw_meta, "url", None)
    canonical_url = _normalize_url(canonical_raw, url) or url

    meta = {
        "url": url,
        "canonical_url": canonical_url,
        "title": raw_meta.title,
        "source_site": raw_meta.sitename,
        "type": None, # classify with LLM
        "format": "webpage",
        "author": raw_meta.author,
        "publication_date": raw_meta.date,
        "favicon_url": _build_favicon_url(urlparse(url)),
    }

    # Markdown
    markdown = trafilatura.extract(
        downloaded,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        include_links=True,
        favor_precision=True,
        deduplicate=True,
    )
    if not markdown or not markdown.strip():
        raise ValueError("Extraction failed: No content could be extracted from the webpage.")

    cleaned_markdown = clean_markdown(markdown)

    return {
        "meta": meta,
        "markdown": cleaned_markdown,
        "text": _strip_markdown(cleaned_markdown),
    }


def _extract_pdf(url: str) -> dict[str, object] | None:
    max_size_mb = 250
    session = requests.Session()
    headers = _browser_headers(url)

    head_headers: dict[str, str] = {}
    head_response: requests.Response | None = None
    try:
        head_response = session.head(url, timeout=30, allow_redirects=True, headers=headers)
        if head_response.ok:
            head_headers = dict(head_response.headers)
        elif head_response.status_code in {403, 405, 501}:
            head_headers = {}
        else:
            head_response.raise_for_status()
    except requests.RequestException:
        head_headers = {}

    cl = head_headers.get("Content-Length")
    if cl:
        try:
            if int(cl) / (1024 * 1024) > max_size_mb:
                raise ValueError(f"PDF too large: {int(cl)/(1024*1024):.2f}MB (max: {max_size_mb}MB)")
        except ValueError:
            pass

    r = session.get(url, timeout=30, headers=headers)
    r.raise_for_status()

    dl_cl = r.headers.get("Content-Length")
    if not cl and dl_cl:
        try:
            if int(dl_cl) / (1024 * 1024) > max_size_mb:
                raise ValueError(f"PDF too large: {int(dl_cl)/(1024*1024):.2f}MB (max: {max_size_mb}MB)")
        except ValueError:
            pass

    # Open PDF from memory to avoid writing temp files
    with fitz.open(stream=r.content, filetype="pdf") as doc:
        data = pf.to_markdown(doc, page_chunks=True)

    # page_chunks=True => list of {"text": ..., "metadata": ...}
    pages = data if isinstance(data, list) else [{"text": data, "metadata": {}}]
    page0 = pages[0] if pages else {}
    raw_meta = page0.get("metadata", {}) if isinstance(page0, dict) else {}

    markdown = "\n\n".join(p.get("text", "") if isinstance(p, dict) else "" for p in pages).strip()

    publication_date = _parse_pdf_date(raw_meta.get("creationDate")) or _parse_pdf_date(raw_meta.get("modDate"))

    meta = {
        "url": url,
        "canonical_url": url,
        "title": _pick_pdf_title(raw_meta, head_headers, markdown),
        "source_site": urlparse(url).netloc,
        "type": None,
        "format": "pdf",
        "author": raw_meta.get("author"),
        "publication_date": publication_date,
        "favicon_url": _build_favicon_url(urlparse(url)),
    }

    cleaned_markdown = clean_markdown(markdown)

    return {
        "meta": meta,
        "markdown": cleaned_markdown,
        "text": _strip_markdown(cleaned_markdown),
    }


def _extract_youtube(url: str) -> dict[str, Any]:
    with yt_dlp.YoutubeDL({'quiet': True, 'skip_download': True}) as ydl:  # type: ignore
        info = ydl.extract_info(url, download=False)

    # metadata
    video_page_url = info.get("webpage_url", url)
    parsed_video_url = urlparse(video_page_url)

    # Get channel avatar - fetch from channel page
    favicon_url = None
    channel_id = info.get("channel_id")
    if channel_id:
        favicon_url = _get_youtube_channel_avatar(channel_id)
    # Fallback to generic YouTube favicon if extraction fails
    if not favicon_url:
        favicon_url = _build_favicon_url(parsed_video_url)

    meta = {
        "url": video_page_url,
        "canonical_url": video_page_url,
        "title": info.get("title"),
        "source_site": "YouTube",
        "type": "video",
        "format": "youtube",
        "author": info.get("uploader"),
        "publication_date": info.get("upload_date"),
        "favicon_url": favicon_url,
        "duration": info.get("duration"),
    }

    # Text content
    text: str | None = None
    try:
        video_id = info.get("id")
        if video_id:
            api = YouTubeTranscriptApi()
            try:
                transcript_list = api.list(video_id)
            except Exception as exc:
                print(
                    f"YouTube transcript fetch failed for {url}: unable to list transcripts: {exc}"
                )
            else:
                transcript = None
                try:
                    transcript = transcript_list.find_generated_transcript(
                        ["en", "en-US", "en-GB"]
                    )
                except Exception:
                    try:
                        transcript = next(iter(transcript_list))
                    except StopIteration:
                        print(
                            f"YouTube transcript fetch failed for {url}: no transcripts available"
                        )
                        transcript = None
                    else:
                        lang_code = getattr(transcript, "language_code", "") or ""
                        if lang_code and not lang_code.startswith("en"):
                            try:
                                transcript = transcript.translate("en")
                            except Exception as exc:
                                print(
                                    f"YouTube transcript fetch failed for {url}: translate to en failed: {exc}"
                                )
                                transcript = None

                if transcript is not None:
                    try:
                        fetched = transcript.fetch()
                        entries = (
                            fetched.to_raw_data()
                            if hasattr(fetched, "to_raw_data")
                            else list(fetched)
                        )
                        text_parts = []
                        for entry in entries:
                            value = None
                            if isinstance(entry, dict):
                                value = entry.get("text")
                            else:
                                value = getattr(entry, "text", None)
                            if value:
                                text_parts.append(value)
                        text = " ".join(text_parts) if text_parts else None
                    except Exception as exc:
                        print(
                            f"YouTube transcript fetch failed for {url}: fetch entries failed: {exc}"
                        )
    except Exception as exc:
        print(f"YouTube transcript fetch failed for {url}: {exc}")
        text = None  # no captions available or access blocked

    return {"meta": meta, "text": text, "markdown": None}


# === UTILITIES ===


def _build_favicon_url(parsed_url) -> str | None:
    if not parsed_url.scheme or not parsed_url.netloc:
        return None
    return urlunparse((parsed_url.scheme, parsed_url.netloc, "/favicon.ico", "", "", ""))


def _get_youtube_channel_avatar(channel_id: str) -> str | None:
    """Extract channel avatar URL from YouTube channel page."""
    try:
        channel_url = f"https://www.youtube.com/channel/{channel_id}"
        response = requests.get(channel_url, timeout=10)
        response.raise_for_status()

        # YouTube embeds avatar URLs in JSON data within the page
        avatar_pattern = r'"avatar":\{"thumbnails":\[\{"url":"([^"]+)"'
        match = re.search(avatar_pattern, response.text)
        if match:
            return match.group(1)
    except Exception:
        # If extraction fails, return None to use fallback
        pass
    return None


def prepare_url(raw_url: str) -> str:
    """Check valid url and add https if missing"""
    candidate = raw_url.strip()
    if not candidate:
        raise ValueError("URL is required")

    if not candidate.startswith(("http://", "https://")):
        candidate = f"https://{candidate}"

    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Please enter a valid URL")

    # URL validation
    domain_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
    if not re.match(domain_pattern, parsed.netloc.split(':')[0]):
        raise ValueError("Please enter a valid URL with a proper domain")

    # Check for localhost and IP addresses
    netloc_host = parsed.netloc.split(':')[0]
    if netloc_host == 'localhost' or re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', netloc_host):
        pass          # Allow localhost and IP addresses for development
    elif not re.match(domain_pattern, netloc_host):
        raise ValueError("Please enter a valid URL with a proper domain")
    elif '.' not in netloc_host:
        raise ValueError("Please enter a valid URL with a proper domain")

    return candidate


# Backwards compatibility for legacy imports
_prepare_url = prepare_url


def _normalize_url(value: str | None, base: str) -> str | None:
    if not value:
        return None
    candidate = urljoin(base, value)
    parsed = urlparse(candidate)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return candidate
    return None


def _strip_markdown(md_text):
    if not md_text:
        return ""
    html = markdown.markdown(md_text)
    return BeautifulSoup(html, 'html.parser').get_text()


def clean_markdown(markdown_text: str) -> str:
    """Normalize extracted markdown and drop trailing reference sections."""
    if not markdown_text:
        return ""
    cleaned = _postprocess_markdown(markdown_text)
    return _clean_article(cleaned)


def _clean_article(markdown_text: str) -> str:
    """Trim markdown after headings like 'References' which typically introduce citation lists."""
    _REFERENCE_HEADINGS = {
        "references",
        "reference",
        "bibliography",
        "works cited",
        "citations",
    }

    if not markdown_text:
        return ""

    lines = markdown_text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip(" -*_\t").rstrip(":").lower()
        else:
            heading = stripped.strip(" -*_\t").rstrip(":").lower()
        if heading in _REFERENCE_HEADINGS and idx > 0:
            cleaned = "\n".join(lines[:idx]).rstrip()
            return cleaned if cleaned else markdown_text

    return markdown_text


def _parse_pdf_date(value: Any) -> datetime | None:
    """Convert common PDF date formats (e.g. D:YYYYMMDDHHmmSSOHH'mm') into aware datetimes."""
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    if text.startswith("D:"):
        text = text[2:]

    iso_candidate = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(iso_candidate)
    except ValueError:
        pass

    normalized = text.replace("'", "")
    m = re.match(
        r"^(?P<year>\d{4})"
        r"(?P<month>\d{2})?"
        r"(?P<day>\d{2})?"
        r"(?P<hour>\d{2})?"
        r"(?P<minute>\d{2})?"
        r"(?P<second>\d{2})?"
        r"(?P<timezone>.*)$",
        normalized,
    )
    if not m:
        return None

    def _to_int(part: str | None, default: int) -> int:
        return int(part) if part and part.isdigit() else default

    year = int(m.group("year"))
    month = _to_int(m.group("month"), 1)
    day = _to_int(m.group("day"), 1)
    hour = _to_int(m.group("hour"), 0)
    minute = _to_int(m.group("minute"), 0)
    second = _to_int(m.group("second"), 0)

    tz_part = m.group("timezone") or ""
    tzinfo = None
    if tz_part:
        tz_char = tz_part[0]
        if tz_char in "+-":
            tz_hours = _to_int(tz_part[1:3], 0)
            tz_minutes = _to_int(tz_part[3:5], 0)
            offset = timedelta(hours=tz_hours, minutes=tz_minutes)
            if tz_char == "-":
                offset = -offset
            tzinfo = timezone(offset)
        elif tz_char == "Z":
            tzinfo = timezone.utc

    try:
        return datetime(year, month, day, hour, minute, second, tzinfo=tzinfo)
    except ValueError:
        return None


def _postprocess_markdown(md: str) -> str:
    # Fix common PDF artifacts and improve Markdown structure.
    md = md.replace("\r\n", "\n").replace("\r", "\n")

    # Join hyphenated line-break words: "equa-\n tion" → "equation"
    md = re.sub(r"(\w)-\n(\w)", r"\1\2", md)

    # Collapse hard-wrapped lines within paragraphs while keeping list/code/heading blocks
    md = _unwrap_paragraphs(md)

    # Normalize headings: lines in ALL CAPS or Title Case followed by blank → turn into ##
    md = re.sub(r"(?m)^(?P<h>[A-Z0-9][A-Z0-9 \t/:,&\-\(\)]{3,})\n(?=\n)", r"## \g<h>\n", md)

    # Ensure lists render correctly
    md = re.sub(r"(?m)^[•·]\s+", "- ", md)   # bullets → hyphen
    md = re.sub(r"(?m)^(?P<i>\s{0,3})(\d+)[\)\.]\s+", r"\g<i>\2. ", md)

    # Dedent over-indented blocks
    md = re.sub(r"(?m)^\s{4,}(?![-*`0-9])", "", md)

    # Keep display math blocks intact if Nougat produced them
    # Add blank lines around $$ … $$ for Markdown engines
    md = re.sub(r"(?s)(\$\$.*?\$\$)", r"\n\1\n", md)

    # Collapse multiple blank lines
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return md


def _unwrap_paragraphs(md: str) -> str:
    lines = md.split("\n")
    out = []
    buf = []
    def flush():
        if not buf:
            return
        # join wrapped lines but keep single spaces
        joined = re.sub(r"\s*\n\s*", " ", "\n".join(buf)).strip()
        out.append(joined)
        buf.clear()

    fence = False
    for ln in lines:
        if re.match(r"^```", ln):
            fence = not fence
            flush()
            out.append(ln)
            continue
        block = fence or re.match(r"^\s*([*+-]\s|\d+\.\s|#|>|$)", ln)
        if block:
            flush()
            out.append(ln)
        else:
            buf.append(ln)
    flush()
    return "\n".join(out)


def _pick_pdf_title(raw_meta: dict, headers: dict, markdown: str | None = None) -> str | None:
    # 1. Prefer metadata if valid
    t = raw_meta.get("title")
    if t and t.strip() and t.strip().lower() not in {"title", "untitled"}:
        return t.strip()

    # 2. Try filename from Content-Disposition
    cd = headers.get("Content-Disposition", "")
    m = re.search(r'filename="?([^"]+)"?', cd)
    if m:
        name = m.group(1)
        name = re.sub(r"\.pdf$", "", name, flags=re.I)
        return name.strip()

    # 3. Try first markdown heading
    if markdown:
        m = re.search(r"^#+\s+(.+)$", markdown, flags=re.M)
        if m:
            return m.group(1).strip()

    return None
