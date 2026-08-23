"""Download and normalize official PAIMANA Project Monitoring archive reports.

Raw PDFs are immutable inputs. The parser intentionally keeps unavailable report
fields null instead of synthesizing values. Archive discovery is resilient: live
PAIMANA discovery is preferred, but checked-in provenance can be used when the
public endpoint is temporarily unavailable.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from html import unescape
from pathlib import Path
import json
import re
import shutil
import subprocess
import time
from urllib.parse import parse_qs, quote, urljoin, urlparse
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_DIR = ROOT / "data" / "raw" / "paimana_archive"
MANIFEST_PATH = ARCHIVE_DIR / "manifest.json"
OUTPUT_PATH = ROOT / "data" / "processed" / "project_monthly_history.csv"
BASE_URL = "https://paimana-proj.mospi.gov.in"
ARCHIVE_PAGE = f"{BASE_URL}/ReportPage/ArchiveProjectMonitoring"
INDEX_URL = f"{BASE_URL}/ReportPage/ArchiveReport?fyear=N&month=0&quater=0&reportType=F"
DEFAULT_FETCH_TIMEOUT_SECONDS = 20
DEFAULT_FETCH_ATTEMPTS = 3

OUTPUT_COLUMNS = [
    "project_id", "project_name", "sector", "ministry", "state", "implementing_agency",
    "original_cost", "revised_cost", "current_expenditure", "planned_start_date",
    "planned_completion_date", "revised_completion_date", "actual_completion_date", "month",
    "physical_progress_percentage", "financial_progress_percentage", "milestone_status", "delay_months",
    "source_report", "source_url",
]


def _fetch(url: str, timeout: int = DEFAULT_FETCH_TIMEOUT_SECONDS, attempts: int = DEFAULT_FETCH_ATTEMPTS) -> bytes:
    """Fetch a public PAIMANA resource with bounded retry/backoff."""
    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            request = Request(url, headers={"User-Agent": "InfraSight-SIH26103/1.0 (+public PAIMANA archive ingestion)"})
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:  # network failures are surfaced after bounded retries
            last_error = exc
            if attempt + 1 < max(1, attempts):
                time.sleep(min(2 ** attempt, 4))
    assert last_error is not None
    raise last_error


def _manifest_records() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    try:
        payload = json.loads(MANIFEST_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def cached_archive_reports() -> list[dict]:
    """Return unique report provenance already known to the repository."""
    reports: dict[tuple[str, str, str], dict] = {}
    for item in _manifest_records():
        financial_year = str(item.get("financial_year") or "").strip()
        label = str(item.get("label") or "").strip()
        report_id = str(item.get("id") or "").strip()
        if not financial_year or not label:
            continue
        reports[(financial_year, label, report_id)] = {
            "financial_year": financial_year,
            "label": label,
            "id": report_id,
            "archive_path": item.get("archive_path") or "",
            "url": item.get("url") or "",
        }
    return sorted(reports.values(), key=lambda row: (row["financial_year"], row["label"], row["id"]))


def _reports_from_index(payload: bytes) -> list[dict]:
    """Parse both PAIMANA archive-index response shapes observed over time."""
    decoded = json.loads(payload.decode("utf-8"))
    reports: list[dict] = []

    if isinstance(decoded, dict) and isinstance(decoded.get("html"), str):
        html = unescape(decoded["html"])
        for row in re.findall(r"<tr>(.*?)</tr>", html, flags=re.I | re.S):
            cells = [re.sub(r"<[^>]+>", "", value).strip() for value in re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.I | re.S)]
            link = re.search(r"href=['\"]([^'\"]+)['\"]", row, flags=re.I)
            if len(cells) < 3 or not link:
                continue
            relative = link.group(1).replace("\\", "/")
            parsed = urlparse(urljoin(BASE_URL, relative))
            if parsed.netloc != urlparse(BASE_URL).netloc or not parsed.path.endswith("/ReportPage/ViewPdf"):
                continue
            query = parse_qs(parsed.query)
            raw_path = query.get("path", [""])[0]
            report_id = query.get("id", [""])[0]
            reports.append({
                "financial_year": cells[1], "label": cells[2], "id": report_id,
                "archive_path": raw_path,
                "url": f"{BASE_URL}/ReportPage/ViewPdf?id={quote(report_id)}&path={quote(raw_path)}",
            })
    elif isinstance(decoded, list):
        # Older endpoint shape: groups with a ``data`` list of report objects.
        for group in decoded:
            data = group.get("data", []) if isinstance(group, dict) else []
            for item in data:
                if not isinstance(item, dict) or item.get("ShowReport") not in {None, "F"}:
                    continue
                financial_year = str(item.get("FinancialYear") or group.get("label") or "").strip()
                label = str(item.get("Month") or item.get("Quater") or item.get("Title") or "report").strip()
                report_id = str(item.get("id") or "").strip()
                raw_path = str(item.get("FilePath") or "").replace("\\", "/")
                if not financial_year or not raw_path:
                    continue
                reports.append({
                    "financial_year": financial_year, "label": label, "id": report_id,
                    "archive_path": raw_path,
                    "url": f"{BASE_URL}/ReportPage/ViewPdf?id={quote(report_id)}&path={quote(raw_path)}",
                })

    deduped = {(r["financial_year"], r["label"], r["id"], r["archive_path"]): r for r in reports}
    return sorted(deduped.values(), key=lambda row: (row["financial_year"], row["label"], row["id"]))


def discover_archive_reports(*, allow_cached: bool = True, timeout: int = DEFAULT_FETCH_TIMEOUT_SECONDS) -> list[dict]:
    """Discover archive reports, falling back to checked-in provenance on outage."""
    try:
        live = _reports_from_index(_fetch(INDEX_URL, timeout=timeout))
        if live:
            return live
        if not allow_cached:
            raise RuntimeError("PAIMANA archive index returned no report records")
    except Exception:
        if not allow_cached:
            raise
    cached = cached_archive_reports()
    if cached:
        return cached
    raise RuntimeError("PAIMANA archive discovery failed and no cached manifest is available")


def archive_discovery_status(timeout: int = 8) -> dict:
    """Machine-readable live/cache discovery status for experiment audits."""
    cached = cached_archive_reports()
    try:
        live = _reports_from_index(_fetch(INDEX_URL, timeout=timeout, attempts=1))
        if live:
            return {"source": "live", "reports": live, "error": None}
        return {"source": "cache", "reports": cached, "error": "live archive index returned zero reports"}
    except Exception as exc:
        return {"source": "cache", "reports": cached, "error": f"{type(exc).__name__}: {exc}"}


def _report_month(financial_year: str, label: str) -> pd.Timestamp | None:
    match = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)", label, re.I)
    if not match or not re.match(r"^\d{4}", financial_year):
        return None
    month = datetime.strptime(match.group(1).title(), "%B").month
    start_year = int(financial_year[:4])
    year = start_year + 1 if month <= 3 else start_year
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def _report_key(item: dict) -> tuple[str, str, str]:
    return (str(item.get("financial_year") or ""), str(item.get("label") or ""), str(item.get("id") or ""))


def _local_pdf_path(record: dict) -> Path:
    filename = str(record.get("filename") or "").replace("\\", "/")
    if filename:
        candidate = ARCHIVE_DIR / filename
        if candidate.exists():
            return candidate
    source_name = Path(str(record.get("archive_path") or "").replace("\\", "/")).name
    legacy = ARCHIVE_DIR / source_name
    if legacy.exists():
        return legacy
    financial_year = re.sub(r"[^0-9A-Za-z-]+", "_", str(record.get("financial_year") or "unknown"))
    report_id = re.sub(r"[^0-9A-Za-z-]+", "_", str(record.get("id") or "na"))
    return ARCHIVE_DIR / financial_year / f"{report_id}__{source_name or 'report.pdf'}"


def download_archive_reports(
    financial_year: str | None = None,
    labels: set[str] | None = None,
    force: bool = False,
    *,
    allow_cached_discovery: bool = True,
) -> list[dict]:
    """Download selected reports; ``financial_year=None`` means every discovered year.

    Existing manifest entries are retained. This prevents a transient source outage
    or one failed report from erasing previously downloaded provenance.
    """
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    reports = discover_archive_reports(allow_cached=allow_cached_discovery)
    selected = [
        r for r in reports
        if (financial_year is None or r["financial_year"] == financial_year)
        and (labels is None or r["label"] in labels)
    ]
    existing = {_report_key(item): item for item in _manifest_records()}
    now = datetime.now(timezone.utc).isoformat()

    for report in selected:
        source_name = Path(report["archive_path"].replace("\\", "/")).name or f"report-{report.get('id') or 'unknown'}.pdf"
        year_dir = re.sub(r"[^0-9A-Za-z-]+", "_", report["financial_year"])
        report_id = re.sub(r"[^0-9A-Za-z-]+", "_", str(report.get("id") or "na"))
        relative_name = f"{year_dir}/{report_id}__{source_name}"
        target = ARCHIVE_DIR / relative_name
        legacy = ARCHIVE_DIR / source_name
        try:
            if not force and target.exists():
                pass
            elif not force and legacy.exists():
                # Preserve the immutable checked-in legacy PDF rather than redownload.
                target = legacy
                relative_name = source_name
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                data = _fetch(report["url"])
                if not data.startswith(b"%PDF"):
                    raise ValueError(f"PAIMANA response for {report['label']} was not a PDF")
                target.write_bytes(data)
            record = {
                **report, "filename": relative_name, "status": "downloaded",
                "bytes": target.stat().st_size,
                "sha256": sha256(target.read_bytes()).hexdigest(),
                "downloaded_at_utc": now,
            }
        except Exception as exc:
            previous = existing.get(_report_key(report))
            # If a verified previous local copy exists, retain it and record refresh failure.
            if previous and previous.get("status") == "downloaded" and _local_pdf_path(previous).exists():
                record = {**previous, "refresh_error": f"{type(exc).__name__}: {exc}", "refresh_attempted_at_utc": now}
            else:
                record = {
                    **report, "filename": relative_name, "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}", "downloaded_at_utc": now,
                }
        existing[_report_key(report)] = record

    manifest = sorted(existing.values(), key=lambda row: _report_key(row))
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest


def extract_report_text(pdf_path: Path) -> str:
    pdftotext = shutil.which("pdftotext")
    if pdftotext:
        completed = subprocess.run([pdftotext, "-layout", str(pdf_path), "-"], check=True, capture_output=True)
        return completed.stdout.decode("utf-8", errors="replace")
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install pypdf or Poppler pdftotext to parse PAIMANA PDFs") from exc
    return "\n".join(page.extract_text() or "" for page in PdfReader(pdf_path).pages)


def _number(value: str | None) -> float | None:
    if not value or value.strip() in {"-", "N.A."}:
        return None
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def _month_date(value: str | None) -> str | None:
    if not value or value.strip() in {"/", "-", "N.A."}:
        return None
    parsed = pd.to_datetime(value.strip(), format="%m/%Y", errors="coerce")
    return None if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def parse_project_list(text: str, report_month: pd.Timestamp, source_report: str, source_url: str) -> pd.DataFrame:
    """Parse the official June-2024-and-later table layout containing project codes."""
    marker = text.lower().find("project list: ongoing projects as of")
    if marker < 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    lines = text[marker:].splitlines()
    start_re = re.compile(r"^(?P<head>.*?)\s{2,}(?P<approval>\d{1,2}/\d{4})\s+(?P<completion>\d{1,2}/\d{4}|/)\s+(?P<cost>[\d,]+\.\d{2})\s+(?P<expenditure>[\d,]+\.\d{2})\s+(?P<progress>\d+(?:\.\d+)?)\s*$")
    starts = []
    for index, line in enumerate(lines):
        match = start_re.match(line)
        if not match:
            continue
        head = match.group("head")
        sequence = re.search(r"(?<!\S)(\d{1,4})\s+(.+)$", head)
        if sequence:
            starts.append((index, match, sequence))
    rows = []
    for position, (line_index, match, sequence) in enumerate(starts):
        block = lines[line_index + 1: starts[position + 1][0] if position + 1 < len(starts) else len(lines)]
        code = agency = None
        name_parts = [sequence.group(2).strip()]
        for line in block:
            code_match = re.fullmatch(r"\s*\(([A-Z]?\d{8})\s*\)\s*", line)
            if code_match:
                code = code_match.group(1)
                break
            agency_match = re.fullmatch(r"\s*\(([^{}]+?)\s*\)\s*", line)
            if agency_match and agency_match.group(1).strip() not in {"N.A.", "-"}:
                agency = agency_match.group(1).strip()
                continue
            project_fragment = line[20:58].strip() if len(line) > 20 else ""
            if project_fragment and not any(token in project_fragment for token in ["{", "(", "Table:", "of 302"]):
                name_parts.append(project_fragment)
        if not code:
            continue
        braces = re.findall(r"\{([^}]+)\}", "\n".join(block))
        parentheses = re.findall(r"\(([^)]+)\)", "\n".join(block))
        anticipated_completion = next((_month_date(x) for x in braces if _month_date(x)), None)
        anticipated_cost = next((_number(x) for x in braces if _number(x) is not None), None)
        revised_completion = next((_month_date(x) for x in parentheses if _month_date(x)), None)
        revised_cost = next((_number(x) for x in parentheses if _number(x) is not None), None)
        original_cost = _number(match.group("cost"))
        expenditure = _number(match.group("expenditure"))
        progress = _number(match.group("progress"))
        completion = _month_date(match.group("completion"))
        effective_cost = anticipated_cost or revised_cost or original_cost
        financial_progress = expenditure / effective_cost * 100 if expenditure is not None and effective_cost else None
        delay = None
        if completion and anticipated_completion:
            start, end = pd.Timestamp(completion), pd.Timestamp(anticipated_completion)
            delay = (end.year - start.year) * 12 + end.month - start.month
        rows.append({
            "project_id": code, "project_name": " ".join(name_parts).strip(), "sector": None, "ministry": None,
            "state": None, "implementing_agency": agency, "original_cost": original_cost, "revised_cost": revised_cost,
            "current_expenditure": expenditure, "planned_start_date": None, "planned_completion_date": completion,
            "revised_completion_date": revised_completion or anticipated_completion, "actual_completion_date": None,
            "month": report_month.strftime("%Y-%m-%d"), "physical_progress_percentage": progress,
            "financial_progress_percentage": None if financial_progress is None else round(financial_progress, 4),
            "milestone_status": None, "delay_months": delay, "source_report": source_report, "source_url": source_url,
        })
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def build_monthly_history(manifest: list[dict] | None = None, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    if manifest is None:
        manifest = _manifest_records()
    frames = []
    for report in manifest:
        if report.get("status") != "downloaded":
            continue
        month = _report_month(str(report.get("financial_year") or ""), str(report.get("label") or ""))
        if month is None:
            continue
        pdf = _local_pdf_path(report)
        if not pdf.exists():
            continue
        frame = parse_project_list(extract_report_text(pdf), month, pdf.name, str(report.get("url") or ""))
        if not frame.empty:
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    if not result.empty:
        result = result.drop_duplicates(["project_id", "month"], keep="last").sort_values(["project_id", "month"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def ingest_archive_history(
    financial_year: str | None = None,
    labels: set[str] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Ingest all available official history, or one explicitly requested year."""
    manifest = download_archive_reports(financial_year, labels=labels, force=force)
    return build_monthly_history(manifest)


def ingest_latest_archive() -> pd.DataFrame:
    """Backward-compatible helper for the checked-in 2024-25 archive cohort."""
    labels = {"June", "October", "November", "December", "January", "February", "March"}
    return ingest_archive_history("2024-25", labels=labels)
