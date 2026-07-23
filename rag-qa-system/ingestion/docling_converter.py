"""Docling-based document conversion with RapidOCR for scanned content.

PDF cascade (lightweight hybrid):
1. Extract embedded text with PyMuPDF.
2. If text is sparse (scanned-like) -> full-page Tesseract OCR.
3. If text is rich AND the page has images -> OCR image regions and append.
4. Prefer Docling when available; fall back to this hybrid path on failure.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

# Windows often lacks symlink privileges; HuggingFace model downloads fail with WinError 1314.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from ingestion.types import LoadedDocument, SourceType
from ingestion.tesseract_ocr import (
	merge_text_and_ocr,
	ocr_image_bytes,
	ocr_pdf_page_images,
	ocr_pdf_page_render,
	tesseract_available,
	tesseract_status_message,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp", ".gif"}
DOCLING_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".html", ".htm", *IMAGE_EXTENSIONS}

_CONVERTER = None
_MIN_TEXT_CHARS = 10
# Pages below this are treated as scanned / text-poor and get full-page OCR.
_TEXT_RICH_THRESHOLD = 50
_DOCLING_IMPORT_ERROR: Exception | None = None


def docling_available() -> bool:
	"""Return True when Docling and its OCR backend are importable."""
	global _DOCLING_IMPORT_ERROR
	try:
		from docling.document_converter import DocumentConverter  # noqa: F401
		return True
	except ImportError as exc:
		_DOCLING_IMPORT_ERROR = exc
		return False


def _require_docling(feature: str) -> None:
	if docling_available():
		return
	message = (
		f"{feature} requires Docling. Install dependencies with:\n"
		"pip install docling rapidocr-onnxruntime onnxruntime"
	)
	if _DOCLING_IMPORT_ERROR is not None:
		raise ImportError(message) from _DOCLING_IMPORT_ERROR
	raise ImportError(message)


def _get_converter():
	"""Return a cached Docling DocumentConverter with RapidOCR enabled for PDFs."""
	global _CONVERTER
	if _CONVERTER is not None:
		return _CONVERTER

	_require_docling("Document conversion")

	from docling.datamodel.base_models import InputFormat
	from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
	from docling.document_converter import DocumentConverter, PdfFormatOption

	pipeline_options = PdfPipelineOptions(
		do_ocr=True,
		ocr_options=RapidOcrOptions(backend="onnxruntime"),
	)
	_CONVERTER = DocumentConverter(
		format_options={
			InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
		}
	)
	return _CONVERTER


def _normalize_text(text: str) -> str:
	return re.sub(r"\s+", " ", (text or "").strip())


def _page_has_images(page) -> bool:
	"""Return True when the PDF page embeds at least one image XObject."""
	try:
		return bool(page.get_images(full=True))
	except Exception:
		return False


def _convert_pdf_hybrid(path: Path) -> List[LoadedDocument]:
	"""Hybrid PDF extraction: PyMuPDF text + Tesseract for scans / image regions.

	Rules per page:
	- text-poor page -> full-page Tesseract OCR (if available)
	- text-rich + images -> keep text layer AND OCR embedded images
	- text-rich, no images -> text layer only
	"""
	import fitz

	if not tesseract_available():
		msg = tesseract_status_message()
		if msg:
			logger.warning(msg)

	entries: List[LoadedDocument] = []
	with fitz.open(str(path)) as doc:
		for i in range(doc.page_count):
			page = doc.load_page(i)
			page_label = i + 1
			text = (page.get_text("text") or "").strip()
			has_images = _page_has_images(page)
			parser = "pymupdf"
			combined = text

			if len(text) < _TEXT_RICH_THRESHOLD:
				if tesseract_available():
					ocr_text = ocr_pdf_page_render(page)
					combined = merge_text_and_ocr(text, ocr_text)
					parser = "pymupdf+tesseract" if text else "tesseract"
					logger.info(
						"Page %s of %s: text-poor (%d chars); used full-page OCR (%d chars)",
						page_label,
						path.name,
						len(text),
						len(ocr_text or ""),
					)
				elif not text:
					logger.warning(
						"Skipping page %s of %s: no text and Tesseract unavailable",
						page_label,
						path.name,
					)
					continue
			elif has_images and tesseract_available():
				image_ocr = ocr_pdf_page_images(page, doc)
				if image_ocr:
					combined = merge_text_and_ocr(text, image_ocr)
					parser = "pymupdf+tesseract"
					logger.info(
						"Page %s of %s: text-rich with images; appended image OCR (%d chars)",
						page_label,
						path.name,
						len(image_ocr),
					)
			elif has_images and not tesseract_available():
				logger.warning(
					"Page %s of %s: has images but Tesseract unavailable; using text layer only",
					page_label,
					path.name,
				)

			combined = _normalize_text(combined)
			if len(combined) < _MIN_TEXT_CHARS:
				logger.warning(
					"Skipping page %s of %s: combined text too short (%d chars)",
					page_label,
					path.name,
					len(combined),
				)
				continue

			entries.append(
				{
					"document_id": f"{path.name}_page_{page_label}",
					"text": combined,
					"page": page_label,
					"source_type": "file",
					"source_url": None,
					"filename": path.name,
					"parser": parser,
				}
			)
	return entries


def _convert_pdf_pymupdf(path: Path) -> List[LoadedDocument]:
	"""Backward-compatible alias for the hybrid PDF cascade."""
	return _convert_pdf_hybrid(path)


def _convert_image_tesseract(path: Path) -> List[LoadedDocument]:
	"""OCR a standalone image file with Tesseract."""
	if not tesseract_available():
		msg = tesseract_status_message() or "Tesseract unavailable"
		raise RuntimeError(msg)

	text = _normalize_text(ocr_image_bytes(path.read_bytes()))
	if len(text) < _MIN_TEXT_CHARS:
		logger.warning("Tesseract produced little or no text for %s", path.name)
		return []

	return [
		{
			"document_id": path.name,
			"text": text,
			"page": 1,
			"source_type": "image",
			"source_url": None,
			"filename": path.name,
			"parser": "tesseract",
		}
	]


def _convert_with_docling(path: Path, source_type: SourceType) -> List[LoadedDocument]:
	converter = _get_converter()
	result = converter.convert(str(path))
	return _page_entries_from_docling(
		result.document,
		base_document_id=path.name,
		filename=path.name,
		source_type=source_type,
	)


def _page_entries_from_docling(
	dl_doc,
	*,
	base_document_id: str,
	filename: str,
	source_type: SourceType,
	source_url: Optional[str] = None,
) -> List[LoadedDocument]:
	"""Extract page-level or whole-document text from a Docling document."""
	entries: List[LoadedDocument] = []

	pages = getattr(dl_doc, "pages", None)
	if pages:
		def _page_sort_key(value) -> int:
			try:
				return int(value)
			except (TypeError, ValueError):
				return 0

		for page_no in sorted(pages.keys(), key=_page_sort_key):
			page = pages[page_no]
			export = getattr(page, "export_to_markdown", None) or getattr(page, "export_to_text", None)
			if export is None:
				continue
			text = _normalize_text(export())
			if len(text) < _MIN_TEXT_CHARS:
				logger.warning(
					"Skipping page %s of %s: extracted text too short (%d chars)",
					page_no,
					filename,
					len(text),
				)
				continue
			page_label = int(page_no) + 1 if str(page_no).isdigit() else int(page_no)
			entries.append(
				{
					"document_id": f"{base_document_id}_page_{page_label}",
					"text": text,
					"page": page_label,
					"source_type": source_type,
					"source_url": source_url,
					"filename": filename,
					"parser": "docling",
				}
			)

	if entries:
		return entries

	fallback = _normalize_text(dl_doc.export_to_markdown())
	if len(fallback) < _MIN_TEXT_CHARS:
		fallback = _normalize_text(getattr(dl_doc, "export_to_text", lambda: "")())
	if len(fallback) < _MIN_TEXT_CHARS:
		logger.warning("Docling produced little or no text for %s", filename)
		return []

	return [
		{
			"document_id": base_document_id,
			"text": fallback,
			"page": 1,
			"source_type": source_type,
			"source_url": source_url,
			"filename": filename,
			"parser": "docling",
		}
	]


def convert_path(path: Path) -> List[LoadedDocument]:
	"""Convert a local file path into loaded document entries."""
	path = Path(path)
	if not path.exists() or not path.is_file():
		raise FileNotFoundError(f"File not found: {path}")

	suffix = path.suffix.lower()
	source_type: SourceType = "image" if suffix in IMAGE_EXTENSIONS else "file"

	if suffix == ".txt":
		text = path.read_text(encoding="utf-8", errors="ignore").strip()
		if len(text) < _MIN_TEXT_CHARS:
			return []
		return [
			{
				"document_id": path.name,
				"text": text,
				"page": 1,
				"source_type": "file",
				"source_url": None,
				"filename": path.name,
				"parser": "txt",
			}
		]

	if suffix not in DOCLING_EXTENSIONS:
		raise ValueError(f"Unsupported file type: {suffix}")

	if suffix == ".pdf":
		if docling_available():
			try:
				return _convert_with_docling(path, source_type)
			except Exception as exc:
				logger.warning(
					"Docling failed for %s (%s); falling back to PyMuPDF+Tesseract hybrid",
					path.name,
					exc,
				)
				return _convert_pdf_hybrid(path)
		logger.warning("Docling not installed; using PyMuPDF+Tesseract hybrid for %s", path.name)
		return _convert_pdf_hybrid(path)

	if suffix in IMAGE_EXTENSIONS:
		if docling_available():
			try:
				return _convert_with_docling(path, source_type)
			except Exception as exc:
				logger.warning(
					"Docling failed for image %s (%s); falling back to Tesseract",
					path.name,
					exc,
				)
		if tesseract_available():
			return _convert_image_tesseract(path)
		_require_docling(f"Loading {suffix} files")

	if not docling_available():
		_require_docling(f"Loading {suffix} files")

	return _convert_with_docling(path, source_type)


def convert_source(
	source: str,
	*,
	source_type: SourceType = "file",
	filename: Optional[str] = None,
	source_url: Optional[str] = None,
) -> List[LoadedDocument]:
	"""Convert a local path or URL into loaded document entries."""
	source = (source or "").strip()
	if not source:
		return []

	parsed = urlparse(source)
	is_url = parsed.scheme in {"http", "https"}

	if is_url:
		display_name = filename or parsed.netloc + parsed.path.replace("/", "_") or "web_page"
		_require_docling("Web page ingestion")
		converter = _get_converter()
		result = converter.convert(source)
		return _page_entries_from_docling(
			result.document,
			base_document_id=source,
			filename=display_name,
			source_type="url",
			source_url=source_url or source,
		)

	path = Path(source)
	if not path.exists():
		raise FileNotFoundError(f"Source not found: {source}")

	return convert_path(path)


def is_supported_extension(suffix: str) -> bool:
	suffix = suffix.lower()
	return suffix == ".txt" or suffix in DOCLING_EXTENSIONS
