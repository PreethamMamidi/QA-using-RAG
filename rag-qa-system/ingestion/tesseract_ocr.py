"""Tesseract OCR helpers for scanned pages and embedded PDF images."""
from __future__ import annotations

import io
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

_TESSERACT_IMPORT_ERROR: Exception | None = None
_MIN_OCR_CHARS = 10


def tesseract_available() -> bool:
	"""Return True when pytesseract and the Tesseract binary are usable."""
	global _TESSERACT_IMPORT_ERROR
	try:
		import pytesseract
		from PIL import Image  # noqa: F401

		# Probe the binary once; raises if tesseract is not on PATH.
		pytesseract.get_tesseract_version()
		return True
	except Exception as exc:  # ImportError, TesseractNotFoundError, etc.
		_TESSERACT_IMPORT_ERROR = exc
		return False


def ocr_image_bytes(image_bytes: bytes, *, lang: str = "eng") -> str:
	"""OCR raw image bytes with Tesseract."""
	if not image_bytes:
		return ""
	if not tesseract_available():
		return ""

	import pytesseract
	from PIL import Image

	with Image.open(io.BytesIO(image_bytes)) as image:
		# Normalize unusual modes (CMYK, P, etc.) for better OCR.
		if image.mode not in ("RGB", "L"):
			image = image.convert("RGB")
		text = pytesseract.image_to_string(image, lang=lang) or ""
	return text.strip()


def ocr_pil_image(image, *, lang: str = "eng") -> str:
	"""OCR a PIL image with Tesseract."""
	if image is None:
		return ""
	if not tesseract_available():
		return ""

	import pytesseract

	if getattr(image, "mode", None) not in ("RGB", "L"):
		image = image.convert("RGB")
	return (pytesseract.image_to_string(image, lang=lang) or "").strip()


def ocr_pdf_page_render(page, *, zoom: float = 2.0, lang: str = "eng") -> str:
	"""Render a full PDF page to an image and OCR it (for scanned pages)."""
	if not tesseract_available():
		return ""

	import fitz

	matrix = fitz.Matrix(zoom, zoom)
	pix = page.get_pixmap(matrix=matrix, alpha=False)
	return ocr_image_bytes(pix.tobytes("png"), lang=lang)


def ocr_pdf_page_images(page, doc, *, lang: str = "eng", min_side: int = 40) -> str:
	"""OCR embedded images on a PDF page and return concatenated text.

	Parameters
	----------
	page :
		PyMuPDF page object.
	doc :
		Parent PyMuPDF document (needed for extract_image).
	lang :
		Tesseract language code.
	min_side :
		Skip tiny images (icons/decorations) below this pixel size.
	"""
	if not tesseract_available():
		return ""

	parts: List[str] = []
	seen_xrefs: set[int] = set()

	for img in page.get_images(full=True) or []:
		xref = int(img[0])
		if xref in seen_xrefs:
			continue
		seen_xrefs.add(xref)

		try:
			extracted = doc.extract_image(xref)
		except Exception as exc:
			logger.debug("Failed to extract image xref=%s: %s", xref, exc)
			continue

		image_bytes = extracted.get("image") or b""
		width = int(extracted.get("width") or 0)
		height = int(extracted.get("height") or 0)
		if min(width, height) < min_side and width and height:
			continue

		text = ocr_image_bytes(image_bytes, lang=lang)
		if len(text) >= _MIN_OCR_CHARS:
			parts.append(text)

	return "\n\n".join(parts).strip()


def merge_text_and_ocr(base_text: str, ocr_text: str) -> str:
	"""Append OCR text when it adds useful content not already present."""
	base = (base_text or "").strip()
	ocr = (ocr_text or "").strip()
	if not ocr:
		return base
	if not base:
		return ocr

	# Avoid duplicating OCR that largely repeats the text layer.
	normalized_base = " ".join(base.lower().split())
	normalized_ocr = " ".join(ocr.lower().split())
	if normalized_ocr and normalized_ocr in normalized_base:
		return base
	if len(normalized_ocr) < 20 and normalized_ocr in normalized_base:
		return base

	return f"{base}\n\n[Image OCR]\n{ocr}".strip()


def tesseract_status_message() -> Optional[str]:
	"""Human-readable hint when Tesseract is unavailable."""
	if tesseract_available():
		return None
	detail = str(_TESSERACT_IMPORT_ERROR) if _TESSERACT_IMPORT_ERROR else "unknown error"
	return (
		"Tesseract OCR is unavailable. Install Tesseract and: pip install pytesseract pillow. "
		f"Detail: {detail}"
	)
