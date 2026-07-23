"""Tests for hybrid OCR merge helpers."""
from ingestion.tesseract_ocr import merge_text_and_ocr


def test_merge_appends_image_ocr():
	merged = merge_text_and_ocr("Body text about trees.", "Figure caption: decision stump")
	assert "Body text about trees." in merged
	assert "[Image OCR]" in merged
	assert "decision stump" in merged


def test_merge_skips_duplicate_ocr():
	merged = merge_text_and_ocr("Hello world from the PDF text layer", "Hello world from the PDF text layer")
	assert "[Image OCR]" not in merged
	assert merged == "Hello world from the PDF text layer"


def test_merge_ocr_only_when_no_base():
	assert merge_text_and_ocr("", "Only OCR text") == "Only OCR text"
