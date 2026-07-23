import nltk
import pytest


@pytest.fixture(scope="session", autouse=True)
def ensure_nltk_resources():
	for resource in ("punkt", "punkt_tab"):
		try:
			nltk.data.find(f"tokenizers/{resource}")
		except LookupError:
			nltk.download(resource, quiet=True)
