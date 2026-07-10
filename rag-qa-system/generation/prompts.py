"""Shared generation instructions for local and Groq answer models."""

ANSWER_INSTRUCTIONS = (
	"Answer comprehensively using the provided context.\n\n"
	"If the context contains sufficient information:\n"
	"- explain concepts clearly\n"
	"- include important details\n"
	"- provide examples when relevant\n\n"
	"Do not omit useful information from the context.\n"
	"Use ONLY the provided context. Do not invent facts.\n"
	"If the context is insufficient to answer the question, say you do not know "
	"based on the uploaded document.\n"
	"Do not copy the context verbatim."
)

GROQ_ANSWER_INSTRUCTIONS = (
	f"{ANSWER_INSTRUCTIONS}\n"
	"Include citations in the answer like '(Source: filename_page_N)' when possible.\n"
	"Cite the best 1-2 sources max.\n"
	"Do not repeat the question."
)
