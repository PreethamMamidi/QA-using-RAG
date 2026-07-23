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
	"When citing sources in the answer, use the exact source labels from the context headers "
	"(for example: 'Computer Networks.pdf (Page 12)').\n"
	"Cite the best 1-3 distinct sources max.\n"
	"Do not repeat the question."
)
