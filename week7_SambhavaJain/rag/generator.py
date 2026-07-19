"""Grounded answer generation with Cohere Chat.

The retrieved chunks are passed to Cohere Chat as structured ``documents``.
Cohere then generates an answer *grounded* in those documents and returns
citations mapping spans of the answer back to the source chunks — exactly the
"answers grounded in actual data" goal from the assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SYSTEM_PREAMBLE = (
    "You are a helpful assistant that answers questions using only the "
    "provided documents. If the answer is not contained in the documents, "
    "say you could not find it in the provided documents. Be concise, "
    "accurate, and cite the sources you used."
)

SUMMARY_PROMPT = (
    "Write a clear overview of the provided document(s) for someone who has "
    "not read them. Structure your response as:\n"
    "1. A 2-3 sentence summary of what the document is about.\n"
    "2. A short bulleted list of the main topics or key points.\n"
    "Base everything strictly on the provided documents."
)


@dataclass
class Answer:
    text: str
    sources: list[dict] = field(default_factory=list)  # the chunks provided as context
    citations: list[dict] = field(default_factory=list)  # answer-span -> source mapping


class Generator:
    def __init__(self, client, chat_model: str):
        self._client = client
        self._model = chat_model

    def generate(self, question: str, chunks: list[dict]) -> Answer:
        if not chunks:
            return Answer(
                text="I could not find anything relevant in the provided documents.",
                sources=[],
                citations=[],
            )

        documents = [
            {
                "id": f"doc_{i}",
                "data": {
                    "text": c["text"],
                    "source": str(c.get("source", "unknown")),
                    "page": str(c.get("page", 0)),
                },
            }
            for i, c in enumerate(chunks)
        ]

        resp = self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PREAMBLE},
                {"role": "user", "content": question},
            ],
            documents=documents,
        )

        return Answer(
            text=self._extract_text(resp),
            sources=chunks,
            citations=self._extract_citations(resp),
        )

    def summarize(self, chunks: list[dict]) -> str:
        """Produce a grounded overview of the given document chunks."""
        if not chunks:
            return "No document content was found to summarize."

        documents = [
            {
                "id": f"doc_{i}",
                "data": {"text": c["text"], "source": str(c.get("source", "unknown"))},
            }
            for i, c in enumerate(chunks)
        ]
        resp = self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PREAMBLE},
                {"role": "user", "content": SUMMARY_PROMPT},
            ],
            documents=documents,
        )
        return self._extract_text(resp)

    # --- response parsing (defensive: the SDK shape varies by version) -------

    @staticmethod
    def _extract_text(resp) -> str:
        try:
            content = resp.message.content
            if content:
                return "".join(getattr(part, "text", "") for part in content).strip()
        except Exception:
            pass
        return str(getattr(resp, "text", "")).strip()

    @staticmethod
    def _extract_citations(resp) -> list[dict]:
        citations: list[dict] = []
        try:
            for cit in resp.message.citations or []:
                citations.append(
                    {
                        "text": getattr(cit, "text", ""),
                        "start": getattr(cit, "start", None),
                        "end": getattr(cit, "end", None),
                        "sources": [
                            getattr(s, "id", "") for s in getattr(cit, "sources", []) or []
                        ],
                    }
                )
        except Exception:
            pass
        return citations
