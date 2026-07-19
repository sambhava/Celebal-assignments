"""Version-locked error-recovery retrieval (RAG) for API_MISUSE only.

The retrieval step is deliberately narrow (TECHNICAL_DESIGN.md §4): it fires
ONLY when the error classifier has positively identified an ``API_MISUSE`` error,
never on data problems, so it can never loop on a genuinely bad file. Its primary
source is the version-locked deprecation cheatsheet (cheap, high hit-rate for a
weak model); an optional FAISS index over pinned pandas docs handles the long
tail and is skipped gracefully when not built.
"""
