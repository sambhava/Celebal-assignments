"""Tests for configuration loading and validation."""

import pytest

from rag.config import Config


def test_defaults_present():
    cfg = Config()
    assert cfg.embed_dim == 1024
    assert cfg.embed_model == "embed-multilingual-v3.0"
    assert cfg.chunk_overlap < cfg.chunk_size


def test_overrides_ignore_blanks():
    cfg = Config.load(cohere_api_key="abc", pinecone_api_key="", index_name=None)
    # blank / None overrides must not wipe existing values
    assert cfg.cohere_api_key == "abc"
    assert cfg.index_name == "rag-document-qa"


def test_validate_requires_keys():
    cfg = Config(cohere_api_key="", pinecone_api_key="")
    with pytest.raises(ValueError) as e:
        cfg.validate()
    assert "COHERE_API_KEY" in str(e.value)
    assert "PINECONE_API_KEY" in str(e.value)


def test_validate_rejects_bad_overlap():
    cfg = Config(cohere_api_key="a", pinecone_api_key="b", chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        cfg.validate()


def test_validate_rejects_rerank_gt_topk():
    cfg = Config(cohere_api_key="a", pinecone_api_key="b", top_k=3, rerank_top_n=5)
    with pytest.raises(ValueError):
        cfg.validate()
