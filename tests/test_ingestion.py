from __future__ import annotations

import io
import pickle
import sys
from types import SimpleNamespace

import pytest

from faq_chatbot.domain import IndexFingerprint
from faq_chatbot.ingestion import (
    DatasetDownloadError,
    IndexCompatibilityError,
    UnsafeDatasetError,
    build_chroma_index,
    download_dataset,
    evaluate_retrieval,
    load_pickle,
    normalize_faqs,
    read_jsonl,
    write_jsonl,
)


def test_normalization_is_stable_and_deduplicates_questions(tmp_path):
    records = normalize_faqs({" 배송은? ": "  내일   출발합니다. ", "배송은?": "duplicate"})
    assert len(records) == 1
    assert records[0].question == "배송은?"
    first = write_jsonl(records, tmp_path / "faqs.jsonl")
    second = write_jsonl(records, tmp_path / "faqs.jsonl")
    assert first == second
    assert read_jsonl(tmp_path / "faqs.jsonl") == records


def test_pickle_loader_accepts_a_string_dictionary(tmp_path):
    raw = tmp_path / "faqs.pkl"
    raw.write_bytes(pickle.dumps({"배송은?": "내일 출발"}))
    assert load_pickle(raw) == {"배송은?": "내일 출발"}


def test_pickle_loader_rejects_non_string_dictionary(tmp_path):
    raw = tmp_path / "unsafe.pkl"
    raw.write_bytes(pickle.dumps({"question": 3}))
    with pytest.raises(UnsafeDatasetError, match="strings"):
        load_pickle(raw)


def test_download_checks_digest_before_replacing_destination(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "faq_chatbot.ingestion.urlopen", lambda url, timeout: io.BytesIO(b"verified")
    )
    destination = tmp_path / "raw.pkl"
    digest = "1c34f88707b55e6104c4eb20e71ffa3d33e414b71ef689a15fad0640d0ac58cb"
    assert download_dataset(destination, expected_sha256=digest) == destination
    assert destination.read_bytes() == b"verified"
    with pytest.raises(DatasetDownloadError, match="checksum"):
        download_dataset(destination, expected_sha256="0" * 64)
    assert destination.read_bytes() == b"verified"


def test_existing_fingerprint_requires_explicit_rebuild(tmp_path):
    fingerprint = IndexFingerprint(dataset_sha256="old", embedding_model="embed")
    (tmp_path / "fingerprint.json").write_text(fingerprint.model_dump_json(), encoding="utf-8")
    records = normalize_faqs({"배송": "내일"})
    with pytest.raises(IndexCompatibilityError, match="faq-index rebuild"):
        build_chroma_index(
            records, data_sha256="new", chroma_path=tmp_path, embedding_model="embed", api_key="key"
        )


@pytest.mark.asyncio
async def test_retrieval_evaluation_calculates_recall_and_mrr():
    class Retriever:
        async def search(self, query, limit):
            from faq_chatbot.domain import Source

            return [
                Source(id="second", question="두 번째", answer_excerpt="", score=0.9),
                Source(id="first", question="첫 번째", answer_excerpt="", score=0.8),
            ]

    metrics = await evaluate_retrieval(
        [
            {"query": "질문", "relevant_ids": ["first"]},
            {"query": "다른 질문", "expected_question_contains": "두"},
            {"query": "날씨", "expected_no_match": True},
        ],
        Retriever(),
        2,
    )
    assert metrics == {
        "recall_at_k": 1.0,
        "mrr": 0.75,
        "cases": 3.0,
        "out_of_scope_accuracy": 0.0,
    }


def test_builds_chroma_index_with_explicit_rebuild(monkeypatch, tmp_path):
    collections = []

    class Collection:
        def add(self, **kwargs):
            collections.append(kwargs)

    class Client:
        def delete_collection(self, name):
            return None

        def create_collection(self, name, metadata):
            return Collection()

    monkeypatch.setitem(
        sys.modules, "chromadb", SimpleNamespace(PersistentClient=lambda path: Client())
    )
    monkeypatch.setattr(
        "openai.OpenAI",
        lambda **kwargs: SimpleNamespace(
            embeddings=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(data=[SimpleNamespace(embedding=[0.1])])
            )
        ),
    )
    fingerprint = build_chroma_index(
        normalize_faqs({"배송": "내일"}),
        data_sha256="digest",
        chroma_path=tmp_path,
        embedding_model="embed",
        api_key="key",
        rebuild=True,
    )
    assert fingerprint.dataset_sha256 == "digest"
    assert collections[0]["ids"]


def test_normalize_cli_writes_canonical_jsonl(monkeypatch, tmp_path, capsys):
    raw = tmp_path / "raw.pkl"
    output = tmp_path / "faqs.jsonl"
    raw.write_bytes(pickle.dumps({"배송": "내일"}))
    monkeypatch.setattr(sys, "argv", ["faq-index", "normalize", str(raw), "--output", str(output)])
    from faq_chatbot.ingestion import main

    main()
    assert "Wrote 1 FAQ records" in capsys.readouterr().out
    assert read_jsonl(output)[0].question == "배송"


def test_cli_reports_missing_input_without_a_traceback(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", ["faq-index", "normalize", str(tmp_path / "missing.pkl")])
    from faq_chatbot.ingestion import main

    with pytest.raises(SystemExit, match="2"):
        main()
    error = capsys.readouterr().err
    assert error.startswith("error: ")
    assert "missing.pkl" in error
