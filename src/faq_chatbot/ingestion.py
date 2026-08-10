"""Safe FAQ normalization and reproducible local Chroma indexing."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import pickle
import re
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from urllib.request import urlopen

from .config import get_settings
from .domain import FAQRecord, IndexFingerprint, Retriever
from .providers import ChromaRetriever

SCHEMA_VERSION = 1
MAX_PICKLE_BYTES = 25 * 1024 * 1024
DATASET_COMMIT = "c5580aafb77a24485ccc59ebe0e79a25ad3289a5"
DATASET_URL = (
    "https://raw.githubusercontent.com/eldor-fozilov/faq-chatbot/"
    f"{DATASET_COMMIT}/data/final_result.pkl"
)
DATASET_SHA256 = "93f2a92172f253b28ff94b6b8fd993b8cec1478e286ce7d6296098bd73a9e52b"
DEFAULT_RAW_DATASET = Path("data/raw/smartstore_faq.pkl")
DEFAULT_NORMALIZED_DATASET = Path("data/faqs.jsonl")


class UnsafeDatasetError(ValueError):
    """The raw pickle does not match the deliberately narrow accepted format."""


class DatasetDownloadError(ValueError):
    """The pinned upstream dataset could not be verified."""


class IndexCompatibilityError(ValueError):
    """An existing vector index has a different immutable fingerprint."""


class MissingOpenAIKeyError(ValueError):
    """Indexing is intentionally unavailable until a local OpenAI key exists."""


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        raise UnsafeDatasetError(f"pickle globals are not allowed: {module}.{name}")


def load_pickle(path: Path, *, maximum_bytes: int = MAX_PICKLE_BYTES) -> dict[str, str]:
    if path.stat().st_size > maximum_bytes:
        raise UnsafeDatasetError(f"dataset exceeds maximum allowed size of {maximum_bytes} bytes")
    with path.open("rb") as file:
        value = RestrictedUnpickler(file).load()
    if not isinstance(value, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in value.items()
    ):
        raise UnsafeDatasetError("dataset must be a dictionary mapping strings to strings")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(
    destination: Path, *, url: str = DATASET_URL, expected_sha256: str = DATASET_SHA256
) -> Path:
    """Download the exact reviewed source, enforcing its size and digest before replacing a file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(url, timeout=30) as response, temporary.open("wb") as file:  # noqa: S310
            total = 0
            while chunk := response.read(64 * 1024):
                total += len(chunk)
                if total > MAX_PICKLE_BYTES:
                    raise DatasetDownloadError(
                        f"dataset exceeds maximum allowed size of {MAX_PICKLE_BYTES} bytes"
                    )
                file.write(chunk)
        actual = file_sha256(temporary)
        if actual != expected_sha256:
            raise DatasetDownloadError("dataset checksum does not match the pinned source")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def normalize_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return re.sub(r"(?:\s*더보기\s*)+$", "", value).strip()


def content_hash(question: str, answer: str) -> str:
    return hashlib.sha256(f"{question}\n{answer}".encode()).hexdigest()


def normalize_faqs(raw: dict[str, str]) -> list[FAQRecord]:
    records: list[FAQRecord] = []
    seen_questions: set[str] = set()
    for question, answer in raw.items():
        normalized_question, normalized_answer = normalize_text(question), normalize_text(answer)
        if (
            not normalized_question
            or not normalized_answer
            or normalized_question in seen_questions
        ):
            continue
        seen_questions.add(normalized_question)
        digest = content_hash(normalized_question, normalized_answer)
        records.append(
            FAQRecord(
                id=digest[:16],
                question=normalized_question,
                answer=normalized_answer,
                content_hash=digest,
            )
        )
    if not records:
        raise UnsafeDatasetError("dataset contains no valid FAQ entries")
    return sorted(records, key=lambda record: record.id)


def write_jsonl(records: Iterable[FAQRecord], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = "".join(record.model_dump_json() + "\n" for record in records)
    path.write_text(serialized, encoding="utf-8")
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[FAQRecord]:
    return [
        FAQRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def build_chroma_index(
    records: list[FAQRecord],
    *,
    data_sha256: str,
    chroma_path: Path,
    embedding_model: str,
    api_key: str,
    rebuild: bool = False,
) -> IndexFingerprint:
    """Build a fingerprinted cosine index; replacing one always requires an explicit rebuild."""
    import chromadb
    from openai import OpenAI

    fingerprint = IndexFingerprint(
        schema_version=SCHEMA_VERSION,
        dataset_sha256=data_sha256,
        embedding_model=embedding_model,
    )
    fingerprint_file = chroma_path / "fingerprint.json"
    if fingerprint_file.exists() and not rebuild:
        existing = IndexFingerprint.model_validate_json(
            fingerprint_file.read_text(encoding="utf-8")
        )
        if existing == fingerprint and (chroma_path / "chroma.sqlite3").is_file():
            return fingerprint
        raise IndexCompatibilityError(
            "FAQ index fingerprint differs from the requested dataset or embedding model; run `faq-index rebuild`."
        )
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    with suppress(Exception):  # Chroma has no collection on a first build.
        client.delete_collection("faqs")
    collection = client.create_collection("faqs", metadata={"hnsw:space": "cosine"})
    embedding_client = OpenAI(api_key=api_key).embeddings
    for offset in range(0, len(records), 100):
        batch = records[offset : offset + 100]
        embeddings = embedding_client.create(
            model=embedding_model, input=[record.question for record in batch]
        ).data
        collection.add(
            ids=[record.id for record in batch],
            documents=[record.question for record in batch],
            embeddings=[item.embedding for item in embeddings],
            metadatas=[
                {
                    "answer": record.answer,
                    "content_hash": record.content_hash,
                    "source": record.source,
                }
                for record in batch
            ],
        )
    fingerprint_file.write_text(fingerprint.model_dump_json(indent=2), encoding="utf-8")
    return fingerprint


def build_index_from_jsonl(dataset: Path, *, rebuild: bool = False) -> IndexFingerprint:
    """Build or explicitly rebuild the local index using environment-backed application settings."""
    settings = get_settings()
    if not settings.is_configured:
        raise MissingOpenAIKeyError(
            "OPENAI_API_KEY is required to create embeddings. Add it to your local .env."
        )
    records = read_jsonl(dataset)
    return build_chroma_index(
        records,
        data_sha256=file_sha256(dataset),
        chroma_path=settings.chroma_path,
        embedding_model=settings.openai_embedding_model,
        api_key=settings.openai_api_key.get_secret_value(),  # type: ignore[union-attr]
        rebuild=rebuild,
    )


async def evaluate_retrieval(
    cases: list[dict[str, object]],
    retriever: Retriever,
    limit: int,
    similarity_threshold: float = 0,
) -> dict[str, float]:
    """Calculate Recall@k and reciprocal rank for labelled retrieval cases."""
    if not cases:
        raise ValueError("evaluation file must contain at least one labelled case")
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    out_of_scope: list[float] = []
    for case in cases:
        query = case.get("query")
        relevant_ids = case.get("relevant_ids", [])
        expected_question = case.get("expected_question")
        expected_contains = case.get("expected_question_contains")
        expected_no_match = case.get("expected_no_match", False)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("every evaluation case needs a non-empty query")
        if not isinstance(relevant_ids, list) or not all(
            isinstance(item, str) for item in relevant_ids
        ):
            raise ValueError("relevant_ids must be a list of strings")
        if (
            expected_no_match is not True
            and not relevant_ids
            and not isinstance(expected_question, str)
            and not isinstance(expected_contains, str)
        ):
            raise ValueError(
                "every retrieval case needs relevant_ids, expected_question, or expected_question_contains"
            )
        results = [
            source
            for source in await retriever.search(query, limit)
            if source.score >= similarity_threshold
        ]
        if expected_no_match is True:
            out_of_scope.append(float(not results))
            continue
        relevant = set(relevant_ids)
        ranks = [
            rank
            for rank, source in enumerate(results, start=1)
            if source.id in relevant
            or source.question == expected_question
            or (isinstance(expected_contains, str) and expected_contains in source.question)
        ]
        recalls.append(float(bool(ranks)))
        reciprocal_ranks.append(1 / ranks[0] if ranks else 0)
    metrics = {
        "recall_at_k": sum(recalls) / len(recalls) if recalls else 0,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0,
        "cases": float(len(cases)),
    }
    if out_of_scope:
        metrics["out_of_scope_accuracy"] = sum(out_of_scope) / len(out_of_scope)
    return metrics


def evaluate_from_file(path: Path) -> dict[str, float]:
    """Run labelled retrieval evaluation against the configured real local index."""
    settings = get_settings()
    if not settings.is_configured:
        raise MissingOpenAIKeyError(
            "OPENAI_API_KEY is required to evaluate retrieval. Add it to your local .env."
        )
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        raise ValueError("evaluation file must be a JSON list of labelled cases")
    retriever = ChromaRetriever(
        path=str(settings.chroma_path),
        api_key=settings.openai_api_key.get_secret_value(),  # type: ignore[union-attr]
        embedding_model=settings.openai_embedding_model,
    )
    return asyncio.run(
        evaluate_retrieval(cases, retriever, settings.top_k, settings.similarity_threshold)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download, normalize, and explicitly index the pinned SmartStore FAQ."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="download and verify the pinned raw pickle")
    download.add_argument("--output", type=Path, default=DEFAULT_RAW_DATASET)
    normalize = commands.add_parser(
        "normalize", help="convert a verified raw pickle to canonical JSONL"
    )
    normalize.add_argument("pickle", type=Path, nargs="?", default=DEFAULT_RAW_DATASET)
    normalize.add_argument("--output", type=Path, default=DEFAULT_NORMALIZED_DATASET)
    for name in ("build", "rebuild"):
        index = commands.add_parser(
            name, help=f"{name} the local Chroma index from normalized JSONL"
        )
        index.add_argument("--input", type=Path, default=DEFAULT_NORMALIZED_DATASET)
    evaluate = commands.add_parser(
        "evaluate", help="calculate Recall@k and MRR from labelled retrieval cases"
    )
    evaluate.add_argument("eval_json", type=Path)
    arguments = parser.parse_args()
    try:
        if arguments.command == "download":
            print(f"Downloaded verified dataset to {download_dataset(arguments.output)}")
        elif arguments.command == "normalize":
            records = normalize_faqs(load_pickle(arguments.pickle))
            digest = write_jsonl(records, arguments.output)
            print(f"Wrote {len(records)} FAQ records to {arguments.output} (sha256={digest})")
        elif arguments.command in {"build", "rebuild"}:
            fingerprint = build_index_from_jsonl(
                arguments.input, rebuild=arguments.command == "rebuild"
            )
            print(f"Built FAQ index at {get_settings().chroma_path} ({fingerprint.dataset_sha256})")
        else:
            metrics = evaluate_from_file(arguments.eval_json)
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
    except (
        FileNotFoundError,
        DatasetDownloadError,
        UnsafeDatasetError,
        IndexCompatibilityError,
        MissingOpenAIKeyError,
        ValueError,
    ) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
