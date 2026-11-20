import json

import joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from app.rag.retrieval import Retriever


def test_retriever_loads_the_canonical_application_data_layout(tmp_path):
    documents = tmp_path / "rag" / "documents"
    artifacts = tmp_path / "artifacts" / "retrieval"
    documents.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    doc = {
        "document_id": "001_profile",
        "pokemon_id": 1,
        "pokemon_name": "Bulbasaur",
        "section": "profile",
        "content": "Bulbasaur é do tipo planta e venenoso.",
        "retrieval_text": "bulbasaur tipo planta venenoso",
    }
    (documents / "pokemon_rag_documents.jsonl").write_text(json.dumps(doc) + "\n", encoding="utf-8")
    (documents / "pokemon_rag_records.jsonl").write_text(json.dumps({"id": 1, "name": "Bulbasaur", "api_name": "bulbasaur"}) + "\n", encoding="utf-8")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2)).fit([doc["retrieval_text"]])
    joblib.dump(vectorizer, artifacts / "tfidf_vectorizer.joblib")
    sparse.save_npz(artifacts / "tfidf_matrix.npz", vectorizer.transform([doc["retrieval_text"]]))
    (artifacts / "query_normalization_rules.json").write_text(json.dumps({"rules": []}), encoding="utf-8")
    (artifacts / "retrieval_manifest.json").write_text(json.dumps({"document_ids": [doc["document_id"]]}), encoding="utf-8")

    retriever = Retriever(tmp_path)

    assert retriever.search("Bulbasaur", top_k=1)[0]["document_id"] == "001_profile"
