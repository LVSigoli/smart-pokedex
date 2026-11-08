from collections import Counter
import json
import re
import unicodedata
from pathlib import Path
import joblib
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from app.core.exceptions import ArtifactError

def normalize_plain(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or "").casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("-", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text)).strip()

class Retriever:
    def __init__(self, data_dir: Path):
        root = Path(data_dir)
        doc_path = root / "processed" / "pokemon_rag_documents.jsonl"
        record_path = root / "processed" / "pokemon_rag_records.jsonl"
        artifact = root / "artifacts" / "retrieval"
        if not doc_path.is_file() or not record_path.is_file():
            raise ArtifactError(f"Artefatos ausentes: {doc_path} e {record_path}")
        self.documents = self._jsonl(doc_path)
        self.records = self._jsonl(record_path)
        self.documents.sort(key=lambda x: (int(x["pokemon_id"]), x["section"], x["document_id"]))
        self.doc_ids = [d["document_id"] for d in self.documents]
        if len(self.doc_ids) != len(set(self.doc_ids)):
            raise ArtifactError("Documentos duplicados")
        self.lookup = {d["document_id"]: d for d in self.documents}
        self.by_id = {int(r["id"]): r for r in self.records}
        try:
            self.vectorizer = joblib.load(artifact / "tfidf_vectorizer.joblib")
            self.matrix = sparse.load_npz(artifact / "tfidf_matrix.npz")
        except Exception as exc:
            raise ArtifactError("Índice TF-IDF ausente ou incompatível") from exc
        if self.matrix.shape[0] != len(self.documents):
            raise ArtifactError("Matriz TF-IDF não corresponde aos documentos")
        rules_path = artifact / "query_normalization_rules.json"
        rules = json.loads(rules_path.read_text(encoding="utf-8")) if rules_path.is_file() else {"rules": []}
        self.rules = sorted([(x["from"], x["to"]) for x in rules.get("rules", [])], key=lambda x: (-len(x[0].split()), -len(x[0])))

    @staticmethod
    def _jsonl(path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def normalize_query(self, text):
        value = normalize_plain(text)
        for source, target in self.rules:
            value = re.sub(r"(?<!\w)" + re.escape(source) + r"(?!\w)", target, value)
        return re.sub(r"\s+", " ", value).strip()

    def search(self, question, top_k=5, max_per_pokemon=None):
        if top_k < 1:
            raise ValueError("top_k deve ser positivo")
        query = self.normalize_query(question)
        if not query:
            return []
        vector = self.vectorizer.transform([query])
        if not vector.nnz:
            return []
        scores = cosine_similarity(vector, self.matrix).ravel()
        indices = [i for i in np.flatnonzero(scores > 1e-12)]
        indices.sort(key=lambda i: (-float(scores[i]), self.doc_ids[i]))
        counts, result = Counter(), []
        for i in indices:
            doc = self.documents[i]
            pid = int(doc["pokemon_id"])
            if max_per_pokemon is not None and counts[pid] >= max_per_pokemon:
                continue
            counts[pid] += 1
            result.append({"rank": len(result) + 1, "document_id": doc["document_id"], "pokemon_id": pid,
                           "pokemon_name": doc["pokemon_name"], "section": doc["section"], "score": float(scores[i]),
                           "factual_scope": doc.get("factual_scope", ""), "source_file": doc.get("source_file", ""),
                           "content": doc["content"]})
            if len(result) >= top_k:
                break
        return result

