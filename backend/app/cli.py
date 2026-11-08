import argparse, hashlib, json
from pathlib import Path
from app.core.config import get_settings

def verify(data_dir: Path):
    processed = data_dir / "processed"; artifacts = data_dir / "artifacts" / "retrieval"
    required = [processed/"pokemon_rag_documents.jsonl", processed/"pokemon_rag_records.jsonl", artifacts/"tfidf_vectorizer.joblib", artifacts/"tfidf_matrix.npz", artifacts/"retrieval_manifest.json"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing: raise SystemExit("Artefatos ausentes:\n- " + "\n- ".join(missing))
    docs = [json.loads(x) for x in (processed/"pokemon_rag_documents.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    records = [json.loads(x) for x in (processed/"pokemon_rag_records.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    ids = [d["document_id"] for d in docs]
    if len(ids) != len(set(ids)) or not {int(r["id"]) for r in records} == set(range(1,152)):
        raise SystemExit("Integridade inválida: IDs duplicados ou corpus incompleto")
    print(json.dumps({"status":"ok", "pokemon_count":len(records), "document_count":len(docs)}, ensure_ascii=False))

def main():
    parser=argparse.ArgumentParser(description="Pipeline da Pokédex Inteligente")
    parser.add_argument("command", choices=["prepare","index","verify"])
    args=parser.parse_args(); data_dir=get_settings().rag_data_dir
    if args.command == "verify": verify(data_dir)
    elif args.command == "prepare": print("Prepare requer os JSON brutos produzidos pelo CORPUS_POKEMON.ipynb; nenhum dado foi fabricado.")
    else: print("Index requer os documentos preparados; execute o notebook 02 ou forneça um corpus processado.")

if __name__ == "__main__": main()

