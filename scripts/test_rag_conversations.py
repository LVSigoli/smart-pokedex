"""Teste conversacional abrangente do RAG da Pokédex.

O script usa o mesmo Retriever, RAGEngine e SessionManager da API. Ele não
coleta dados, não ajusta TF-IDF e não altera nenhum artefato persistido.

Exemplos, executados na raiz do projeto:

    python scripts/test_rag_conversations.py
    python scripts/test_rag_conversations.py --max-pokemon 10 --pair-limit 3
    python scripts/test_rag_conversations.py --generate --max-pokemon 3 --pair-limit 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.core.config import Settings  # noqa: E402
from app.rag.engine import RAGEngine  # noqa: E402
from app.rag.retrieval import Retriever  # noqa: E402
from app.sessions.manager import SessionManager  # noqa: E402


REFUSAL_STATUSES = {"insufficient_evidence", "out_of_scope"}
EXPECTED_CLARIFICATION = "clarification_required"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-pokemon",
        type=int,
        default=151,
        help="Quantidade máxima de Pokémon usada no bloco individual (padrão: 151).",
    )
    parser.add_argument(
        "--pair-limit",
        type=int,
        default=None,
        help="Quantidade máxima de pares usada no bloco comparativo (padrão: todos os pares adjacentes).",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Também executa a geração Gemini; exige GEMINI_API_KEY e pode consumir cota.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Diretório data alternativo; por padrão usa data/ na raiz do projeto.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "evaluation" / "rag_conversation_test.jsonl",
        help="Arquivo JSONL de saída.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "data" / "evaluation" / "rag_conversation_test_summary.json",
        help="Resumo JSON de saída.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Preserva código 0 mesmo quando o relatório registrar falhas.",
    )
    return parser.parse_args()


def load_runtime(data_dir: Path) -> tuple[RAGEngine, SessionManager, list[dict[str, Any]]]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Diretório de dados não encontrado: {data_dir}")
    settings = Settings(rag_data_dir=data_dir.resolve())
    retriever = Retriever(settings.rag_data_dir)
    engine = RAGEngine(retriever, settings)
    manager = SessionManager(engine, settings.session_ttl_seconds)
    records = sorted(retriever.records, key=lambda item: int(item["id"]))
    if not records:
        raise RuntimeError("Nenhum Pokémon foi encontrado nos artefatos persistidos.")
    return engine, manager, records


def create_session(manager: SessionManager) -> UUID:
    return manager.create().session_id


def source_ids(result: dict[str, Any]) -> set[int]:
    return {int(item["pokemon_id"]) for item in result.get("source_documents", [])}


def active_names(engine: RAGEngine, active_ids: list[int]) -> list[str]:
    return [
        engine.retriever.by_id.get(pokemon_id, {}).get("name", str(pokemon_id))
        for pokemon_id in active_ids
    ]


def execute_turn(
    *,
    engine: RAGEngine,
    manager: SessionManager,
    session_id: UUID,
    scenario: str,
    turn_number: int,
    message: str,
    expected_ids: set[int] | None,
    expect_clarification: bool,
    generate: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    session = manager.get(session_id)
    before = list(session.active_ids)
    result = manager.ask(session_id, message, generate=generate)
    after = list(manager.get(session_id).active_ids)
    observed_ids = source_ids(result)
    failures: list[str] = []

    if expect_clarification:
        if result.get("status") != EXPECTED_CLARIFICATION:
            failures.append(
                f"status esperado {EXPECTED_CLARIFICATION}, recebido {result.get('status')}"
            )
    elif not result.get("answer"):
        failures.append("resposta vazia")

    if expected_ids and result.get("status") not in REFUSAL_STATUSES | {EXPECTED_CLARIFICATION}:
        missing = expected_ids - observed_ids
        if missing:
            failures.append(f"fontes ausentes para Pokémon {sorted(missing)}")

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario,
        "turn": turn_number,
        "session_id": str(session_id),
        "message": message,
        "generate": generate,
        "active_before": before,
        "active_before_names": active_names(engine, before),
        "active_after": after,
        "active_after_names": active_names(engine, after),
        "status": result.get("status"),
        "answer_mode": result.get("answer_mode"),
        "answer": result.get("answer"),
        "expected_pokemon_ids": sorted(expected_ids or set()),
        "source_pokemon_ids": sorted(observed_ids),
        "source_documents": result.get("source_documents", []),
        "gate_reason": result.get("gate_reason"),
        "generation_error": result.get("generation_error"),
        "failures": failures,
    }
    return row, ({"scenario": scenario, "turn": turn_number, "failures": failures} if failures else None)


def run_individual_conversations(
    engine: RAGEngine,
    manager: SessionManager,
    records: list[dict[str, Any]],
    generate: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in records:
        pokemon_id = int(record["id"])
        name = record.get("name") or record.get("api_name") or str(pokemon_id)
        session_id = create_session(manager)
        turns = [
            (f"Fale sobre o Pokémon {name}", {pokemon_id}, False),
            (f"Qual é o ataque base de {name}?", {pokemon_id}, False),
            (f"Onde encontro {name}?", {pokemon_id}, False),
            ("Como evolui ele", {pokemon_id}, False),
            ("Onde encontro ele", {pokemon_id}, False),
        ]
        for number, (message, expected_ids, clarification) in enumerate(turns, 1):
            row, failure = execute_turn(
                engine=engine, manager=manager, session_id=session_id,
                scenario="individual", turn_number=number, message=message,
                expected_ids=expected_ids, expect_clarification=clarification,
                generate=generate,
            )
            row["pokemon_id"] = pokemon_id
            row["pokemon_name"] = name
            rows.append(row)
            if failure:
                failures.append({**failure, "pokemon_id": pokemon_id, "pokemon_name": name})
    return rows, failures


def run_pair_conversations(
    engine: RAGEngine,
    manager: SessionManager,
    records: list[dict[str, Any]],
    generate: bool,
    pair_limit: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    pairs = list(zip(records, records[1:]))
    if pair_limit is not None:
        pairs = pairs[: max(pair_limit, 0)]

    for pair_number, (first, second) in enumerate(pairs, 1):
        first_id, second_id = int(first["id"]), int(second["id"])
        first_name = first.get("name") or first.get("api_name") or str(first_id)
        second_name = second.get("name") or second.get("api_name") or str(second_id)
        both = {first_id, second_id}
        session_id = create_session(manager)
        turns = [
            (f"Quais são os tipos de {first_name} e {second_name}?", both, False),
            ("Qual dos dois tem maior velocidade?", both, False),
            ("Onde encontro eles", both, False),
            (f"Onde encontro {first_name} e {second_name}?", both, False),
            (f"Agora fale sobre {second_name}", {second_id}, False),
            ("Onde encontro ele", {second_id}, False),
        ]
        for number, (message, expected_ids, clarification) in enumerate(turns, 1):
            row, failure = execute_turn(
                engine=engine, manager=manager, session_id=session_id,
                scenario="pair", turn_number=number, message=message,
                expected_ids=expected_ids, expect_clarification=clarification,
                generate=generate,
            )
            row.update({
                "pair_number": pair_number,
                "pokemon_ids": sorted(both),
                "pokemon_names": [first_name, second_name],
            })
            rows.append(row)
            if failure:
                failures.append({**failure, "pair_number": pair_number, "pokemon_ids": sorted(both)})
    return rows, failures


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.max_pokemon < 1:
        raise SystemExit("--max-pokemon deve ser positivo.")
    data_dir = (args.data_dir or ROOT / "data").resolve()
    engine, manager, all_records = load_runtime(data_dir)
    records = all_records[: args.max_pokemon]

    print(f"RAG conversacional: {len(records)} Pokémon | geração local: {args.generate}")
    print("Recuperador: TF-IDF persistido; sessões: SessionManager da API")
    individual_rows, individual_failures = run_individual_conversations(
        engine, manager, records, args.generate
    )
    pair_rows, pair_failures = run_pair_conversations(
        engine, manager, records, args.generate, args.pair_limit
    )
    rows = individual_rows + pair_rows
    failures = individual_failures + pair_failures
    write_jsonl(args.output.resolve(), rows)

    summary = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "pokemon_count": len(records),
        "pair_count": max(len(records) - 1, 0) if args.pair_limit is None else min(max(args.pair_limit, 0), max(len(records) - 1, 0)),
        "turn_count": len(rows),
        "generate": args.generate,
        "status_counts": dict(Counter(str(row.get("status")) for row in rows)),
        "answer_mode_counts": dict(Counter(str(row.get("answer_mode")) for row in rows)),
        "failure_count": len(failures),
        "failures": failures[:100],
        "output": str(args.output.resolve()),
        "summary_output": str(args.summary_output.resolve()),
    }
    write_json(args.summary_output.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures and not args.allow_failures:
        print("Falhas encontradas. Use --allow-failures para apenas gerar o relatório.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
