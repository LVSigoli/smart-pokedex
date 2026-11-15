from types import SimpleNamespace
from app.sessions.manager import SessionManager

class Engine:
    def answer(self, message, generate=False): return {"status":"retrieval_only","answer":message}

def test_sessions_are_isolated():
    manager=SessionManager(Engine())
    first, second = manager.create(), manager.create()
    assert first.session_id != second.session_id
    assert manager.ask(first.session_id, "a")["answer"] == "a"
    assert manager.get(second.session_id).turns == []

class ContextualEngine:
    retriever = SimpleNamespace(by_id={4: {"name": "Charmander"}, 133: {"name": "Eevee"}, 25: {"name": "Pikachu"}})
    def mentioned_ids(self, message):
        lowered = message.casefold()
        return [pokemon_id for pokemon_id, name in ((4, "charmander"), (133, "eevee"), (25, "pikachu")) if name in lowered]
    def answer(self, message, generate=False):
        ids = self.mentioned_ids(message)
        return {"status": "retrieval_only", "answer": message, "source_documents": [{"pokemon_id": pokemon_id} for pokemon_id in ids], "supporting_excerpts": []}

def test_location_followup_keeps_active_pokemon():
    manager = SessionManager(ContextualEngine())
    session = manager.create()
    manager.ask(session.session_id, "Fale sobre Eevee e Pikachu")
    result = manager.ask(session.session_id, "E onde eu encontro eles?")
    assert result["answer"].startswith("E onde eu encontro eles? sobre ")
    assert "Eevee" in result["answer"] and "Pikachu" in result["answer"]

def test_location_followup_keeps_single_active_pokemon():
    manager = SessionManager(ContextualEngine())
    session = manager.create()
    manager.ask(session.session_id, "Fale sobre Eevee")
    result = manager.ask(session.session_id, "Onde eu encontro ele?")
    assert result["answer"] == "Onde eu encontro ele? sobre Eevee"

def test_evolution_followup_without_species_keeps_active_pokemon():
    manager = SessionManager(ContextualEngine())
    session = manager.create()
    manager.ask(session.session_id, "Como evoluir Charmander?")
    result = manager.ask(session.session_id, "Mas como evoluo?")
    assert result["answer"] == "Mas como evoluo? sobre Charmander"


class ComparisonContextualEngine(ContextualEngine):
    def answer(self, message, generate=False):
        ids = self.mentioned_ids(message)
        if "mais forte" in message.casefold():
            names = [self.retriever.by_id[pokemon_id]["name"] for pokemon_id in ids]
            return {
                "status": "clarification_needed",
                "gate_reason": "needs_comparison_criterion",
                "answer": f"Em qual atributo você quer comparar {' e '.join(names)}?",
                "source_documents": [],
                "supporting_excerpts": [],
            }
        return super().answer(message, generate=generate)


class RankingContextualEngine(ContextualEngine):
    def answer(self, message, generate=False):
        if "mais fortes" in message.casefold():
            return {"status": "clarification_needed", "gate_reason": "needs_comparison_criterion",
                    "answer": "Em qual atributo?", "source_documents": [], "supporting_excerpts": []}
        return super().answer(message, generate=generate)


def test_pronoun_comparison_keeps_both_pokemon_and_attribute_answers_clarification():
    manager = SessionManager(ComparisonContextualEngine())
    session = manager.create()
    manager.ask(session.session_id, "Fale sobre Eevee")

    clarification = manager.ask(session.session_id, "Ele é mais forte que o Pikachu?", generate=True)
    assert clarification["status"] == "clarification_needed"
    assert "Eevee" in clarification["answer"] and "Pikachu" in clarification["answer"]
    assert manager.get(session.session_id).active_ids == [133, 25]

    comparison = manager.ask(session.session_id, "Ataque", generate=True)
    assert comparison["answer"] == "Compare o ataque de Eevee e Pikachu."
    assert {item["pokemon_id"] for item in comparison["source_documents"]} == {133, 25}


def test_global_strength_ranking_uses_attribute_from_the_followup():
    manager = SessionManager(RankingContextualEngine())
    session = manager.create()
    clarification = manager.ask(session.session_id, "Quais são os pokemons mais fortes?")
    assert clarification["status"] == "clarification_needed"
    assert "ranquear" in clarification["answer"]
    ranking = manager.ask(session.session_id, "Ataque")
    assert ranking["answer"] == "Quais são os pokemons com maior ataque?"

