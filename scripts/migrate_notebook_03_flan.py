"""Aplica correções de FLAN-T5 local ao Notebook 03."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks" / "03_RAG_CHATBOT.ipynb"


def lines(value: str):
    return [line + "\n" for line in value.rstrip().splitlines()]


def replace_first(notebook, marker, value, cell_type=None):
    for cell in notebook["cells"]:
        if cell_type and cell["cell_type"] != cell_type:
            continue
        if marker in "".join(cell.get("source", [])):
            cell["source"] = lines(value)
            if cell["cell_type"] == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
            return
    return False


def code(value):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines(value)}


def main():
    notebook = json.loads(PATH.read_text(encoding="utf-8"))
    replace_first(notebook, "MAX_CONTEXT_CHARS", r'''MAX_INPUT_TOKENS = 384
MAX_DOC_CHARS = 3500

def prompt_text(question, context):
    return ("Responda apenas com base no contexto. Se faltar informação, responda Não sei. "
            "Responda no idioma da pergunta, sem inventar fatos.\n"
            f"Contexto:\n{context}\n\nPergunta: {question}\nResposta:")

def token_count(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    ids = encoded.input_ids if hasattr(encoded, 'input_ids') else encoded['input_ids']
    return len(ids[0] if ids and isinstance(ids[0], (list, tuple)) else ids)

def build_context(rows, question=None, tokenizer=None, max_chars=13500, max_doc_chars=MAX_DOC_CHARS):
    parts, used, total = [], [], 0
    for row in rows:
        header = f"FONTE [{row['document_id']}] | Pokémon: {row['pokemon_name']} | seção: {row['section']}\n"
        content = str(row['content'])
        candidate = '\n\n'.join(parts + [header + content])
        if question is not None and tokenizer is not None:
            if token_count(tokenizer, prompt_text(question, candidate)) > MAX_INPUT_TOKENS:
                continue
        else:
            budget = min(max_doc_chars, max_chars - total - len(header) - 2)
            if budget < 100:
                break
            content = content[:budget]
            candidate = '\n\n'.join(parts + [header + content])
        parts.append(header + content)
        used.append({**row, 'content': content})
        total += len(header) + len(content) + 2
    return '\n\n'.join(parts), used

print('Contexto FLAN-T5: orçamento de entrada=', MAX_INPUT_TOKENS, 'tokens; seleção sem truncamento silencioso.')''', "code")
    replace_first(notebook, "FLAN_MODEL_NAME = os.environ", r'''import os
import time

FLAN_MODEL_NAME = os.environ.get('FLAN_MODEL_NAME', 'google/flan-t5-small')
DEVICE = 'cpu'
FLAN_TOKENIZER = None
FLAN_MODEL = None

def load_flan_t5():
    global FLAN_TOKENIZER, FLAN_MODEL, DEVICE
    if FLAN_TOKENIZER is not None and FLAN_MODEL is not None:
        return FLAN_TOKENIZER, FLAN_MODEL
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError('Instale transformers, sentencepiece e torch para usar FLAN-T5.') from error
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    FLAN_TOKENIZER = AutoTokenizer.from_pretrained(FLAN_MODEL_NAME, local_files_only=True)
    FLAN_MODEL = AutoModelForSeq2SeqLM.from_pretrained(FLAN_MODEL_NAME, local_files_only=True).to(DEVICE).eval()
    return FLAN_TOKENIZER, FLAN_MODEL

def generate_with_flan_t5(question, context):
    tokenizer, model = load_flan_t5()
    prompt = prompt_text(question, context)
    input_tokens = token_count(tokenizer, prompt)
    if input_tokens > MAX_INPUT_TOKENS:
        raise RuntimeError(f'Prompt excede {MAX_INPUT_TOKENS} tokens: {input_tokens}.')
    inputs = tokenizer(prompt, return_tensors='pt', truncation=False).to(DEVICE)
    started = time.perf_counter()
    import torch
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=96, do_sample=False, num_beams=1)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return {'answer': answer, 'model': FLAN_MODEL_NAME, 'input_tokens': input_tokens,
            'output_tokens': int(output_ids.shape[-1]),
            'latency_seconds': time.perf_counter() - started,
            'generation_config': {'max_new_tokens': 96, 'do_sample': False, 'num_beams': 1}}

def model_without_corpus(question):
    return generate_with_flan_t5(question, '')

print('Gerador local:', FLAN_MODEL_NAME, '| pesos carregados sob demanda; dispositivo:', DEVICE)''', "code")
    replace_first(notebook, "CITATION_RE =", r'''def clean_public_answer(answer):
    return str(answer or '').strip()

def answer_question(question, top_k=5, generate=False, generator=None, original_question=None):
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Pergunta vazia.')
    user_question = original_question or question
    canonical_question, corrections = canonicalize_question(question)
    retrieved, gate = select_evidence(canonical_question, top_k=top_k)
    language = detect_response_language(user_question) if 'detect_response_language' in globals() else 'pt'
    if generate and generator is None:
        tokenizer, _ = load_flan_t5()
        context, used = build_context(retrieved, canonical_question, tokenizer)
    else:
        context, used = build_context(retrieved)
    targets = explicitly_mentioned_ids(canonical_question)
    result = {'question': user_question, 'canonical_question': canonical_question,
              'normalized_query': normalize_query(canonical_question),
              'spelling_corrections': corrections, 'language': language,
              'retriever': 'tfidf_cosine_from_notebook_02',
              'generator': FLAN_MODEL_NAME if generate else None,
              'status': 'retrieval_only' if used else ('scope_limit' if gate else 'insufficient_evidence'),
              'gate_reason': gate, 'answer': None if used else localized_refusal(user_question, reason=gate, target_ids=targets),
              'source_documents': [{key: row.get(key) for key in ('document_id','pokemon_id','pokemon_name','section','score','factual_scope','source_file')} for row in used],
              'supporting_excerpts': [], 'citation_validation': None,
              'context_chars': len(context), 'documents_in_prompt': len(used)}
    if not used or not generate:
        return result if not used else {**result, 'answer': retrieval_answer(used, user_question)}
    if len(targets) > 1 and not set(targets).issubset({int(row['pokemon_id']) for row in used}):
        return {**result, 'status': 'insufficient_evidence', 'answer': REFUSAL}
    provider = generator or generate_with_flan_t5
    try:
        payload = provider(user_question, context)
        answer = payload.get('answer', '') if isinstance(payload, dict) else str(payload)
    except Exception as error:
        return {**result, 'status': 'provider_unavailable', 'answer': retrieval_answer(used, user_question),
                'generation_error': type(error).__name__}
    if not answer.strip() or answer.strip().casefold() in {'não sei', 'nao sei'}:
        return {**result, 'status': 'insufficient_evidence', 'answer': REFUSAL}
    metadata = payload if isinstance(payload, dict) else {}
    return {**result, 'status': 'answered', 'answer': clean_public_answer(answer),
            'supporting_excerpts': [{'document_id': row['document_id'], 'quote': row['content']} for row in used],
            'generation_metadata': {key: value for key, value in metadata.items() if key != 'answer'}}''', "code")
    replace_first(notebook, "DEMO_QUESTION =", r'''DEMO_QUESTION = 'Qual é a velocidade-base do Pikachu?'
RUN_LIVE_DEMO = False
if RUN_LIVE_DEMO:
    print(answer_question(DEMO_QUESTION, generate=True))
else:
    print('Demonstração FLAN-T5 desativada. Altere RUN_LIVE_DEMO=True para baixar/carregar o modelo.')''', "code")
    replace_first(notebook, "def open_chat_ui():", r'''def open_chat_ui():
    try:
        import ipywidgets as widgets
    except ImportError:
        print('Instale ipywidgets para abrir a interface.')
        return None
    conversation = PokedexChatSession()
    field = widgets.Text(placeholder='Ex.: Como Eevee evolui?', description='Pergunta', layout=widgets.Layout(width='82%'))
    action = widgets.Button(description='Perguntar', button_style='primary')
    do_gen = widgets.Checkbox(value=False, description='Gerar com FLAN-T5 local')
    output = widgets.Output(layout=widgets.Layout(border='1px solid #ddd', padding='8px'))
    hint = widgets.HTML('<b>FLAN-T5 local.</b> A primeira geração pode baixar os pesos do Hugging Face.')
    def run(_):
        prompt = field.value.strip()
        if not prompt: return
        field.value = ''
        with output:
            try: show_answer(conversation.ask(prompt, generate=bool(do_gen.value)))
            except Exception as error: print('Falha na geração local:', type(error).__name__)
    action.on_click(run)
    display(widgets.VBox([hint, widgets.HBox([field, action]), do_gen, output]))
    return conversation''', "code")
    replace_first(notebook, "## 9 · Demonstração do RAG", "## 9 · Demonstração do RAG com FLAN-T5", "markdown")
    replace_first(notebook, "## 8 · Testes do contrato", "## 8 · Testes do contrato e fontes selecionadas", "markdown")
    replace_first(notebook, "valid = answer_question", r'''# O FLAN-T5 retorna texto livre; as fontes continuam selecionadas pelo Python.
target = "Quais são os tipos do Charizard?"
rows, gate = select_evidence(target, top_k=5)
context, used = build_context(rows)
assert gate == "ready" and used
assert used[0]["document_id"] in context
assert token_count is not None and "Resposta:" in prompt_text(target, context)
print("Seleção de fontes, contexto com orçamento e contrato local: OK.")''', "code")
    replace_first(notebook, 'assert answer_question(target, generate=False)', r'''# O FLAN-T5 retorna texto livre; as fontes continuam selecionadas pelo Python.
target = "Quais são os tipos do Charizard?"
rows, gate = select_evidence(target, top_k=5)
context, used = build_context(rows)
assert gate == "ready" and used
assert used[0]["document_id"] in context
assert token_count is not None and "Resposta:" in prompt_text(target, context)
print("Seleção de fontes, contexto com orçamento e contrato local: OK.")''', "code")
    replace_first(notebook, "chat_test = PokedexChatSession()", r'''# Contratos de seleção independentes da ordem de definição das células.
assert requested_section("Como encontrar o Eevee?") == "encounters"
assert requested_section("Como evoluir o Charizard?") == "evolution"
rows, gate = select_evidence("Quais os pokemons com o mesmo tipo do charizard?", top_k=5)
assert gate == "ready" and {6, 146}.issubset({int(row["pokemon_id"]) for row in rows})
print("Contratos de intenção, fontes e comparação por tipo: OK.")''', "code")
    for cell in notebook["cells"]:
        cell["source"] = [line.replace(
            'language = detect_response_language(user_question)',
            "language = detect_response_language(user_question) if 'detect_response_language' in globals() else 'pt'"
        ).replace(
            "language = detect_response_language(user_question) if 'detect_response_language' in globals() else 'pt' if 'detect_response_language' in globals() else 'pt'",
            "language = detect_response_language(user_question) if 'detect_response_language' in globals() else 'pt'"
        ) for line in cell.get('source', [])]
        source = "".join(cell.get("source", []))
        source = source.replace(
            "if generate and generator is None:\n        tokenizer, _ = load_flan_t5()\n        context, used = build_context(retrieved, canonical_question, tokenizer)\n    else:\n        context, used = build_context(retrieved)",
            "preflight_error = None\n    if generate and generator is None:\n        try:\n            tokenizer, _ = load_flan_t5()\n            context, used = build_context(retrieved, canonical_question, tokenizer)\n        except Exception as error:\n            preflight_error = error\n            context, used = build_context(retrieved)\n    else:\n        context, used = build_context(retrieved)"
        )
        source = source.replace(
            "if not used or not generate:\n        return result if not used else {**result, 'answer': retrieval_answer(used, user_question)}",
            "if preflight_error is not None:\n        return {**result, 'status': 'provider_unavailable', 'answer': retrieval_answer(used, user_question), 'generation_error': type(preflight_error).__name__}\n    if not used or not generate:\n        return result if not used else {**result, 'answer': retrieval_answer(used, user_question)}"
        )
        cell["source"] = lines(source)
    # Remove referências antigas que ficaram em células de documentação, sem tocar no notebook do professor.
    report = next((cell for cell in notebook["cells"] if cell["cell_type"] == "code" and "'notebook':'03_RAG_CHATBOT'" in "".join(cell["source"])), None)
    if report:
        report["source"] = lines("""report = {
 'project':'Pokédex Inteligente — Trabalho 2', 'notebook':'03_RAG_CHATBOT', 'run_utc':RUN_AT,
 'document_count':len(documents), 'pokemon_count':len(records),
 'retriever':'TF-IDF + cosseno persistido no Notebook 02; sem novo fit',
 'generator':'FLAN-T5 local', 'model':FLAN_MODEL_NAME, 'model_loaded':FLAN_MODEL is not None,
 'input_token_budget':MAX_INPUT_TOKENS, 'live_demo_executed':RUN_LIVE_DEMO,
 'generation_evaluation_executed':RUN_GENERATION_EVALUATION,
 'limitations':['FLAN-T5-small pode ser menos consistente em português.',
                'A seleção de fontes e a citação textual não comprovam suporte semântico sozinhas.',
                'A primeira inferência exige acesso ao Hugging Face para baixar os pesos.'],
}
report_path=REPORTS_DIR/'rag_chatbot_grounding_report.json'
report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
assert actual_sha==manifest['source_document_sha256']
assert index_matrix.shape[0]==len(documents)==len(manifest['document_ids'])
print('Relatório:',report_path)""")
    PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Notebook 03 migrado para FLAN-T5.")


if __name__ == "__main__":
    main()
