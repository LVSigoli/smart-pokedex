import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { chat, createSession, listPokemon } from './services/api';
import type { ChatResponse, PokemonRecord } from './types/api';
import './styles.css';

type View = 'chat' | 'lab';
type Message = { role: 'user' | 'assistant'; content: string; response?: ChatResponse; context?: boolean };
type LabTurn = Message & { turn: number; error?: string };

const FALLBACK_QUESTIONS = ['Como evoluir Charmander?', 'Onde encontro o Eevee?', 'Qual é a velocidade do Pikachu?'];
const STATUS_LABELS: Record<string, string> = {
  answered: 'respondida', retrieval_only: 'RAG extrativo', clarification_required: 'precisa esclarecer',
  insufficient_evidence: 'evidência insuficiente', out_of_scope: 'fora do escopo', provider_unavailable: 'modelo indisponível',
};

function labelStatus(status?: string) { return STATUS_LABELS[status ?? ''] ?? status ?? 'sem status'; }
function delay(milliseconds: number) { return new Promise((resolve) => window.setTimeout(resolve, milliseconds)); }

function Sources({ response }: { response?: ChatResponse }) {
  if (!response) return null;
  const names = response.source_documents.map((source) => source.pokemon_name).filter((name, index, all) => all.indexOf(name) === index);
  return <div className="message-meta"><span className={`status-dot status-${response.status}`} /><span>{labelStatus(response.status)}</span>{response.answer_mode && <span className="mode-chip">{response.answer_mode.replace('_', ' ')}</span>}{names.length > 0 && <span>{names.join(' · ')}</span>}</div>;
}

function ChatPanel() {
  const [session, setSession] = useState('');
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [busy, setBusy] = useState(false);

  async function typeAnswer(text: string, response?: ChatResponse) {
    setMessages((current) => [...current, { role: 'assistant', content: '', response }]);
    for (let index = 1; index <= text.length; index += 1) {
      await delay(10);
      setMessages((current) => { const next = [...current]; const last = next[next.length - 1]; next[next.length - 1] = { ...last, content: text.slice(0, index) }; return next; });
    }
  }

  async function send() {
    const question = message.trim();
    if (!question || busy) return;
    setMessage(''); setMessages((current) => [...current, { role: 'user', content: question }]); setBusy(true);
    try {
      const currentSession = session || await createSession(); setSession(currentSession);
      const response = await chat(currentSession, question, true);
      await typeAnswer(response.answer || 'Não foi possível gerar uma resposta.', response);
    } catch (error) { await typeAnswer(error instanceof Error ? error.message : 'Erro ao consultar o backend.'); }
    finally { setBusy(false); }
  }

  return <section className="panel chat-panel">
    <div className="panel-heading"><div><span className="eyebrow">conversa livre</span><h2>Converse com a Pokédex</h2><p>Faça perguntas e acompanhe as fontes que fundamentaram cada resposta.</p></div><span className="live-pill"><span /> API online</span></div>
    <div className="chat-window">
      {messages.length === 0 && <div className="empty-chat"><div className="empty-orbit">✦</div><strong>Comece uma conversa</strong><span>Experimente perguntar sobre evolução, localização ou atributos.</span><div className="suggestion-row">{FALLBACK_QUESTIONS.map((question) => <button key={question} onClick={() => setMessage(question)}>{question}</button>)}</div></div>}
      {messages.map((item, index) => <div className={`message-row ${item.role}`} key={`${item.role}-${index}`}><div className={`avatar ${item.role}`}>{item.role === 'user' ? 'Você' : 'RAG'}</div><div className="message-stack"><div className="message-bubble">{item.content || (busy && item.role === 'assistant' ? <span className="typing"><i /><i /><i /></span> : '')}</div>{item.role === 'assistant' && <Sources response={item.response} />}</div></div>)}
    </div>
    <form className="composer" onSubmit={(event) => { event.preventDefault(); void send(); }}><input value={message} onChange={(event) => setMessage(event.target.value)} placeholder="Como evoluir Charmander?" disabled={busy} /><button className="primary-button" type="submit" disabled={busy || !message.trim()}>{busy ? 'Consultando' : 'Enviar'} <span>↗</span></button></form>
    <div className="panel-footnote">Sessão {session ? `${session.slice(0, 8)}…` : 'ainda não iniciada'} · FLAN-T5 local + TF-IDF</div>
  </section>;
}

function PokemonSelect({ label, value, records, onChange }: { label: string; value: number; records: PokemonRecord[]; onChange: (value: number) => void }) {
  return <label className="select-field"><span>{label}</span><select value={value} onChange={(event) => onChange(Number(event.target.value))}>{records.map((pokemon) => <option key={pokemon.id} value={pokemon.id}>#{String(pokemon.id).padStart(3, '0')} · {pokemon.name}</option>)}</select></label>;
}

function LabPanel() {
  const [records, setRecords] = useState<PokemonRecord[]>([]); const [loadingRecords, setLoadingRecords] = useState(true); const [loadError, setLoadError] = useState('');
  const [firstId, setFirstId] = useState(25); const [secondId, setSecondId] = useState(133); const [scenario, setScenario] = useState<'individual' | 'pair'>('individual'); const [generate, setGenerate] = useState(false);
  const [running, setRunning] = useState(false); const [progress, setProgress] = useState(0); const [turns, setTurns] = useState<LabTurn[]>([]); const [runError, setRunError] = useState('');

  useEffect(() => { let mounted = true; listPokemon().then((items) => { if (!mounted) return; setRecords(items); if (!items.some((item) => item.id === 25)) setFirstId(items[0]?.id ?? 1); if (!items.some((item) => item.id === 133)) setSecondId(items[1]?.id ?? 2); setLoadingRecords(false); }).catch((error) => { if (!mounted) return; setLoadError(error instanceof Error ? error.message : 'Não foi possível carregar os Pokémon.'); setLoadingRecords(false); }); return () => { mounted = false; }; }, []);

  const names = useMemo(() => new Map(records.map((pokemon) => [pokemon.id, pokemon.name])), [records]); const firstName = names.get(firstId) ?? 'Pikachu'; const secondName = names.get(secondId) ?? 'Eevee';
  const prompts = scenario === 'individual' ? [
    { text: `Fale sobre o Pokémon ${firstName}`, contextual: false }, { text: `Qual é o ataque base de ${firstName}?`, contextual: false }, { text: `Onde encontro ${firstName}?`, contextual: false }, { text: 'Como evolui ele', contextual: true }, { text: 'Onde encontro ele', contextual: true },
  ] : [
    { text: `Quais são os tipos de ${firstName} e ${secondName}?`, contextual: false }, { text: 'Qual dos dois tem maior velocidade?', contextual: true }, { text: 'Onde encontro eles', contextual: true }, { text: `Onde encontro ${firstName} e ${secondName}?`, contextual: false }, { text: `Agora fale sobre ${secondName}`, contextual: false }, { text: 'Onde encontro ele', contextual: true },
  ];

  async function run() {
    if (running || loadingRecords || !records.length) return; setRunning(true); setTurns([]); setRunError(''); setProgress(0);
    try { const session = await createSession(); for (let index = 0; index < prompts.length; index += 1) { const prompt = prompts[index]; setTurns((current) => [...current, { role: 'user', content: prompt.text, turn: index + 1, context: prompt.contextual }]); const response = await chat(session, prompt.text, generate); setTurns((current) => [...current, { role: 'assistant', content: response.answer || 'Sem resposta textual.', response, turn: index + 1, context: prompt.contextual }]); setProgress(index + 1); } }
    catch (error) { setRunError(error instanceof Error ? error.message : 'O teste não pôde ser concluído.'); }
    finally { setRunning(false); }
  }

  const completedTurns = turns.filter((turn) => turn.role === 'assistant'); const answered = completedTurns.filter((turn) => turn.response?.status === 'answered' || turn.response?.status === 'retrieval_only').length;
  return <section className="panel lab-panel">
    <div className="panel-heading lab-heading"><div><span className="eyebrow">laboratório conversacional</span><h2>Veja o RAG pensando em contexto</h2><p>Execute uma conversa controlada pela API e acompanhe fontes, status e troca de Pokémon turno a turno.</p></div><div className="lab-mark">RAG<br /><small>LAB</small></div></div>
    <div className="lab-controls"><div className="scenario-tabs"><button className={scenario === 'individual' ? 'selected' : ''} onClick={() => setScenario('individual')} disabled={running}>Um Pokémon</button><button className={scenario === 'pair' ? 'selected' : ''} onClick={() => setScenario('pair')} disabled={running}>Comparar dois</button></div><div className="select-grid"><PokemonSelect label="Pokémon principal" value={firstId} records={records} onChange={setFirstId} />{scenario === 'pair' && <PokemonSelect label="Segundo Pokémon" value={secondId} records={records} onChange={setSecondId} />}</div><div className="run-options"><label className="toggle-label"><input type="checkbox" checked={generate} onChange={(event) => setGenerate(event.target.checked)} disabled={running} /><span className="toggle" /> Usar FLAN-T5 local</label><button className="primary-button run-button" onClick={() => void run()} disabled={running || loadingRecords || Boolean(loadError)}>{running ? `Executando ${progress}/${prompts.length}` : 'Executar conversa'} <span>▶</span></button></div>{loadingRecords && <p className="inline-note">Carregando os 151 Pokémon…</p>}{loadError && <p className="error-note">{loadError}</p>}</div>
    {runError && <div className="error-banner">{runError}</div>}
    {turns.length === 0 ? <div className="lab-empty"><div className="lab-empty-icon">◎</div><strong>Escolha um cenário e execute</strong><span>O laboratório vai criar uma nova sessão e enviar cada turno na ordem, como o chatbot real.</span></div> : <div className="lab-conversation"><div className="run-summary"><span><b>{completedTurns.length}</b>/{prompts.length} turnos</span><span><b>{answered}</b> respostas com fonte</span><span>{generate ? 'FLAN-T5 habilitado' : 'modo determinístico'}</span></div>{turns.map((turn, index) => <div className={`lab-turn ${turn.role}`} key={`${turn.turn}-${turn.role}-${index}`}><div className="turn-index">{String(turn.turn).padStart(2, '0')}</div><div className="turn-content"><div className="turn-label">{turn.role === 'user' ? 'pergunta' : 'resposta'} {turn.context && <span className="context-chip">↳ contexto ativo</span>}</div><div className="turn-text">{turn.content}</div>{turn.role === 'assistant' && <Sources response={turn.response} />}{turn.role === 'assistant' && turn.response?.source_documents.length > 0 && <details className="source-details"><summary>Ver documentos recuperados ({turn.response.source_documents.length})</summary><div>{turn.response.source_documents.map((source) => <span key={source.document_id} className="source-chip">{source.document_id} · {source.section}</span>)}</div></details>}</div></div>)}</div>}
  </section>;
}

function App() {
  const [view, setView] = useState<View>('chat');
  return <div className="app-shell"><header className="topbar"><div className="brand"><div className="brand-icon">✦</div><div><strong>Pokédex</strong><span>INTELIGENTE</span></div></div><nav className="main-nav"><button className={view === 'chat' ? 'active' : ''} onClick={() => setView('chat')}>Chat</button><button className={view === 'lab' ? 'active' : ''} onClick={() => setView('lab')}>Teste RAG <span className="new-badge">LAB</span></button></nav><div className="scope-label"><span /> KANTO · 151</div></header><main className="app-content">{view === 'chat' ? <ChatPanel /> : <LabPanel />}</main><footer className="app-footer">TF-IDF + FLAN-T5 local <span>·</span> Evidência antes da geração</footer></div>;
}

createRoot(document.getElementById('root')!).render(<App />);
