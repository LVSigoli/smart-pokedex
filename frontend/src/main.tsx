import { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { chat, createSession, deleteSession, getPokemon, listPokemon } from './services/api';
import type { ChatResponse, PokemonRecord } from './types/api';
import './styles.css';

type Message = { role: 'user' | 'assistant'; content: string; response?: ChatResponse };
const titleCase = (value: string) => value.replace(/-/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
const statValue = (pokemon: PokemonRecord, key: string) => String(pokemon.stats_gen3?.[key] ?? '—');

function SourceChips({ response }: { response?: ChatResponse }) {
  if (!response || response.source_documents.length === 0) return null;
  return <div className="sources"><span className="source-label">FONTES</span>{response.source_documents.map((source) => <span className="source-badge" key={source.document_id}>{source.document_id}</span>)}</div>;
}

function PokemonDisplay({ pokemon, records, onChange }: { pokemon: PokemonRecord; records: PokemonRecord[]; onChange: (id: number) => void }) {
  const types = pokemon.types_gen3 ?? [];
  const sprite = pokemon.sprites?.firered_front_default ?? pokemon.sprites?.current_front_default;
  return <section className="left-column"><div className="left-bezel"><div className="display">
    <div className="display-top"><span>REGISTRO ATIVO</span><label className="display-select"><span className="sr-only">Selecionar Pokémon</span><select value={pokemon.id} onChange={(event) => onChange(Number(event.target.value))}>{records.map((item) => <option value={item.id} key={item.id}>Nº {String(item.id).padStart(3, '0')} / 151 · {item.name}</option>)}</select></label></div>
    <div className="pokemon-art">{sprite ? <img src={sprite} alt={`Sprite de ${pokemon.name}`} /> : <span className="art-placeholder">?</span>}</div>
    <div className="pokemon-meta"><div><div className="poke-name">{pokemon.name}</div><div className="poke-number">POKÉMON · #{String(pokemon.id).padStart(3, '0')}</div></div><div className="type-pill">✦ {titleCase(types[0] ?? 'desconhecido')}</div></div>
  </div><div className="unit-grid"><div className="unit"><small>ATAQUE</small><strong>{statValue(pokemon, 'attack')}</strong><span>base stat</span></div><div className="unit"><small>VELOCIDADE</small><strong>{statValue(pokemon, 'speed')}</strong><span>base stat</span></div><div className="unit"><small>ALTURA</small><strong>{String(pokemon.height_m ?? '—').replace('.', ',')}<em> m</em></strong><span>registro</span></div></div></div>
    <div className="left-controls"><div className="control-cross"><i className="cross-v" /><i className="cross-h" /><i className="cross-center" /></div><div className="action-buttons"><button type="button" aria-label="Voltar" className="round-button">B</button><button type="button" aria-label="Confirmar" className="round-button last">A</button></div></div><div className="left-footer"><span>● POKÉDEX DATABASE / ONLINE</span><span className="vent"><i /><i /><i /></span></div>
  </section>;
}

function ChatPanel({ activePokemon }: { activePokemon: PokemonRecord }) {
  const [session, setSession] = useState(''); const [question, setQuestion] = useState(''); const [messages, setMessages] = useState<Message[]>([]); const [busy, setBusy] = useState(false); const [error, setError] = useState('');
  const ask = async (value: string) => { const text = value.trim(); if (!text || busy) return; setQuestion(''); setError(''); setMessages((current) => [...current, { role: 'user', content: text }]); setBusy(true); try { const activeSession = session || await createSession(); setSession(activeSession); const response = await chat(activeSession, text, true); setMessages((current) => [...current, { role: 'assistant', content: response.answer || 'Não foi possível gerar uma resposta.', response }]); } catch (reason) { setError(reason instanceof Error ? reason.message : 'Erro ao consultar a API.'); } finally { setBusy(false); } };
  const reset = async () => { if (session) await deleteSession(session).catch(() => undefined); setSession(''); setMessages([]); setError(''); };
  return <section className="right-column"><div className="right-screen"><div className="screen-top"><div className="screen-title"><div className="assistant-ico">⌁</div><div><strong>Assistente Pokédex</strong><p>Pergunte sobre Pokémon, atributos e evoluções</p></div></div><div className="screen-status"><span /> GEMINI ONLINE</div></div>
    <div className="messages" aria-live="polite">{messages.length === 0 && <><span className="tiny-time">NOVA CONVERSA · KANTO / FIRE RED</span><div className="empty-message">Selecione uma pergunta abaixo para consultar o RAG sobre <strong>{activePokemon.name}</strong>.</div></>}{messages.map((message, index) => <div className={`message ${message.role}`} key={`${message.role}-${index}`}>{message.role === 'assistant' && <div className="small-ico">✦</div>}<div className="message-content"><div className="message-bubble">{message.content}</div>{message.role === 'assistant' && <SourceChips response={message.response} />}</div></div>)}{busy && <div className="message assistant"><div className="small-ico">✦</div><div className="message-bubble typing"><i /><i /><i /></div></div>}{error && <div className="api-error" role="alert">{error}</div>}</div>
    <form className="message-form" onSubmit={(event) => { event.preventDefault(); void ask(question); }}><div className="suggestions"><button type="button" onClick={() => void ask(`Como evolui ${activePokemon.name}?`)}>Como evoluir?</button><button type="button" onClick={() => void ask(`Quais são os atributos de ${activePokemon.name}?`)}>Atributos</button><button type="button" onClick={() => void ask(`Onde encontro ${activePokemon.name} no FireRed?`)}>Onde encontro?</button><button type="button" onClick={() => void reset()}>Nova conversa ↻</button></div><div className="input-line"><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Pergunte algo à sua Pokédex..." disabled={busy} /><button type="submit" disabled={busy || !question.trim()}>Enviar ↗</button></div></form>
  </div><div className="screen-foot"><span><span className="foot-led" /> RECUPERAÇÃO TF-IDF + GERAÇÃO GEMINI</span><span>SMART DEVICE / 01</span></div></section>;
}

function App() {
  const [records, setRecords] = useState<PokemonRecord[]>([]); const [active, setActive] = useState<PokemonRecord | null>(null); const [error, setError] = useState('');
  useEffect(() => { listPokemon().then((items) => { setRecords(items); const initial = items.find((item) => item.id === 25) ?? items[0]; if (initial) return getPokemon(initial.id).then(setActive); }).catch((reason) => setError(reason instanceof Error ? reason.message : 'Não foi possível carregar os Pokémon.')); }, []);
  return <div className="stage"><div className="stage-top"><div><div className="mark">✧</div><div><div className="stage-title">Smart Pokédex</div><div className="stage-sub">Conceito de interface · dispositivo clássico + experiência conversacional</div></div></div><div className="stage-chip">WEB EXPERIENCE / CONCEPT 01</div></div><div className="device"><div className="chrome-head"><div className="lights"><div className="lens" /><i className="led r" /><i className="led y" /><i className="led g" /></div><div className="device-brand"><b>SMART POKÉDEX</b><small>KANTO REGION · GEN III</small></div></div>{error ? <div className="loading-error" role="alert">{error}</div> : !active ? <div className="loading-state">Carregando dados do corpus…</div> : <div className="main-grid"><PokemonDisplay pokemon={active} records={records} onChange={(id) => { void getPokemon(id).then(setActive); }} /><div className="hinge" /><ChatPanel activePokemon={active} /></div>}<div className="device-foot"><span>● {records.length || 151} REGISTROS VALIDADOS / ONLINE</span><span>TF-IDF + EVIDÊNCIAS</span></div></div></div>;
}

createRoot(document.getElementById('root')!).render(<App />);
