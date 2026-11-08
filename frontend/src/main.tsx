import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { chat, createSession } from './services/api';

type Message = { role: 'user' | 'assistant'; content: string };

function App() {
  const [session, setSession] = useState('');
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [busy, setBusy] = useState(false);

  async function typeAnswer(text: string) {
    setMessages((current) => [...current, { role: 'assistant', content: '' }]);
    for (let index = 1; index <= text.length; index += 1) {
      await new Promise((resolve) => window.setTimeout(resolve, 12));
      setMessages((current) => {
        const next = [...current];
        next[next.length - 1] = { role: 'assistant', content: text.slice(0, index) };
        return next;
      });
    }
  }

  async function send() {
    const question = message.trim();
    if (!question || busy) return;
    setMessage('');
    setMessages((current) => [...current, { role: 'user', content: question }]);
    setBusy(true);
    try {
      const currentSession = session || await createSession();
      setSession(currentSession);
      const response = await chat(currentSession, question, true);
      await typeAnswer(response.answer || 'Não foi possível gerar uma resposta.');
    } catch (error) {
      await typeAnswer(error instanceof Error ? error.message : 'Erro ao consultar o backend.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <main style={{ minHeight: '100vh', background: '#f3f4f6', padding: '2rem 1rem', fontFamily: 'system-ui' }}>
      <section style={{ maxWidth: 760, margin: '0 auto', background: 'white', borderRadius: 16, padding: 24, boxShadow: '0 8px 30px #00000012' }}>
        <header>
          <h1 style={{ margin: 0 }}>Pokédex Inteligente</h1>
          <p style={{ color: '#6b7280' }}>Converse sobre os Pokémon de Kanto e FireRed.</p>
        </header>

        <div style={{ minHeight: 360, maxHeight: 520, overflowY: 'auto', padding: '1rem 0' }}>
          {messages.length === 0 && <p style={{ color: '#9ca3af', textAlign: 'center', marginTop: 120 }}>Faça uma pergunta para começar.</p>}
          {messages.map((item, index) => (
            <div key={`${item.role}-${index}`} style={{ display: 'flex', justifyContent: item.role === 'user' ? 'flex-end' : 'flex-start', margin: '0.75rem 0' }}>
              <div style={{ maxWidth: '82%', whiteSpace: 'pre-wrap', lineHeight: 1.5, padding: '0.8rem 1rem', borderRadius: 14, background: item.role === 'user' ? '#2563eb' : '#eef2ff', color: item.role === 'user' ? 'white' : '#1f2937' }}>
                {item.content || (busy && item.role === 'assistant' ? 'Digitando…' : '')}
              </div>
            </div>
          ))}
        </div>

        <form onSubmit={(event) => { event.preventDefault(); void send(); }} style={{ display: 'flex', gap: 8 }}>
          <input value={message} onChange={(event) => setMessage(event.target.value)} placeholder="Como evoluir Charmander?" disabled={busy} style={{ flex: 1, border: '1px solid #d1d5db', borderRadius: 10, padding: '0.8rem' }} />
          <button type="submit" disabled={busy || !message.trim()} style={{ border: 0, borderRadius: 10, padding: '0 1.2rem', background: '#2563eb', color: 'white', cursor: 'pointer' }}>{busy ? '...' : 'Enviar'}</button>
        </form>
      </section>
    </main>
  );
}

createRoot(document.getElementById('root')!).render(<App />);

