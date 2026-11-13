import type { ChatResponse, PokemonListResponse, PokemonRecord } from '../types/api';
const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8000/api/v1';
export async function createSession(): Promise<string> { const r=await fetch(`${API}/sessions`,{method:'POST'}); return (await r.json()).session_id; }
export async function chat(session_id:string,message:string,generate=true):Promise<ChatResponse>{ const r=await fetch(`${API}/chat`,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({session_id,message,generate})}); if(!r.ok) throw new Error(await r.text()); return r.json(); }
async function pokemonPage(page: number): Promise<PokemonListResponse> {
  const response = await fetch(`${API}/pokemon?page=${page}&page_size=100`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}
export async function listPokemon(): Promise<PokemonRecord[]> {
  const first = await pokemonPage(1);
  const pages = [first];
  for (let page = 2; pages.reduce((total, item) => total + item.items.length, 0) < first.total; page += 1) {
    pages.push(await pokemonPage(page));
  }
  return pages.flatMap((page) => page.items).sort((a, b) => a.id - b.id);
}

