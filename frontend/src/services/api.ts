import type { ChatResponse } from '../types/api';
const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8000/api/v1';
export async function createSession(): Promise<string> { const r=await fetch(`${API}/sessions`,{method:'POST'}); return (await r.json()).session_id; }
export async function chat(session_id:string,message:string):Promise<ChatResponse>{ const r=await fetch(`${API}/chat`,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({session_id,message})}); if(!r.ok) throw new Error(await r.text()); return r.json(); }

