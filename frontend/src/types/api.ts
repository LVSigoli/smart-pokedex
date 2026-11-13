export type ChatStatus = 'answered' | 'retrieval_only' | 'clarification_required' | 'out_of_scope' | 'insufficient_evidence' | 'provider_unavailable';
export type Source = { document_id: string; pokemon_id: number; pokemon_name: string; section: string; score?: number };
export type ChatResponse = { session_id: string; status: ChatStatus; answer?: string | null; answer_mode?: 'generative' | 'extractive' | 'extractive_fallback' | null; source_documents: Source[]; supporting_excerpts: { document_id: string; quote: string }[]; generation_metadata?: Record<string, unknown> | null };
export type PokemonRecord = { id: number; name: string; api_name?: string; [key: string]: unknown };
export type PokemonListResponse = { items: PokemonRecord[]; page: number; page_size: number; total: number };

