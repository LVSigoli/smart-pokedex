export type ChatStatus = 'answered' | 'retrieval_only' | 'clarification_required' | 'out_of_scope' | 'insufficient_evidence' | 'provider_unavailable';
export type Source = { document_id: string; pokemon_id: number; pokemon_name: string; section: string; score?: number };
export type ChatResponse = { session_id: string; status: ChatStatus; answer?: string | null; source_documents: Source[]; supporting_excerpts: { document_id: string; quote: string }[] };

