import json
from app.core.exceptions import ProviderUnavailable

class GeminiGenerator:
    def __init__(self, api_key: str | None, model: str):
        self.model = model
        self.client = None
        if api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=api_key)
            except Exception as exc:
                raise ProviderUnavailable("Cliente Gemini indisponível") from exc

    def __call__(self, question: str, context: str):
        if not self.client:
            raise ProviderUnavailable("GEMINI_API_KEY não configurada")
        prompt = ("Responda exclusivamente com JSON {can_answer:boolean, answer:string, evidence:[{document_id,quote}]}\n"
                   "Use somente as fontes abaixo, inclua citações [document_id] e quotes literais.\n"
                   f"PERGUNTA: {question}\nFONTES:\n{context}")
        response = self.client.models.generate_content(model=self.model, contents=prompt)
        return json.loads(getattr(response, "text", ""))

