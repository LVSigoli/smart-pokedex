from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    gemini_api_key: str = ""
    gemini_model_name: str = "gemini-3.5-flash-lite"
    gemini_max_input_tokens: int = 8192
    gemini_max_output_tokens: int = 768
    gemini_temperature: float = 0.1
    rag_data_dir: Path = Path("../data")
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    session_ttl_seconds: int = 1800
    max_message_chars: int = 1000
    rag_top_k: int = 5
    rag_min_score: float = 0.15
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def origins(self) -> list[str]:
        return [x.strip() for x in self.allowed_origins.split(",") if x.strip()]

@lru_cache
def get_settings() -> Settings:
    return Settings()

