from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    flan_model: str = "google/flan-t5-small"
    rag_data_dir: Path = Path("../data")
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    session_ttl_seconds: int = 1800
    max_message_chars: int = 1000
    rag_top_k: int = 5
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def origins(self) -> list[str]:
        return [x.strip() for x in self.allowed_origins.split(",") if x.strip()]

@lru_cache
def get_settings() -> Settings:
    return Settings()

