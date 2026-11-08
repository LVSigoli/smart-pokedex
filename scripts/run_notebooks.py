"""Executa os notebooks acadêmicos com caminhos locais do monorepo.

As adaptações são feitas em memória: os notebooks originais não são sobrescritos.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
EXECUTED = ROOT / "data" / "executed_notebooks"


def code_cells(notebook):
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


def adapt(notebook, name: str):
    """Adapta somente caminhos/integrações de ambiente, sem alterar a lógica RAG."""
    for cell in code_cells(notebook):
        source = "".join(cell.source)

        source = source.replace(
            "from google.colab import drive",
            "try:\n    from google.colab import drive\nexcept ImportError:\n    class _Drive:\n        @staticmethod\n        def mount(*args, **kwargs):\n            return None\n    drive = _Drive()",
        )
        source = source.replace("drive.mount('/content/drive')", "drive.mount('/content/drive') if IN_COLAB else None")
        source = source.replace('drive.mount("/content/drive")', 'drive.mount("/content/drive") if IN_COLAB else None')

        if name == "00_CORPUS_POKEMON.ipynb":
            source = "IN_COLAB = False\n" + source
            source = source.replace(
                'from google.colab import drive\n\n',
                'try:\n    from google.colab import drive\nexcept ImportError:\n    class _Drive:\n        @staticmethod\n        def mount(*args, **kwargs):\n            return None\n    drive = _Drive()\n\n',
            )
            source = source.replace('drive.mount("/content/drive")', 'drive.mount("/content/drive") if "google.colab" in str(drive.__class__) else None')
            source = source.replace(
                'base_path = Path(\n    "/content/drive/MyDrive/IFSP-ADS/ultimo_periodo/tc_2/avaliacao/corpus"\n)',
                f'base_path = Path(r"{(ROOT / "data").as_posix()}")',
            )
            source = source.replace(
                'root = base_path / "corpus_data_firered"',
                'root = base_path',
            )
            source = source.replace('RAW_DIR = root / "raw_data"', 'RAW_DIR = root / "raw"')
            source = source.replace('PROCESSED_DIR = root / "processed_data" / "txt"', 'PROCESSED_DIR = root / "processed" / "txt"')
            source = source.replace('METADATA_DIR = root / "metadata"', 'METADATA_DIR = root / "processed" / "metadata"')
            source = source.replace('ARTIFACTS_DIR = root / "artifacts"', 'ARTIFACTS_DIR = root / "artifacts" / "corpus"')
            source = source.replace('RESULTS_DIR = root / "results"', 'RESULTS_DIR = root / "reports"')
            source = source.replace('LOG_DIR = root / "logs" / "trabalho_1"', 'LOG_DIR = root / "logs" / "corpus"')

        elif name == "01_RAG_DATA_PREPARATION.ipynb":
            source = source.replace("AVALIACAO_DIR = (Path('/content/drive/MyDrive/IFSP-ADS/ultimo_periodo/tc_2/avaliacao')\n                 if IN_COLAB else Path.cwd())", f"AVALIACAO_DIR = Path(r'{ROOT.as_posix()}')")
            source = source.replace("SOURCE_DIR = AVALIACAO_DIR / 'corpus'", "SOURCE_DIR = AVALIACAO_DIR / 'data'")
            source = source.replace("RAW_DIR = SOURCE_DIR / 'corpus_data_firered' / 'raw_data'", "RAW_DIR = SOURCE_DIR / 'raw'")
            source = source.replace("APPLICATION_DIR = AVALIACAO_DIR / 'application'", "APPLICATION_DIR = AVALIACAO_DIR")
            source = source.replace("RAG_DIR = APPLICATION_DIR / 'data' / 'rag'", "RAG_DIR = APPLICATION_DIR / 'data'")
            source = source.replace("DOC_DIR = RAG_DIR / 'documents'", "DOC_DIR = RAG_DIR / 'processed'")
            source = source.replace("REFERENCE_DIR = RAG_DIR / 'reference'", "REFERENCE_DIR = RAG_DIR / 'processed' / 'reference'")
            source = source.replace("REPORT_DIR = APPLICATION_DIR / 'reports'", "REPORT_DIR = APPLICATION_DIR / 'data' / 'reports'")
            source = source.replace("LOG_DIR = APPLICATION_DIR / 'logs'", "LOG_DIR = APPLICATION_DIR / 'data' / 'logs'")

        elif name in {"02_RETRIEVAL_AND_EVALUATION.ipynb", "03_RAG_CHATBOT.ipynb"}:
            source = source.replace("DEFAULT_AVALIACAO_DIR = (\n    '/content/drive/MyDrive/IFSP-ADS/ultimo_periodo/tc_2/avaliacao'\n    if IN_COLAB else str(Path.cwd())\n)", f"DEFAULT_AVALIACAO_DIR = r'{ROOT.as_posix()}'")
            source = source.replace("APPLICATION_DIR = AVALIACAO_DIR / 'application'", "APPLICATION_DIR = AVALIACAO_DIR")
            source = source.replace("RAG_DIR = APPLICATION_DIR / 'data' / 'rag'", "RAG_DIR = APPLICATION_DIR / 'data'")
            source = source.replace("DOC_DIR = RAG_DIR / 'documents'", "DOC_DIR = RAG_DIR / 'processed'")
            source = source.replace("REFERENCE_DIR = RAG_DIR / 'reference'", "REFERENCE_DIR = RAG_DIR / 'processed' / 'reference'")
            source = source.replace("EVAL_DIR = APPLICATION_DIR / 'data' / 'evaluation'", "EVAL_DIR = APPLICATION_DIR / 'data' / 'evaluation'")
            source = source.replace("ARTIFACTS_DIR = APPLICATION_DIR / 'artifacts' / 'retrieval'", "ARTIFACTS_DIR = APPLICATION_DIR / 'data' / 'artifacts' / 'retrieval'")
            source = source.replace("REPORT_DIR = APPLICATION_DIR / 'reports'", "REPORT_DIR = APPLICATION_DIR / 'data' / 'reports'")
            source = source.replace("REPORTS_DIR = APPLICATION_DIR / 'reports'", "REPORTS_DIR = APPLICATION_DIR / 'data' / 'reports'")
            source = source.replace("LOG_DIR = APPLICATION_DIR / 'logs'", "LOG_DIR = APPLICATION_DIR / 'data' / 'logs'")
            source = source.replace("LOGS_DIR = APPLICATION_DIR / 'logs'", "LOGS_DIR = APPLICATION_DIR / 'data' / 'logs'")

        if name == "03_RAG_CHATBOT.ipynb":
            source = source.replace('os.environ["GEMINI_API_KEY"] = getpass(\n    "Digite sua Gemini API Key: "\n)', 'os.environ.setdefault("GEMINI_API_KEY", "")')
            source = source.replace("GEMINI_MODEL=os.environ.get('GEMINI_MODEL','gemini-3.5-flash-lite')", "GEMINI_MODEL=os.environ.get('GEMINI_MODEL','gemini-2.0-flash')")

        cell.source = source
    return notebook


def execute(name: str):
    source = NOTEBOOKS / name
    os.environ["PLN_AVALIACAO_DIR"] = str(ROOT)
    notebook = adapt(nbformat.read(source, as_version=4), name)
    EXECUTED.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(notebook, timeout=-1, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbformat.write(notebook, EXECUTED / name)
    print(f"OK: {name} -> {EXECUTED / name}")


def main():
    parser = argparse.ArgumentParser(description="Executa o pipeline acadêmico localmente")
    parser.add_argument("--from-step", choices=["00", "01", "02", "03"], default="00")
    args = parser.parse_args()
    order = ["00_CORPUS_POKEMON.ipynb", "01_RAG_DATA_PREPARATION.ipynb", "02_RETRIEVAL_AND_EVALUATION.ipynb", "03_RAG_CHATBOT.ipynb"]
    start = int(args.from_step)
    for index, name in enumerate(order):
        if index >= start:
            execute(name)


if __name__ == "__main__":
    main()

