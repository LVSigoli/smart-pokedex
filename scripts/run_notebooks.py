"""Executa os quatro notebooks oficiais com a raiz do projeto configurada."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

try:
    import jupyter_core.paths as _jupyter_paths
    _jupyter_paths.win32_restrict_file_to_user = lambda _filename: None
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
EXECUTED = ROOT / "data" / "executed_notebooks"
ORDER = [
    "00_CORPUS_POKEMON.ipynb",
    "01_RAG_DATA_PREPARATION.ipynb",
    "02_RAG_RETRIEVAL.ipynb",
    "03_SMART_POKEDEX.ipynb",
]


def adapt(notebook, _name: str):
    """Não reescreve a lógica acadêmica; os notebooks resolvem seus próprios caminhos."""
    return notebook


def execute(name: str):
    os.environ["PLN_AVALIACAO_DIR"] = str(ROOT)
    os.environ["POKEDEX_PROJECT_DIR"] = str(ROOT)
    source = NOTEBOOKS / name
    notebook = adapt(nbformat.read(source, as_version=4), name)
    EXECUTED.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(
        notebook,
        timeout=-1,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    output = EXECUTED / name
    nbformat.write(notebook, output)
    print(f"OK: {name} -> {output}")


def main():
    parser = argparse.ArgumentParser(description="Executa o pipeline acadêmico oficial")
    parser.add_argument("--from-step", choices=["00", "01", "02", "03"], default="00")
    args = parser.parse_args()
    start = int(args.from_step)
    for index, name in enumerate(ORDER):
        if index >= start:
            execute(name)


if __name__ == "__main__":
    main()
