"""
Tests for the multi-model benchmark's models file and discovery modes
(eval/benchmark.py). No LLM needed.

Run from the project root:
    python -m pytest tests/        (or: python tests/test_benchmark.py)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.catalog import CATALOG_PATH  # noqa: E402
from eval.benchmark import discovery_modes, load_models  # noqa: E402
from eval.eval_discovery import HARD_CATALOG_PATH  # noqa: E402


def test_models_file_takes_ids_with_or_without_an_endpoint():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "models.yaml"
        path.write_text("models:\n  - id: org/a\n  - id: org/b\n    api_base: https://b/v1\n  - org/c\n")
        assert load_models(path) == [{"id": "org/a", "api_base": None},
                                     {"id": "org/b", "api_base": "https://b/v1"},
                                     {"id": "org/c", "api_base": None}]
        for bad in ("models: []\n", "models:\n  - api_base: https://x/v1\n"):
            path.write_text(bad)
            try:
                load_models(path)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted {bad!r}")


def test_discovery_modes_go_from_easiest_to_the_drafted_catalog():
    draft = Path("catalog_draft.yaml")
    assert discovery_modes(None) == [(CATALOG_PATH, False), (CATALOG_PATH, True),
                                     (HARD_CATALOG_PATH, False), (HARD_CATALOG_PATH, True)]
    assert discovery_modes(draft)[-1] == (draft, True)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
