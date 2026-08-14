import ast
import re
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
COURSE_DIR = ROOT / "docs" / "course"
SOURCE_BLOCK = re.compile(
    r"<!-- source-sync: ([^:]+)::([^ ]+) -->\n"
    r"```python\n(.*?)\n```",
    re.DOTALL,
)


def extract_symbol(source_path: Path, qualified_name: str) -> str:
    source = source_path.read_text()
    node: ast.AST = ast.parse(source)
    current_body = node.body
    target: ast.AST | None = None

    for name in qualified_name.split("."):
        target = next(
            (
                item
                for item in current_body
                if isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name == name
            ),
            None,
        )
        if target is None:
            raise AssertionError(f"{qualified_name} not found in {source_path}")
        current_body = target.body

    assert target is not None
    lines = source.splitlines()
    return textwrap.dedent(
        "\n".join(lines[target.lineno - 1 : target.end_lineno])
    )


@pytest.mark.parametrize(
    "chapter",
    sorted(COURSE_DIR.glob("[0-9][0-9]-*.md")),
    ids=lambda path: path.stem,
)
def test_course_embedded_source_matches_real_implementation(chapter: Path):
    matches = SOURCE_BLOCK.findall(chapter.read_text())

    assert matches, f"{chapter} must embed at least one synchronized source block"
    for relative_path, qualified_name, embedded in matches:
        actual = extract_symbol(ROOT / relative_path, qualified_name)
        assert embedded == actual, (
            f"embedded source for {relative_path}::{qualified_name} is stale in {chapter}"
        )
