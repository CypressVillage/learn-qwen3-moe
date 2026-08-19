"""Validate lesson links and cumulative source checkpoints."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
CHECKPOINT_REF = re.compile(r"<!-- checkpoint: ([a-z0-9-]+) -->")
INLINE_CODE = re.compile(r"`+[^`\n]*`+")
GLOSSARY_REF = re.compile(r"\[\[([^\]\n]+)\]\]")
PRINCIPLE_OPEN = re.compile(r"^:::principle[ \t]+(\S.*)$")
PRINCIPLE_CLOSE = re.compile(r"^:::endprinciple[ \t]*$")
FENCE_LINE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def _is_line_subsequence(earlier: str, later: str) -> bool:
    earlier_lines = iter(earlier.splitlines(keepends=True))
    expected = next(earlier_lines, None)
    for line in later.splitlines(keepends=True):
        if line == expected:
            expected = next(earlier_lines, None)
    return expected is None


def _update_fence(line: str, fence: str | None) -> tuple[str | None, bool]:
    match = FENCE_LINE.match(line)
    if match is None:
        return fence, False
    marker, trailing = match.groups()
    if fence is None:
        if marker[0] == "`" and "`" in trailing:
            return fence, False
        return marker, True
    if marker[0] == fence[0] and len(marker) >= len(fence) and not trailing.strip(" "):
        return None, True
    return fence, False


def _without_fenced_code(markdown: str) -> str:
    fence: str | None = None
    prose: list[str] = []
    for line in markdown.splitlines():
        fence, changed = _update_fence(line, fence)
        prose.append("" if fence is not None or changed else line)
    return "\n".join(prose)


def _validate_glossary(lesson_paths: list[Path]) -> int:
    glossary = json.loads((ROOT / "lessons" / "glossary.json").read_text(encoding="utf-8"))
    assert isinstance(glossary, dict) and glossary, "glossary must be a non-empty object"
    for key, definition in glossary.items():
        assert isinstance(key, str) and key == key.strip() and key, f"invalid glossary key: {key!r}"
        assert isinstance(definition, dict), f"invalid glossary definition: {key}"
        assert set(definition) == {"full_name", "summary"}, f"invalid glossary fields: {key}"
        assert all(isinstance(value, str) and value.strip() for value in definition.values()), (
            f"empty glossary definition: {key}"
        )

    references = 0
    for lesson_path in lesson_paths:
        lesson = lesson_path.read_text(encoding="utf-8")
        prose = INLINE_CODE.sub("", _without_fenced_code(lesson))
        for key in GLOSSARY_REF.findall(prose):
            assert key == key.strip(), f"glossary reference has surrounding whitespace: {key!r}"
            assert key in glossary, f"unknown glossary term in {lesson_path.name}: {key}"
            references += 1
    assert references, "no glossary references found"
    return references


def _validate_principles(lesson_paths: list[Path]) -> int:
    blocks = 0
    for lesson_path in lesson_paths:
        active: tuple[str, int] | None = None
        body_has_content = False
        fence: str | None = None
        for line_number, line in enumerate(
            lesson_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            fence, changed = _update_fence(line, fence)
            if changed:
                if active is not None:
                    body_has_content = True
                continue
            if fence is not None:
                if active is not None and line.strip():
                    body_has_content = True
                continue

            opener = PRINCIPLE_OPEN.fullmatch(line)
            if opener:
                assert active is None, (
                    f"nested principle block in {lesson_path.name}:{line_number}"
                )
                active = (opener.group(1), line_number)
                body_has_content = False
                continue
            if PRINCIPLE_CLOSE.fullmatch(line):
                assert active is not None, (
                    f"orphan principle closer in {lesson_path.name}:{line_number}"
                )
                assert body_has_content, (
                    f"empty principle block in {lesson_path.name}:{active[1]}"
                )
                active = None
                blocks += 1
                continue
            if line.lstrip().startswith((":::principle", ":::endprinciple")):
                raise AssertionError(
                    f"malformed principle delimiter in {lesson_path.name}:{line_number}"
                )
            if active is not None:
                assert "<!-- checkpoint:" not in line, (
                    f"checkpoint inside principle block in {lesson_path.name}:{line_number}"
                )
                if line.strip():
                    body_has_content = True

        assert active is None, (
            f"unterminated principle block in {lesson_path.name}:{active[1]}"
            if active is not None
            else ""
        )
    assert blocks, "no principle blocks found"
    return blocks


def _validate_step(checkpoint_path: Path, require_current_source: bool) -> int:
    checkpoint_document = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoints = checkpoint_document["checkpoints"]
    step = checkpoint_path.stem
    checkpoint_ids: set[str] = set()
    checkpoint_order: list[str] = []
    previous_files: dict[str, str] | None = None

    for checkpoint in checkpoints:
        checkpoint_id = checkpoint["id"]
        assert checkpoint["step"] == step
        assert checkpoint_id not in checkpoint_ids, f"duplicate checkpoint: {checkpoint_id}"
        checkpoint_ids.add(checkpoint_id)
        checkpoint_order.append(checkpoint_id)
        files = {
            item["path"]: item["content"]
            for item in checkpoint["repository_snapshot"]["files"]
        }
        if files:
            assert checkpoint["active_file"] in files
        else:
            assert checkpoint["active_file"] == ""
        for relative_path, snapshot in files.items():
            path = Path(relative_path)
            assert not path.is_absolute() and ".." not in path.parts
            source = (ROOT / path).read_text(encoding="utf-8")
            assert _is_line_subsequence(snapshot, source), (
                f"checkpoint {checkpoint_id} is not a line-preserving subset of {relative_path}"
            )
        if previous_files is not None:
            assert previous_files.keys() <= files.keys(), (
                f"checkpoint {checkpoint_id} removes files"
            )
            for relative_path, previous_snapshot in previous_files.items():
                assert _is_line_subsequence(previous_snapshot, files[relative_path]), (
                    f"checkpoint {checkpoint_id} removes or rewrites lines in {relative_path}"
                )
        previous_files = files
        focus = checkpoint["focus_range"]
        active_file = checkpoint["active_file"]
        lines = files[active_file].splitlines() if active_file else []
        if focus["start"] == 0:
            assert focus == {"start": 0, "end": 0, "symbol": ""}
            if active_file:
                assert files[active_file] == ""
        else:
            assert 1 <= focus["start"] <= focus["end"] <= len(lines)
            assert focus["symbol"].split(".")[-1] in "\n".join(
                lines[focus["start"] - 1 : focus["end"]]
            )

    assert previous_files is not None
    if require_current_source:
        for relative_path, snapshot in previous_files.items():
            assert (ROOT / relative_path).read_text(encoding="utf-8") == snapshot, (
                f"final checkpoint is stale: {relative_path}"
            )

    lesson_paths = sorted((ROOT / "lessons").glob(f"{step}-*.md"))
    assert len(lesson_paths) == 1, f"expected one lesson for {step}"
    lesson = lesson_paths[0].read_text(encoding="utf-8")
    assert CHECKPOINT_REF.findall(lesson) == checkpoint_order
    return len(checkpoint_ids)


def main() -> None:
    checkpoint_paths = sorted((ROOT / "lessons" / "checkpoints").glob("step*.json"))
    assert checkpoint_paths, "no course checkpoints found"
    total = sum(
        _validate_step(path, require_current_source=path == checkpoint_paths[-1])
        for path in checkpoint_paths
    )
    lesson_paths = sorted((ROOT / "lessons").glob("step*.md"))
    glossary_references = _validate_glossary(lesson_paths)
    principle_blocks = _validate_principles(lesson_paths)
    print(
        f"validated {total} checkpoints across {len(checkpoint_paths)} lessons "
        f"with {glossary_references} glossary references and {principle_blocks} principle blocks"
    )


if __name__ == "__main__":
    main()
