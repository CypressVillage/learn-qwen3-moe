"""Validate links between the Step 01 lesson and source checkpoints."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
CHECKPOINT_REF = re.compile(r"<!-- checkpoint: ([a-z0-9-]+) -->")


def _is_line_subsequence(earlier: str, later: str) -> bool:
    earlier_lines = iter(earlier.splitlines(keepends=True))
    expected = next(earlier_lines, None)
    for line in later.splitlines(keepends=True):
        if line == expected:
            expected = next(earlier_lines, None)
    return expected is None


def main() -> None:
    checkpoint_document = json.loads(
        (ROOT / "lessons" / "checkpoints" / "step01.json").read_text(
            encoding="utf-8"
        )
    )
    checkpoints = checkpoint_document["checkpoints"]
    checkpoint_ids: set[str] = set()
    checkpoint_order: list[str] = []
    previous_files: dict[str, str] | None = None

    for checkpoint in checkpoints:
        checkpoint_id = checkpoint["id"]
        assert checkpoint_id not in checkpoint_ids, f"duplicate checkpoint: {checkpoint_id}"
        checkpoint_ids.add(checkpoint_id)
        checkpoint_order.append(checkpoint_id)
        files = {
            item["path"]: item["content"]
            for item in checkpoint["repository_snapshot"]["files"]
        }
        assert checkpoint["active_file"] in files
        for relative_path, snapshot in files.items():
            path = Path(relative_path)
            assert not path.is_absolute() and ".." not in path.parts
            source = (ROOT / path).read_text(encoding="utf-8")
            assert _is_line_subsequence(snapshot, source), (
                f"checkpoint {checkpoint_id} is not a line-preserving subset of {relative_path}"
            )
        if previous_files is not None:
            assert files.keys() == previous_files.keys()
            for relative_path, previous_snapshot in previous_files.items():
                assert _is_line_subsequence(previous_snapshot, files[relative_path]), (
                    f"checkpoint {checkpoint_id} removes or rewrites lines in {relative_path}"
                )
        previous_files = files
        focus = checkpoint["focus_range"]
        lines = files[checkpoint["active_file"]].splitlines()
        if focus["start"] == 0:
            assert focus == {"start": 0, "end": 0, "symbol": ""}
            assert files[checkpoint["active_file"]] == ""
        else:
            assert 1 <= focus["start"] <= focus["end"] <= len(lines)
            assert focus["symbol"].split(".")[-1] in "\n".join(
                lines[focus["start"] - 1 : focus["end"]]
            )

    assert previous_files is not None
    for relative_path, snapshot in previous_files.items():
        assert (ROOT / relative_path).read_text(encoding="utf-8") == snapshot, (
            f"final checkpoint is stale: {relative_path}"
        )

    lesson = (ROOT / "lessons" / "step01-overview-config-weights.md").read_text(
        encoding="utf-8"
    )
    assert CHECKPOINT_REF.findall(lesson) == checkpoint_order
    print(f"validated {len(checkpoint_ids)} checkpoints and Step 01 lesson links")


if __name__ == "__main__":
    main()
