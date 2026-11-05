"""The README usage example must run."""

from __future__ import annotations

import re
from pathlib import Path


def test_readme_example_runs(capsys) -> None:
    text = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert blocks
    exec(compile(blocks[0], "README.md", "exec"), {})  # noqa: S102
    out = capsys.readouterr().out
    assert "['small', 'large']" in out or "['large', 'small']" in out
    assert "attempts=1" in out or " 1" in out
