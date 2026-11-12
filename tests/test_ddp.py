import json
import subprocess
import sys
from pathlib import Path

from steerable_timm import CommandLog


def test_two_ranks_apply_the_same_version_at_the_same_step(tmp_path):
    script = Path(__file__).parent / "ddp_run.py"
    subprocess.run([sys.executable, str(script), str(tmp_path)], check=True, timeout=600, cwd=script.parent)
    a, b = (json.loads((tmp_path / f"rank{r}.json").read_text()) for r in (0, 1))
    assert a["trace"] == b["trace"]
    assert a["checksum"] == b["checksum"]
    assert a["sample"] != b["sample"]
    versions = [row[1] for row in a["trace"]]
    assert versions == sorted(versions) and versions[0] == 0 and versions[-1] == 4
    by_step = {row[0]: row for row in a["trace"]}
    assert by_step[3][5] < by_step[6][5] < by_step[2][5]
    assert by_step[8][4] == 8 and by_step[10][4] == 4
    assert by_step[11][6] == "AdamW" and by_step[12][6] == "SGD"
    assert {row[3] for row in a["trace"][9:]} == {"train", "hard"}
    statuses = [r["status"] for r in CommandLog(tmp_path).log()]
    assert statuses == ["applied"] * 4 + ["rejected"]
