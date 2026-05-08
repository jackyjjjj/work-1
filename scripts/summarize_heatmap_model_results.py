from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SETTING_RE = re.compile(
    r"^\|\s*(?P<setting>[^|]+?)\s*\|\s*(?P<episodes>\d+)\s*\|\s*"
    r"(?P<accuracy>[^|]+?)\s*\|\s*(?P<balanced>[^|]+?)\s*\|\s*(?P<macro_f1>[^|]+?)\s*\|"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize three heatmap model classification outputs.")
    parser.add_argument("--summary-jsonl", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_summary_rows(Path(args.summary_jsonl))
    output = Path(args.output_md)
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Three Heatmap Model Pseudo-Mask Classification Summary",
        "",
        "| Model | Status | Setting | Accuracy | Balanced Acc | Macro-F1 | Result | Message |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        result_path = Path(row.get("result_md") or "")
        metrics = parse_result_md(result_path) if result_path.exists() else []
        if not metrics:
            lines.append(
                "| {model} | {status} | - | - | - | - | `{result}` | {message} |".format(
                    model=escape_cell(row.get("model", "")),
                    status=escape_cell(row.get("status", "")),
                    result=escape_cell(str(result_path)) if str(result_path) else "-",
                    message=escape_cell(row.get("message", "")),
                )
            )
            continue
        for metric in metrics:
            lines.append(
                "| {model} | {status} | {setting} | {accuracy} | {balanced} | {macro_f1} | `{result}` | {message} |".format(
                    model=escape_cell(row.get("model", "")),
                    status=escape_cell(row.get("status", "")),
                    setting=escape_cell(metric["setting"]),
                    accuracy=escape_cell(metric["accuracy"]),
                    balanced=escape_cell(metric["balanced"]),
                    macro_f1=escape_cell(metric["macro_f1"]),
                    result=escape_cell(str(result_path)),
                    message=escape_cell(row.get("message", "")),
                )
            )

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote summary: {output}")


def load_summary_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_result_md(path: Path) -> list[dict[str, str]]:
    metrics = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        match = SETTING_RE.match(line)
        if not match:
            continue
        setting = match.group("setting").strip()
        if setting == "Setting":
            continue
        metrics.append(
            {
                "setting": setting,
                "accuracy": match.group("accuracy").strip(),
                "balanced": match.group("balanced").strip(),
                "macro_f1": match.group("macro_f1").strip(),
            }
        )
    return metrics


def escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()
