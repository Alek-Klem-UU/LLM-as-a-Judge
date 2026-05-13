"""
Prepare GPT-4o Batch API request files for scoring all saved conversations.

This mirrors main.py's judging setup, but writes OpenAI Batch API JSONL files
instead of making live requests. It creates one batch request file per answer
model, so the four output files can be uploaded/submitted independently.

Outputs:
    data/o4_full_batches/{Model}/requests.jsonl
    data/o4_full_batches/{Model}/manifest.json

Each request custom_id is:
    {Model}__{Source}__{ConversationId}__{Dimension}

That custom_id is enough to recover exactly which row each batch result belongs
to, even if a batch expires or only partially completes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from main import DEFAULT_DATA_FILENAME, JUDGES, SAVED_DATA_DIR, load_file


_HERE = Path(__file__).parent
OUT_DIR = _HERE / "data" / "o4_full_batches"
DATA_FILENAME = DEFAULT_DATA_FILENAME
JUDGE_MODEL = "gpt-4o-2024-08-06"
BATCH_ENDPOINT = "/v1/chat/completions"
DIMENSIONS = tuple(JUDGES.keys())


def _json_dumps(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _safe_custom_id(answer_model: str, source: str, conv_id: str, dim: str) -> str:
    return f"{answer_model}__{source}__{conv_id}__{dim}"


def _request_line(answer_model: str, source: str, conv_id: str, turn: dict, dim: str, cfg: dict) -> dict:
    return {
        "custom_id": _safe_custom_id(answer_model, source, conv_id, dim),
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": {
            "model": JUDGE_MODEL,
            "messages": [
                {"role": "system", "content": cfg["system"]},
                {
                    "role": "user",
                    "content": cfg["user_template"].format(
                        question=turn["user"],
                        advice=turn["model_output"],
                    ),
                },
            ],
            "temperature": 0,
            "max_completion_tokens": 1,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_custom_id(custom_id: str) -> tuple[str, str, str, str]:
    parts = custom_id.split("__")
    if len(parts) != 4:
        raise ValueError(f"Unexpected custom_id format: {custom_id!r}")
    answer_model, source, conv_id, dim = parts
    if dim not in JUDGES:
        raise ValueError(f"Unexpected dimension in custom_id: {custom_id!r}")
    return answer_model, source, conv_id, dim


def _parse_score(text: str) -> int | None:
    text = (text or "").strip()
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line in ("0", "1"):
            return int(line)
        match = re.search(r"\b([01])\b\s*$", line)
        if match:
            return int(match.group(1))
    return None


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def discover_answer_models() -> list[str]:
    if not SAVED_DATA_DIR.exists():
        raise FileNotFoundError(f"Saved data directory not found: {SAVED_DATA_DIR}")
    return sorted(p.name for p in SAVED_DATA_DIR.iterdir() if p.is_dir())


def source_files_for_model(answer_model: str) -> list[tuple[str, Path]]:
    model_dir = SAVED_DATA_DIR / answer_model
    entries: list[tuple[str, Path]] = []
    for source_dir in sorted(model_dir.iterdir()):
        if not source_dir.is_dir():
            continue
        path = source_dir / DATA_FILENAME
        if path.exists():
            entries.append((source_dir.name, path))
    return entries


def prepare_model_batch(answer_model: str) -> dict:
    model_out = OUT_DIR / answer_model
    requests_path = model_out / "requests.jsonl"
    manifest_path = model_out / "manifest.json"

    rows = []
    source_counts: dict[str, int] = {}
    dimension_counts = {dim: 0 for dim in JUDGES}

    for source, path in source_files_for_model(answer_model):
        conversations = load_file(path)
        source_counts[source] = len(conversations)
        for conv_id, turn in conversations.items():
            for dim, cfg in JUDGES.items():
                rows.append(_request_line(answer_model, source, conv_id, turn, dim, cfg))
                dimension_counts[dim] += 1

    text = "".join(_json_dumps(row) + "\n" for row in rows)
    _write_atomic(requests_path, text)

    manifest = {
        "answer_model": answer_model,
        "judge_model": JUDGE_MODEL,
        "endpoint": BATCH_ENDPOINT,
        "requests_file": str(requests_path),
        "request_count": len(rows),
        "source_conversation_counts": source_counts,
        "dimension_request_counts": dimension_counts,
        "sha256": _sha256(requests_path),
        "custom_id_format": "{Model}__{Source}__{ConversationId}__{Dimension}",
        "recovery_note": (
            "The requests file is deterministic and can be regenerated. Batch results can be "
            "merged by custom_id, so completed rows are recoverable even if a batch expires."
        ),
    }
    _write_atomic(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def merge_batch_output(answer_model: str, batch_output_path: Path) -> dict:
    """
    Merge one downloaded Batch API output JSONL into a per-model results file.

    Safe to rerun: existing rows are loaded first, then completed judgments from
    batch_output_path are merged by (model, source, id, dimension).
    """
    model_out = OUT_DIR / answer_model
    results_path = model_out / "results.jsonl"
    errors_path = model_out / "merge_errors.jsonl"

    rows: dict[tuple[str, str, str], dict] = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows[(row["model"], row["source"], row["id"])] = row

    merged = 0
    errors = []
    for line_no, line in enumerate(batch_output_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            custom_id = item["custom_id"]
            item_model, source, conv_id, dim = _parse_custom_id(custom_id)
            if item_model != answer_model:
                raise ValueError(f"Output for {item_model!r} cannot be merged into {answer_model!r}")

            response = item.get("response") or {}
            body = response.get("body") or {}
            message = body.get("choices", [{}])[0].get("message", {})
            score = _parse_score(message.get("content", ""))
            if score not in (0, 1):
                raise ValueError(f"Unparseable score for {custom_id!r}")

            key = (answer_model, source, conv_id)
            row = rows.setdefault(
                key,
                {"model": answer_model, "source": source, "id": conv_id},
            )
            row[dim] = score
            merged += 1
        except Exception as exc:
            errors.append(
                {
                    "line": line_no,
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw": line[:1000],
                }
            )

    ordered_rows = []
    for row in sorted(rows.values(), key=lambda r: (r["model"], r["source"], r["id"])):
        ordered_rows.append(
            {
                "model": row["model"],
                "source": row["source"],
                "id": row["id"],
                "validation": row.get("validation"),
                "indirectness": row.get("indirectness"),
                "framing": row.get("framing"),
            }
        )

    _write_atomic(results_path, "".join(_json_dumps(row) + "\n" for row in ordered_rows))
    _write_atomic(errors_path, "".join(_json_dumps(err) + "\n" for err in errors))

    complete_rows = sum(
        all(row.get(dim) in (0, 1) for dim in DIMENSIONS)
        for row in ordered_rows
    )
    return {
        "answer_model": answer_model,
        "batch_output": str(batch_output_path),
        "results_file": str(results_path),
        "errors_file": str(errors_path),
        "merged_scores": merged,
        "result_rows": len(ordered_rows),
        "complete_rows": complete_rows,
        "errors": len(errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and merge GPT-4o full-judge Batch API files.")
    subparsers = parser.add_subparsers(dest="command")

    prepare_parser = subparsers.add_parser("prepare", help="Prepare Batch API request JSONL files.")
    prepare_parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Answer model folders to prepare. Defaults to every folder under SavedData.",
    )

    merge_parser = subparsers.add_parser("merge", help="Merge a downloaded Batch API output JSONL file.")
    merge_parser.add_argument("answer_model", help="Answer model folder, e.g. ChatGPT or Claude.")
    merge_parser.add_argument("batch_output_path", type=Path, help="Downloaded Batch API output JSONL.")

    args = parser.parse_args()

    if args.command in (None, "prepare"):
        answer_models = args.models or discover_answer_models()
        manifests = [prepare_model_batch(model) for model in answer_models]

        print(f"Wrote {len(manifests)} batch request files under {OUT_DIR}")
        for manifest in manifests:
            print(
                f"- {manifest['answer_model']}: {manifest['request_count']} requests, "
                f"sha256={manifest['sha256'][:12]}..."
            )
        return

    if args.command == "merge":
        summary = merge_batch_output(args.answer_model, args.batch_output_path)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return


if __name__ == "__main__":
    main()
