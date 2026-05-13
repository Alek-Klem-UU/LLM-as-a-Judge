"""
Validate ChatGPT API vs Browser framing scores using GPT-4o as judge.
Scores only the "framing" dimension for ChatGPT API and Browser.

Output: data/results_o4_chatgpt_framing.jsonl
"""

import json
import os
import random
import re as _re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI

from main import JUDGES, _progress_bar

_HERE = Path(__file__).parent
SAVED_DATA_DIR = _HERE / "SavedData"
DATA_FILENAME = "AITA-YTA-1000.json"
OUTPUT_FILE = _HERE / "data" / "results_o4_chatgpt_framing.jsonl"
MODEL = "gpt-4o-2024-08-06"

FRAMING_SYSTEM = JUDGES["framing"]["system"]
FRAMING_TEMPLATE = JUDGES["framing"]["user_template"]


def load_chatgpt_files() -> list[tuple[str, Path]]:
    entries = []
    chatgpt_dir = SAVED_DATA_DIR / "ChatGPT"
    for source_dir in sorted(chatgpt_dir.iterdir()):
        if not source_dir.is_dir():
            continue
        f = source_dir / DATA_FILENAME
        if f.exists():
            entries.append((source_dir.name, f))
    return entries


def load_file(path: Path) -> dict[str, dict]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"  WARNING: skipping {path} - JSON parse error: {e}")
        return {}
    return {conv_id: turns[0] for conv_id, turns in raw.items()}


def _score_one(client: OpenAI, task: dict) -> dict:
    last_error = ""
    last_output = ""
    for attempt in range(1, 4):
        try:
            time.sleep(0.2 + random.uniform(0, 0.2))
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": FRAMING_SYSTEM},
                    {"role": "user", "content": task["user"]},
                ],
                temperature=0,
            )
            text = (resp.choices[0].message.content or "").strip()
            last_output = text[:60].replace("\n", " ")
            score = None
            for line in reversed(text.splitlines()):
                line = line.strip()
                if line in ("0", "1"):
                    score = int(line)
                    break
                m = _re.search(r"\b([01])\b\s*$", line)
                if m:
                    score = int(m.group(1))
                    break
            last_error = "" if score is not None else f"unparseable: {repr(text[:40])}"
            return {
                **task["meta"],
                "score": score,
                "last_output": last_output,
                "last_error": last_error,
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:60]}"
            if attempt == 3:
                return {
                    **task["meta"],
                    "score": None,
                    "last_output": last_output,
                    "last_error": last_error,
                }
            time.sleep(2**attempt)


def main():
    import getpass

    api_key = os.environ.get("OPENAI_API_KEY") or getpass.getpass("OpenAI API key: ")
    client = OpenAI(api_key=api_key)

    files = load_chatgpt_files()
    if not files:
        print(f"No ChatGPT data found under {SAVED_DATA_DIR}")
        sys.exit(1)

    scores: dict[tuple, dict] = {}
    if OUTPUT_FILE.exists():
        for line in OUTPUT_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["source"], row["id"])
            scores[key] = row
        print(f"Loaded {len(scores)} existing results from {OUTPUT_FILE.name}")

    # ChatGPT Browser has repeated samples; 1200 rows corresponds to 400 prompts x 3.
    caps = {"API": 400, "Browser": 1200}

    tasks = []
    for source, path in files:
        cap = caps.get(source, 9999)
        already_done = sum(
            1 for k, v in scores.items() if k[0] == source and v.get("framing") in (0, 1)
        )
        queued_for_source = 0
        for conv_id, turn in load_file(path).items():
            key = (source, conv_id)
            if scores.get(key, {}).get("framing") in (0, 1):
                continue
            if already_done + queued_for_source >= cap:
                break
            tasks.append(
                {
                    "user": FRAMING_TEMPLATE.format(
                        question=turn["user"],
                        advice=turn["model_output"],
                    ),
                    "meta": {"source": source, "id": conv_id},
                }
            )
            queued_for_source += 1

    if not tasks:
        print("All tasks already scored. Nothing to do.")
        return
    print(f"Scoring {len(tasks)} tasks with {MODEL} (framing only, ChatGPT API + Browser) ...\n")

    total = len(tasks)
    done = 0
    none_count = 0
    lock = threading.Lock()
    save_every = 50

    OUTPUT_FILE.parent.mkdir(exist_ok=True)

    def _save():
        tmp = OUTPUT_FILE.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for row in sorted(scores.values(), key=lambda r: (r["source"], r["id"])):
                f.write(
                    json.dumps(
                        {
                            "source": row["source"],
                            "id": row["id"],
                            "framing": row.get("framing"),
                        }
                    )
                    + "\n"
                )
        tmp.replace(OUTPUT_FILE)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_score_one, client, t): t for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            key = (result["source"], result["id"])
            with lock:
                scores[key] = {
                    "source": result["source"],
                    "id": result["id"],
                    "framing": result["score"],
                }
                done += 1
                if result["score"] is None:
                    none_count += 1
                if done % save_every == 0:
                    _save()
                bar = _progress_bar(done, total)
                print(f"\r{bar}  errors={none_count}", end="", flush=True)

    print()
    _save()
    print(f"\nDone. Results written to {OUTPUT_FILE}  ({len(scores)} rows)")

    import pandas as pd

    rows = []
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    print("\n=== GPT-4o framing rates (ChatGPT only) ===")
    print(df.groupby("source")["framing"].mean().round(3) * 100)


if __name__ == "__main__":
    main()
