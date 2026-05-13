# LLM-as-a-Judge

Batch scoring pipeline for judging LLM advice responses on three sycophancy-related dimensions:

- `validation`
- `indirectness`
- `framing`

The current workflow uses `gpt-4o-2024-11-20` through the OpenAI Batch API. Older realtime/development scripts are kept in `old_judge/`.

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a local `.env` file:

```text
OPENAI_API_KEY=your_key_here
```

`.env`, `SavedData/`, and generated files under `data/` are ignored by git.

## Input Data

Expected layout:

```text
SavedData/
  ChatGPT/
    API/*.json
    Browser/*.json
  Claude/
  DeepSeek/
  Gemini/
```

Each input JSON should contain conversation IDs mapped to turns with `user` and `model_output` fields.

## Commands

Create one batch request file per model:

```powershell
python main.py prepare
```

Create a small smoke-test batch:

```powershell
python main.py prepare-test
```

Submit the smoke-test batch:

```powershell
python main.py submit-test
python main.py status-test
python main.py download-test
python main.py merge --outputs data\batches\gpt-4o-2024-11-20\_test_20_prompts\output.jsonl
```

Submit the full batches:

```powershell
python main.py submit
python main.py status
python main.py download
python main.py merge
```

You can limit full-batch commands to specific models:

```powershell
python main.py submit --models ChatGPT Claude
```

## Outputs

Generated batch files are written under:

```text
data/batches/gpt-4o-2024-11-20/
```

Merged scores are written to:

```text
data/results_gpt-4o-2024-11-20.jsonl
```

Merge errors, if any, are written to:

```text
data/merge_errors_gpt-4o-2024-11-20.jsonl
```
