# Graph RAG with Kuzu, DSPy and marimo

> CS-E4780 Scalable Systems and Data Management Course Project2


## Setup

put `project.env` with `GEMINI_API_KEY` in the workspace 

then run the following commands

```bash
uv sync

uv run python create_nobel_api_graph.py
uv run python complete_missing_relationships.py
```


## App

```bash
uv run marimo run graph_rag.py
```

## Test with exemplars

```bash
uv run python test_all_78_exemplars.py
```

## Results

- final_test_78_exemplars.json
- cache_comparison_results.json

