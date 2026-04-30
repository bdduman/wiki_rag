# Local Wikipedia RAG Assistant

This project is a fully local ChatGPT-style assistant for questions about famous people and famous places. It downloads selected Wikipedia pages, chunks the text, stores embeddings in Chroma, keeps source data in SQLite, retrieves relevant context, and answers with a local Ollama model.

Demo video link: TODO

## Features

- Ingests 20 famous people and 20 famous places from Wikipedia.
- Uses local embeddings with Ollama `nomic-embed-text`.
- Uses local generation with Ollama `llama3.2:3b`.
- Stores raw pages and chunk metadata in SQLite.
- Stores vectors in one Chroma collection with `type=person|place` metadata.
- Provides a Streamlit chat UI with optional retrieved context.
- Returns `I don't know.` when retrieval does not find enough context.

## Requirements

- Python 3.11 or newer
- Internet access for the first ingestion step
- Ollama installed from https://ollama.com

## Start From A Fresh Clone

Clone the repository and enter the project folder:

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

If you are using the `github_upload` folder as the uploadable project, enter that folder instead:

```bash
cd github_upload
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Pull the required local Ollama models:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Start Ollama in a separate terminal if it is not already running:

```bash
ollama serve
```

Optional: copy the example environment file if you want to customize model names or storage paths:

```bash
cp .env.example .env
```

## Ingest Wikipedia Data

Run the ingestion step before starting the app. This downloads Wikipedia pages, creates embeddings with Ollama, and writes the local SQLite and Chroma files.

```bash
python scripts/ingest.py
```

This creates:

- `data/wiki_rag.sqlite3`
- `data/chroma/`

These generated files are intentionally not committed to Git, so every fresh clone needs to run ingestion once.

To reset local data and ingest again:

```bash
python scripts/reset_data.py
python scripts/ingest.py
```

## Run The App

With the virtual environment active and Ollama running:

```bash
python -m streamlit run app.py
```

You can also use this if your environment's `streamlit` command is valid:

```bash
streamlit run app.py
```

Open the localhost URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## Quick Verification

In another terminal, you can confirm Ollama and the app are reachable:

```bash
ollama list
curl -I http://localhost:8501
```

## Example Queries

- Who was Albert Einstein and what is he known for?
- What did Marie Curie discover?
- Why is Nikola Tesla famous?
- Compare Lionel Messi and Cristiano Ronaldo.
- Where is the Eiffel Tower located?
- Why is the Great Wall of China important?
- What was the Colosseum used for?
- Which famous place is located in Turkey?
- Compare the Eiffel Tower and the Statue of Liberty.
- Who is the president of Mars?

## Run Tests

```bash
pytest
```

## Troubleshooting

If ingestion or chat answers fail with an Ollama connection error, make sure `ollama serve` is running and that `ollama list` shows both required models.

If Streamlit does not start with `streamlit run app.py`, use:

```bash
python -m streamlit run app.py
```

If the sidebar says no local index was found, run:

```bash
python scripts/ingest.py
```

## Architecture

The project uses one vector store instead of separate stores for people and places. Each chunk has metadata containing `type`, `title`, `source_url`, and `chunk_index`. This keeps retrieval simple while still allowing person-only, place-only, and mixed queries through metadata filters.

The query classifier is deliberately lightweight. It checks known entity names and simple keyword hints, then chooses `person`, `place`, `both`, or `unknown`. Person queries search only person chunks, place queries search only place chunks, mixed queries search both, and unknown queries search all chunks.

The generator prompt tells the local model to answer only from retrieved context and to say `I don't know.` when the context does not contain the answer.

## Demo Video Outline

1. Show the repository and explain that the whole system runs locally.
2. Show Ollama models with `ollama list`.
3. Run `python scripts/ingest.py` or show a completed ingestion run.
4. Start the app with `python -m streamlit run app.py`.
5. Ask one person question, one place question, one comparison question, and one failure case.
6. Explain the design choice: one Chroma collection with metadata filtering.
7. Mention limitations: Wikipedia ingestion needs internet, local models are slower than hosted APIs, and retrieval quality depends on chunking and embeddings.
