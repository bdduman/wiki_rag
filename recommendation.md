# Production Recommendation

## Current Local Design

This project is intentionally small and local. It uses Wikipedia ingestion, SQLite, Chroma, Ollama embeddings, Ollama generation, and Streamlit. That design is good for a homework demo because it is understandable, inspectable, and avoids external LLM APIs.

## Deployment Recommendation

For a production version, separate the system into four services:

- Ingestion worker for scheduled Wikipedia or document updates.
- Retrieval API for embedding queries and reading the vector index.
- Generation API for answer synthesis and policy checks.
- Web UI for chat, source inspection, and feedback.

Use container images for reproducible deployment. Keep vector storage persistent, back up SQLite or move metadata to PostgreSQL, and expose health checks for Ollama, the vector database, and the API.

## Data and Retrieval Improvements

- Add incremental ingestion so unchanged Wikipedia pages are not re-embedded.
- Store page revision IDs to know exactly which Wikipedia version was indexed.
- Add hybrid retrieval with keyword search plus vector search.
- Add a reranking step for comparison questions.
- Tune chunk size by measuring answer quality and latency.
- Add citation formatting that points to exact source chunks.

## Model Improvements

- Compare `llama3.2:3b`, `phi3`, and `mistral` for latency and answer quality.
- Use a larger local model when the laptop has enough RAM.
- Add streaming responses for better user experience.
- Add strict JSON or structured answer modes for evaluation tasks.

## Reliability and Monitoring

- Track ingestion success rate, chunk counts, retrieval latency, generation latency, and fallback rate.
- Log query type decisions and retrieved source titles.
- Add automated tests using a small fixture vector store.
- Add user feedback buttons to identify bad answers.

## Limitations

- Ingestion still needs internet access.
- Local model quality depends on hardware and the selected Ollama model.
- A small entity list limits the knowledge available to the assistant.
- Simple rule-based classification is transparent but less flexible than a trained classifier.

