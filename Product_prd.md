# Product PRD: Local Wikipedia RAG Assistant

## Purpose

Build a local ChatGPT-style assistant that answers questions about famous people and famous places using Wikipedia content stored on the user's laptop. The product demonstrates retrieval augmented generation with local data, local embeddings, local vector search, and a local language model.

## Users

The primary user is an instructor or student evaluating a small RAG system. The user should be able to follow the README, ingest data, start the app, ask questions, and inspect retrieved context without extra guidance.

## Goals

- Ingest at least 20 people and 20 places from Wikipedia.
- Store raw text and chunk metadata locally in SQLite.
- Store chunk embeddings locally in Chroma.
- Retrieve relevant chunks for person, place, and mixed queries.
- Generate grounded answers with Ollama `llama3.2:3b`.
- Provide a Streamlit chat interface.
- Make setup and operation clear enough for a public GitHub submission.

## Non-Goals

- This is not a production hosted chatbot.
- This does not use external LLM APIs.
- This does not attempt to index all of Wikipedia.
- This does not guarantee perfect factual coverage beyond the ingested pages.

## Functional Requirements

- The system downloads Wikipedia page extracts through the MediaWiki API.
- The system chunks long documents with overlap.
- The system embeds chunks with `nomic-embed-text` through Ollama.
- The system stores vectors in a single Chroma collection with metadata filters.
- The system classifies queries as `person`, `place`, `both`, or `unknown`.
- The system answers with only retrieved context.
- The system returns `I don't know.` when retrieval confidence is too low.
- The UI supports asking questions, viewing answers, showing retrieved context, and clearing chat history.

## Success Criteria

- A fresh user can install dependencies, pull models, run ingestion, and start the app by following README instructions.
- The app answers the assignment's example questions with relevant retrieved context.
- The failure case `Who is the president of Mars?` does not hallucinate a confident answer.
- The repository contains source code, README, PRD, recommendation document, requirements file, and reset/ingestion scripts.

## Technical Decisions

- Python is used because it has strong local ML and data tooling.
- Streamlit is used for a quick chat-style localhost UI.
- Ollama is used for local model serving.
- Chroma is used as the local vector database.
- SQLite is used as the durable local record of downloaded pages and chunks.
- One vector store with metadata filtering is used because it avoids duplicated retrieval code and still supports type-specific search.

