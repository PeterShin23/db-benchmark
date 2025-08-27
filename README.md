# Vector Database Benchmark

A minimal, production-feeling demo that compares multiple vector databases (and an optional graph DB) for RAG/GraphRAG. Everything runs locally with Docker.

## Features

- Compare performance of multiple vector databases:
  - Qdrant
  - Weaviate
  - Redis
  - pgvector
  - Neo4j (optional, for GraphRAG)
- Single embedding pass over a Kaggle CSV dataset
- Apache Parquet storage for efficient data sharing between databases
- Simple web UI for indexing, searching, and clearing databases
- All services Dockerized for easy setup

## Quickstart

1. Set up the virtual environment and install dependencies:
   ```bash
   make venv
   ```

2. Start all services:
   ```bash
   docker compose -f scripts/docker-compose.yml up -d
   ```

3. Download data (Using BEIR FiQA 2018 data):
   ```bash
   make loader
   ```

4. Create embeddings from a CSV dataset:
   ```bash
   make embed
   ```

5. Start the backend server:
   ```bash
   make run
   ```

6. Open the frontend UI:
   Open `ui/frontend/app.html` in your browser


## CSV Format Expectations

The input CSV should have at least these columns:
- `id`: Unique identifier for each row
- `text`: The text content to be embedded

Optionally:
- `doc_id`: Grouping identifier (defaults to `id` if not provided)

## How to Use

1. **Prepare your data**: Place your CSV file in the `data/` directory.

2. **Generate embeddings**: Run the embedding script to create a Parquet file:
   ```bash
   python embeddings/embed.py --input data/your_file.csv --output embeddings/your_file.parquet
   ```

3. **Index data**: Use the web UI to index the Parquet data into your chosen database.

4. **Search**: Enter natural language queries in the search section to find similar documents.

5. **Switch databases**: Use the dropdown to switch between different vector databases for comparison.

## Note on Apache Parquet

Apache Parquet is a free, open-source columnar storage format that provides efficient data compression and encoding schemes. The embeddings are computed once and stored in Parquet format, which is then shared by all databases for fair comparison.

## Database Configuration

All database connection parameters can be configured through environment variables. See `.env.sample` for the required variables.

## Requirements

- Docker
- Python 3.8+
- Required Python packages (see `requirements.txt`)

## License

This project is licensed under the Apache-2.0 License.
