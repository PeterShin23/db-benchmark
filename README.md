# Vector Database Benchmark

A minimal, production-feeling demo that compares multiple vector databases for RAG. Everything runs locally with Docker.

## Learnings and Demo

<img width="1338" height="792" alt="image" src="https://github.com/user-attachments/assets/0a46f843-dffa-4b4f-9c05-1ac98c2fdf26" />


1. Weaviate and Qdrant have very similar performance across the board. Weaviate seems to win out by just a little more because it has a very fast indexing speed.
2. Redis is fast... But in terms of retrieval, it's just not as good. If you need something fast and pretty good, Redis is a great option, especially if it's already being used in the tech stack. But Weaviate and Qdrant just seem better.
3. Neo4j can be used as a vector database, it should really be used for GraphRAG though. Again, if it's already part of your stack, it might be great to just get up and running.
4. Pgvector has a similar feeling to Neo4j. Pgvector is a plugin and may be the best option is Postgres is part of your stack already.

My winner: Weaviate! 

## Features

- Compare performance of multiple vector databases:
  - Qdrant
  - Weaviate
  - Redis
  - Postgres (pgvector)
  - Neo4j
  - Looking to add Chroma, Pinecone, etc. in the future!
- Single embedding pass over a BEIR FiQA 2018 data
- Apache Parquet storage for efficient embedded data sharing between databases
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
   Open `ui/frontend/index.html` in your browser


## CSV Format Expectations

The input CSV should have at least these columns:
- `id`: Unique identifier for each row
- `text`: The text content to be embedded

Optionally:
- `doc_id`: Grouping identifier (defaults to `id` if not provided)

## How to Use

1. **Prepare your data**: Place your CSV file in the `data/` directory.

2. **Generate embeddings**: Run the embedding script to create a Parquet file (use e5):
   ```bash
   python embeddings/embed.py --input data/your_file.csv --output embeddings/your_file.parquet
   ```

3. **Index data**: Use the web UI to index the Parquet data into your chosen database.

4. **Search**: Enter natural language queries in the search section to find similar documents.

5. **Switch databases**: Use the dropdown to switch between different vector databases for comparison.

## Quick Description of Metrics
### 🔹 Recall@10

#### The fraction of all relevant documents that appear anywhere in the top-10 results.

Intuition: “Out of everything that should have been retrieved, how much did we actually surface within the first 10?”

Why it matters: High recall means the system rarely misses relevant documents, which is critical in domains where missing evidence is costly (e.g. finance, legal, compliance).

### 🔹 nDCG@10 (Normalized Discounted Cumulative Gain at 10)

#### The quality of ranking of the top-10 results, with higher weight for relevant documents appearing near the top.

Intuition: “Did we not only find the right answers, but also put them in the best order?”

Why it matters: Users don’t just want relevant documents—they want them ranked correctly so the best answers appear first. nDCG rewards good ordering and penalizes burying good results at the bottom.

### 🔹 MRR@10 (Mean Reciprocal Rank at 10)

#### The reciprocal of the rank of the first relevant result, averaged across queries.

Intuition: “How far does a user usually have to look before finding something useful?”

Why it matters: Reflects the “time-to-first-answer” experience. High MRR means the system usually puts a relevant doc right at the top, minimizing user effort.

## Note on Apache Parquet

Apache Parquet is a free, open-source columnar storage format that provides efficient data compression and encoding schemes. The embeddings are computed once and stored in Parquet format, which is then shared by all databases for fair comparison.

## Database Configuration

All database connection parameters can be configured through environment variables. See `.env.sample` for the required variables.

## Requirements

- Docker
- Python 3.8+
- Required Python packages (see `requirements.txt`)

## License

This project is licensed under the MIT License.
