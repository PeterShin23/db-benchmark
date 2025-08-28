import argparse
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
import os

def embed_csv_to_parquet(input_path, output_path, model_name='all-MiniLM-L6-v2', use_float16=False):
    # Load the CSV file
    df = pd.read_csv(input_path)
    
    # Check if doc_id column exists, if not create it from id
    if 'doc_id' not in df.columns:
        df['doc_id'] = df['id']
    
    # Handle NaN values in text column
    df['text'] = df['text'].fillna('')
    
    # Initialize the sentence transformer model
    model = SentenceTransformer(model_name)
    
    # Generate embeddings
    embeddings = model.encode(df['text'].tolist(), normalize_embeddings=True)
    
    # Convert to appropriate dtype
    if use_float16:
        embeddings = embeddings.astype(np.float16)
    else:
        embeddings = embeddings.astype(np.float32)
    
    # Add embeddings to dataframe
    df['emb'] = embeddings.tolist()
    
    # Convert to PyArrow table
    table = pa.Table.from_pandas(df)
    
    # Write to Parquet file
    pq.write_table(table, output_path)
    
    print(f"Embeddings saved to {output_path}")
    print(f"Number of rows: {len(df)}")
    print(f"Embedding dimension: {len(embeddings[0])}")

def main():
    parser = argparse.ArgumentParser(description='Generate embeddings from CSV and save to Parquet')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', required=True, help='Output Parquet file path')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Sentence transformer model name')
    parser.add_argument('--float16', action='store_true', help='Use float16 for embeddings')
    
    args = parser.parse_args()
    
    embed_csv_to_parquet(args.input, args.output, args.model, args.float16)

if __name__ == '__main__':
    main()
