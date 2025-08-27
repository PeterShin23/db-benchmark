import os
import pandas as pd
from pathlib import Path
from beir import util
from beir.datasets.data_loader import GenericDataLoader

def fetch_fiqa(output_csv: str = "data/sample.csv", max_samples: int = 100000):
    # Check if file already exists
    if os.path.exists(output_csv):
        # Count rows in existing file
        df = pd.read_csv(output_csv)
        print(f"File already exists with {len(df)} rows → {output_csv}")
        return

    dataset = "fiqa"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = util.download_and_unzip(url, "data/")

    corpus, queries, qrels = GenericDataLoader(out_dir).load(split="test")

    # Flatten corpus into a dataframe
    rows = []
    for doc_id, doc in list(corpus.items())[:max_samples]:  # Limit to max_samples
        text = (doc.get("title") or "") + "\n" + (doc.get("text") or "")
        rows.append({
            "id": doc_id,          # unique id
            "doc_id": doc_id,      # keep original doc id (can be same here)
            "text": text.strip()
        })

    df = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df)} rows → {output_csv}")

if __name__ == "__main__":
    fetch_fiqa()
