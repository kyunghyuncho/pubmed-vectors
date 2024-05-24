import sqlite3
import numpy as np
from nomic import embed
from tqdm import tqdm

# Define constants
DB_FILE = "pubmed_data_sample.db"
VECTOR_FILE = "embedded_vectors.npy"
PMID_FILE = "pmids.npy"

def embed_pubmed(db_file: str=DB_FILE,
                 vector_file: str=VECTOR_FILE,
                 pmid_file: str=PMID_FILE,
                 dtype: str="float16"):
    # Connect to SQLite database
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    # Fetch all abstracts from the database
    cur.execute("SELECT pmid, abstract FROM articles WHERE abstract IS NOT NULL")
    rows = cur.fetchall()

    # Initialize a list to store the embeddings
    embeddings = []
    pmids = []

    # Embed each abstract and store the result
    for pmid, abstract in tqdm(rows, desc="Embedding abstracts", unit="abstract", ncols=100):
        output = embed.text(
            texts=[abstract],
            model='nomic-embed-text-v1.5',
            task_type="search_document",
            inference_mode='local',
            dimensionality=768,
        )

        embedding = np.array(output['embeddings'], dtype=dtype).squeeze()
        embeddings.append(embedding)
        pmids.append(pmid)

    # Convert the list of embeddings to a NumPy array
    embeddings_array = np.array(embeddings, dtype=dtype)
    pmids_array = np.array(pmids)

    # Save the embeddings and PMIDs to a binary file
    np.save(vector_file, embeddings_array)
    np.save(pmid_file, pmids_array)

    # Close the database connection
    conn.close()

    print(f"Embeddings saved to {vector_file} and PMIDs saved to {pmid_file}")

if __name__ == "__main__":
    embed_pubmed()