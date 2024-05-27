from sentence_transformers import SentenceTransformer
import sqlite3
import numpy as np
from tqdm import tqdm

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')
model.eval()  # Set the model to evaluation mode

# Function to embed text snippets
def embed_texts(texts, model):
    return model.encode(texts, convert_to_tensor=False)

# Function to process a chunk of records
def process_chunk(cursor, model, start, chunk_size):
    cursor.execute(f"SELECT pmid, abstract FROM articles LIMIT {chunk_size} OFFSET {start}")
    records = cursor.fetchall()

    if not records:
        return False  # No more records to process

    # Filter out records with None abstracts
    records = [record for record in records if record[1] is not None]

    if not records:
        return True  # Skip this chunk if no valid records are found

    pmids = np.array([row[0] for row in records])
    abstracts = [row[1] for row in records]

    # Embed the abstracts
    embeddings = embed_texts(abstracts, model)

    # Save PMIDs and embeddings to memory-mapped arrays
    start_idx = process_chunk.start_idx
    process_chunk.pmid_mmap[start_idx:start_idx + len(pmids)] = pmids
    process_chunk.embedding_mmap[start_idx:start_idx + len(pmids), :] = embeddings

    process_chunk.start_idx += len(pmids)
    return True  # Records processed

# Connect to SQLite database
conn = sqlite3.connect('pubmed_data.db')
cursor = conn.cursor()

# Get the total number of records
#cursor.execute("SELECT COUNT(*) FROM articles")
#total_records = cursor.fetchone()[0]
total_records = 25_337_445 # just to speed it up

# Define chunk size
chunk_size = 100000

# Determine the embedding size (dimensionality)
dummy_embedding = embed_texts(["dummy text"], model)
embedding_dim = dummy_embedding.shape[1]

# Create memory-mapped files for PMIDs and embeddings
pmid_mmap = np.memmap('pmids.dat', dtype='int64', mode='w+', shape=(total_records,))
embedding_mmap = np.memmap('embeddings.dat', dtype='float16', mode='w+', shape=(total_records, embedding_dim))

# Attach memory-mapped arrays and start index to the function
process_chunk.pmid_mmap = pmid_mmap
process_chunk.embedding_mmap = embedding_mmap
process_chunk.start_idx = 0

# Process records in chunks
start = 0

with tqdm(total=total_records, unit='record') as pbar:
    while True:
        if not process_chunk(cursor, model, start, chunk_size):
            break
        start += chunk_size
        pbar.update(chunk_size)

# Flush changes to disk
pmid_mmap.flush()
embedding_mmap.flush()

# Close the connection
conn.close()

print("Embedding process completed and saved to memory-mapped arrays.")


