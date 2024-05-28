from transformers import AutoTokenizer, AutoModel
import torch
import sqlite3
import numpy as np
from tqdm import tqdm
import h5py

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True).to(device)
model.eval()  # Set the model to evaluation mode

# Function to embed text snippets
def embed_texts(texts, tokenizer, model, device):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Taking the mean of the hidden states as the embedding

# Function to process a chunk of records
def process_chunk(cursor, tokenizer, model, start, chunk_size, device, pmid_dataset, embedding_dataset):
    cursor.execute(f"SELECT pmid, title, abstract FROM articles WHERE rowid IN (SELECT rowid FROM articles ORDER BY rowid LIMIT {chunk_size} OFFSET {start})")
    records = cursor.fetchall()

    if not records:
        return False  # No more records to process

    # Use title if abstract is None or empty
    pmids = []
    texts = []
    for pmid, title, abstract in records:
        if abstract is None or abstract.strip() == "":
            texts.append(title)
        else:
            texts.append(abstract)
        pmids.append(pmid)

    # Embed the texts
    embeddings = embed_texts(texts, tokenizer, model, device)

    # Save PMIDs and embeddings to HDF5 datasets
    start_idx = process_chunk.start_idx
    pmid_dataset[start_idx:start_idx + len(pmids)] = pmids
    embedding_dataset[start_idx:start_idx + len(pmids), :] = embeddings

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
chunk_size = 100

# Determine the embedding size (dimensionality)
dummy_embedding = embed_texts(["dummy text"], tokenizer, model, device)
embedding_dim = dummy_embedding.shape[1]

# Create HDF5 file and datasets for PMIDs and embeddings
with h5py.File('embeddings.h5', 'w') as h5f:
    pmid_dataset = h5f.create_dataset('pmids', (total_records,), dtype='int64')
    embedding_dataset = h5f.create_dataset('embeddings', (total_records, embedding_dim), dtype='float16')

    # Attach datasets and start index to the function
    process_chunk.pmid_dataset = pmid_dataset
    process_chunk.embedding_dataset = embedding_dataset
    process_chunk.start_idx = 0

    # Process records in chunks
    start = 0

    with tqdm(total=total_records, unit='record') as pbar:
        while True:
            if not process_chunk(cursor, tokenizer, model, start, chunk_size, device, pmid_dataset, embedding_dataset):
                break
            start += chunk_size
            pbar.update(chunk_size)

# Close the connection
conn.close()

print("Embedding process completed and saved to HDF5 file.")


