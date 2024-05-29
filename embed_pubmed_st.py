from transformers import AutoTokenizer, AutoModel
import os
import torch
import sqlite3
import numpy as np
from tqdm import tqdm

from array_io import write_pair_to_file, read_pair_from_file

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
def process_chunk(cursor, tokenizer, model, start, chunk_size, device, file):
    cursor.execute(f"SELECT pmid, title, abstract FROM articles WHERE rowid IN (SELECT rowid FROM articles LIMIT {chunk_size} OFFSET {start})")
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

    for pid, emb in zip(pmids, embeddings):
        write_pair_to_file(file, pid, emb)

    process_chunk.start_idx += len(pmids)
    return True  # Records processed

# Function to get the saved offset
def get_saved_offset(offset_file):
    if os.path.exists(offset_file):
        with open(offset_file, 'r') as f:
            return int(f.read().strip())
    return 0

# Function to save the current offset
def save_offset(offset_file, offset):
    with open(offset_file, 'w') as f:
        f.write(str(offset))

# Connect to SQLite database
conn = sqlite3.connect('pubmed_data.db')
cursor = conn.cursor()

# Get the total number of records
#cursor.execute("SELECT COUNT(*) FROM articles")
#total_records = cursor.fetchone()[0]
total_records = 25_337_445 # just to speed it up

# Define chunk size
chunk_size = 10

# Determine the embedding size (dimensionality)
dummy_embedding = embed_texts(["dummy text"], tokenizer, model, device)
embedding_dim = dummy_embedding.shape[1]

offset_file = 'offset.txt'
start_offset = get_saved_offset(offset_file)
print(f"Resuming from offset: {start_offset}")

# Create HDF5 file and datasets for PMIDs and embeddings
with open("embeddings.bin", "ab") as file:

    # Attach datasets and start index to the function
    process_chunk.file = file
    process_chunk.start_idx = 0

    # Process records in chunks
    for start in tqdm(range(start_offset, total_records, chunk_size)):
        process_chunk(cursor, tokenizer, model, start, chunk_size, device, file)
        save_offset(offset_file, start + chunk_size)

# Close the connection
conn.close()

print("Embedding process completed and saved to a binary file.")


