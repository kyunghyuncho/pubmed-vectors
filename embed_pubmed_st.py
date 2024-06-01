from transformers import AutoTokenizer, AutoModel
import os
import torch
import sqlite3
import numpy as np
from tqdm import tqdm

from array_io import write_pair_to_file, read_pair_from_file

# file paths
DB_FILE="pubmed_data.db"
SAVE_FILE="pubmed_embeddings.bin"

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True).to(device)
model.eval()  # Set the model to evaluation mode

# Function to embed text snippets
def embed_texts(texts, tokenizer, model, device):
    with torch.no_grad():
        inputs = tokenizer(texts, 
                           padding=True, 
                           truncation=True, 
                           return_tensors="pt").to(device)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Taking the mean of the hidden states as the embedding

    return embeddings

# Function to process a chunk of records
def process_chunk(cursor, tokenizer, model, start, chunk_size, device, file):
    records = cursor.fetchmany(chunk_size)

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
        pmids.append(int(pmid))

    # Embed the texts
    embeddings = embed_texts(texts, tokenizer, model, device)

    # Save PMIDs and embeddings to memory-mapped arrays
    start_idx = process_chunk.start_idx
    process_chunk.pmid_mmap[start_idx:start_idx + len(pmids)] = pmids
    process_chunk.embedding_mmap[start_idx:start_idx + len(pmids), :] = embeddings

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
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Get the total number of records
cursor.execute("SELECT COUNT(*) FROM articles")
total_records = cursor.fetchone()[0]
#total_records = 25_337_445 # just to speed it up

# Define chunk size
chunk_size = 25

# Determine the embedding size (dimensionality)
dummy_embedding = embed_texts(["dummy text"], tokenizer, model, device)
embedding_dim = dummy_embedding.shape[1]
print(f"Embedding dim {embedding_dim}")

# Create memory-mapped files for PMIDs and embeddings
pmid_mmap = np.memmap('pmids_test.dat', dtype='int64', mode='w+', shape=(total_records,))
embedding_mmap = np.memmap('embeddings_test.dat', dtype='float32', mode='w+', shape=(total_records, embedding_dim))

start_offset = 0
cursor = cursor.execute(f"SELECT pmid, title, abstract FROM articles")

# Attach memory-mapped arrays and start index to the function
process_chunk.pmid_mmap = pmid_mmap
process_chunk.embedding_mmap = embedding_mmap
process_chunk.start_idx = 0

with open(SAVE_FILE, "ab") as file:
    # Process records in chunks
    for start in tqdm(range(start_offset, total_records, chunk_size)):
        process_chunk(cursor, tokenizer, model, start, chunk_size, device, file)
        #save_offset(offset_file, start + chunk_size)

# Flush changes to disk
pmid_mmap.flush()
embedding_mmap.flush()

# Close the connection
conn.close()

print("Embedding process completed and saved to a binary file.")


