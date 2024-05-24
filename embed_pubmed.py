import sqlite3
import numpy as np
from nomic import embed

# Define constants
DB_FILE = "pubmed_data.db"
VECTOR_FILE = "embedded_vectors.npy"

# Connect to SQLite database
conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()

# Load Nomic's embedding model
embedder = Embedder()

# Fetch all abstracts from the database
cur.execute("SELECT pmid, abstract FROM articles WHERE abstract IS NOT NULL")
rows = cur.fetchall()

# Initialize a list to store the embeddings
embeddings = []
pmids = []

# Embed each abstract and store the result
for pmid, abstract in rows:
    output = embed.text(
        texts=[abstract],
        model='nomic-embed-text-v1.5',
        task_type="search_document",
        inference_mode='local',
        dimensionality=768,
    )

    embedding = np.array(output['embeddings']).squeeze()
    embeddings.append(embedding)
    pmids.append(pmid)

# Convert the list of embeddings to a NumPy array
embeddings_array = np.array(embeddings)
pmids_array = np.array(pmids)

# Save the embeddings and PMIDs to a binary file
np.save(VECTOR_FILE, embeddings_array)
np.save("pmids.npy", pmids_array)

# Close the database connection
conn.close()

print(f"Embeddings saved to {VECTOR_FILE} and PMIDs saved to pmids.npy")