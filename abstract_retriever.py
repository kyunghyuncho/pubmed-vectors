import numpy as np
import os
from transformers import AutoTokenizer, AutoModel
import torch
import sqlite3
from tqdm import tqdm

CHUNK_SIZE=10_000

class AbstractRetriever:
    def __init__(self, vectors_file, ids_file, db_file, 
                 model_name="nomic-ai/nomic-embed-text-v1.5",
                 chunk_size=CHUNK_SIZE):
        self.vectors_file = vectors_file
        self.ids_file = ids_file
        self.db_file = db_file
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.dimension = self._infer_dimension()
        self.dtype_vectors = 'float32'
        self.dtype_ids = 'int64'
        self.num_rows = self._get_num_rows()
        self.doc_vectors_memmap = np.memmap(vectors_file, dtype=self.dtype_vectors, mode='r', shape=(self.num_rows, self.dimension))
        self.doc_ids_memmap = np.memmap(ids_file, dtype=self.dtype_ids, mode='r', shape=(self.num_rows,))
        self.connection = self._connect_db()
    
    def _infer_dimension(self):
        dummy_query = "dummy"
        inputs = self.tokenizer(dummy_query, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        dummy_vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        dimension = dummy_vector.shape[0]
        return dimension
    
    def _get_num_rows(self):
        vectors_size = os.path.getsize(self.vectors_file)
        ids_size = os.path.getsize(self.ids_file)
        num_rows_vectors = vectors_size // (np.dtype(self.dtype_vectors).itemsize * self.dimension)
        num_rows_ids = ids_size // np.dtype(self.dtype_ids).itemsize
        assert num_rows_vectors == num_rows_ids, f"Number of rows in vectors {num_rows_vectors} and ids {num_rows_ids} files must match"
        return num_rows_vectors
    
    def _connect_db(self):
        connection = sqlite3.connect(self.db_file)
        return connection
    
    def _fetch_document_info(self, pmids):
        cursor = self.connection.cursor()

        query = "SELECT pmid, title, authors, abstract, publication_year FROM articles WHERE pmid IN ("
        query += ",".join([str(id) for id in pmids])
        query += ")"

        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()

        # Convert rows to list of dictionaries
        documents = []
        for row in rows:
            documents.append({
                'pmid': row[0],
                'title': row[1],
                'authors': row[2],
                'abstract': row[3],
                'publication_year': row[4]
            })
        return documents
    
    def embed_query(self, query):
        inputs = self.tokenizer("search_query: " + query, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        query_vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return query_vector.astype('float32')
    
    def search(self, query, top_k=10):
        query_vector = self.embed_query(query)
        query_vector = query_vector / ((query_vector ** 2).sum() ** 0.5)
        
        # compute the cosine distance between all rows in self.doc_vectors_memmap and query.
        # do it in chunks to avoid memory errors.
        # keep only top-k distances and corresponding indices.
        # do not keep all distances to save memory.
        distances = np.zeros(top_k, dtype='float32')
        indices = np.zeros(top_k, dtype='int64')
        for i in tqdm(range(0, self.num_rows, self.chunk_size)):
            end = min(i + self.chunk_size, self.num_rows)
            chunk = self.doc_vectors_memmap[i:end]

            # use cosine similarity to compute distances.
            # TODO: it's probably a better idea to normalize all vectors in advance.
            norms = (chunk ** 2).sum(axis=1) ** 0.5
            chunk_distances = (chunk * query_vector[None, :]).sum(axis=1)
            chunk_distances /= norms
            chunk_indices = np.arange(i, end)
            for j in range(top_k):
                min_index = np.argmin(distances)
                if chunk_distances[j] > distances[min_index]:
                    distances[min_index] = chunk_distances[j]
                    indices[min_index] = chunk_indices[j]

        pmids = self.doc_ids_memmap[indices]
        documents = self._fetch_document_info(pmids)

        return pmids, distances, documents

# Example usage:
if __name__ == "__main__":
    vectors_file = 'embeddings_test.dat'
    ids_file = 'pmids_test.dat'
    db_file = 'pubmed_db.db'

    retriever = AbstractRetriever(vectors_file, ids_file, db_file)
    query = "example search query"
    distances, results = retriever.search(query, top_k=10)

    print("Top K results:", results)
