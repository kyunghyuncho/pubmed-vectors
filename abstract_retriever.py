import numpy as np
import os
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
import sqlite3

class AbstractRetriever:
    def __init__(self, vectors_file, ids_file, db_file, model_name="nomic-ai/nomic-embed-text-v1.5"):
        self.vectors_file = vectors_file
        self.ids_file = ids_file
        self.db_file = db_file
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.dimension = self._infer_dimension()
        self.dtype_vectors = 'float32'
        self.dtype_ids = 'int64'
        self.num_rows = self._get_num_rows()
        self.doc_vectors_memmap = np.memmap(vectors_file, dtype=self.dtype_vectors, mode='r', shape=(self.num_rows, self.dimension))
        self.doc_ids_memmap = np.memmap(ids_file, dtype=self.dtype_ids, mode='r', shape=(self.num_rows,))
        self.index = self._build_faiss_index()
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
        assert num_rows_vectors == num_rows_ids, "Number of rows in vectors and ids files must match"
        return num_rows_vectors
    
    def _build_faiss_index(self):
        index = faiss.IndexFlatL2(self.dimension)
        index.add(self.doc_vectors_memmap)
        return index
    
    def _connect_db(self):
        connection = sqlite3.connect(self.db_file)
        return connection
    
    def _fetch_document_info(self, pmids):
        cursor = self.connection.cursor()
        query = f"SELECT pmid, title, authors, abstract, publication_year FROM articles WHERE pmid IN ({','.join(['?']*len(pmids))})"
        cursor.execute(query, tuple(pmids))
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
        distances, indices = self.index.search(np.array([query_vector]), top_k)
        pmids = self.doc_ids_memmap[indices[0]]
        documents = self._fetch_document_info(pmids)
        return distances, documents

# Example usage:
if __name__ == "__main__":
    vectors_file = 'embeddings_test.dat'
    ids_file = 'pmids_test.dat'
    db_file = 'pubmed_db.db'

    retriever = AbstractRetriever(vectors_file, ids_file, db_file)
    query = "example search query"
    distances, results = retriever.search(query, top_k=10)

    print("Top K results:", results)
