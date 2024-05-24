import numpy as np
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity

class AbstractRetriever:
    def __init__(self, db_file, vector_file, pmid_file):
        self.db_file = db_file
        self.vector_file = vector_file
        self.pmid_file = pmid_file
        self.embeddings = np.load(vector_file)
        self.pmids = np.load(pmid_file)
        
    def _connect_db(self):
        self.conn = sqlite3.connect(self.db_file)
        self.cur = self.conn.cursor()
        
    def _close_db(self):
        self.conn.close()
        
    def get_top_k_similar(self, query_embedding, k=5):
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_k_indices = similarities.argsort()[-k:][::-1]
        top_k_pmids = self.pmids[top_k_indices]
        top_k_similarities = similarities[top_k_indices]
        
        return top_k_pmids, top_k_similarities
    
    def fetch_abstracts(self, pmids):
        self._connect_db()
        placeholders = ','.join('?' for _ in pmids)
        query = f"SELECT pmid, title, authors, abstract, publication_year FROM articles WHERE pmid IN ({placeholders})"
        self.cur.execute(query, pmids)
        results = self.cur.fetchall()
        self._close_db()
        
        return results

    def find_similar_abstracts(self, query_embedding, k=5):
        top_k_pmids, top_k_similarities = self.get_top_k_similar(query_embedding, k)
        abstracts = self.fetch_abstracts(top_k_pmids)
        
        return abstracts, top_k_similarities

# Example usage:
if __name__ == "__main__":
    db_file = "pubmed_data.db"
    vector_file = "embedded_vectors.npy"
    pmid_file = "pmids.npy"

    retriever = AbstractRetriever(db_file, vector_file, pmid_file)

    # Example query embedding (replace with an actual embedding)
    query_embedding = np.random.rand(512)  # Assuming 512-dimensional embeddings

    top_k = 5
    similar_abstracts, similarities = retriever.find_similar_abstracts(query_embedding, top_k)
    
    for i, (abstract, similarity) in enumerate(zip(similar_abstracts, similarities)):
        print(f"Rank {i + 1}, Similarity: {similarity}")
        print(f"PMID: {abstract[0]}")
        print(f"Title: {abstract[1]}")
        print(f"Authors: {abstract[2]}")
        print(f"Abstract: {abstract[3]}")
        print(f"Publication Year: {abstract[4]}")
        print("-----")