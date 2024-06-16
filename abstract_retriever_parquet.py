import dask.dataframe as dd
from dask.diagnostics import ProgressBar

import sqlite3
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel

def _cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / np.maximum(norm_v1 * norm_v2, 1e-5)

def _get_top_k(chunk, query_vector, k):
    chunk['cosine_similarity'] = chunk['doc_vector'].apply(lambda x: _cosine_similarity(np.array(x), query_vector)) 
    top_k_chunk = chunk.nlargest(k, 'cosine_similarity')
    top_k_chunk = top_k_chunk.drop(columns=['doc_vector'])  # Drop the doc_vector column
    return top_k_chunk

class AbstractRetrieverParquet:
    def __init__(self, s3_path, db_file, 
                 model_name="nomic-ai/nomic-embed-text-v1.5", 
                 chunk_size='2GB',
                 use_cuda=False,
                 use_cosine=True):
        self.s3_path = s3_path
        self.db_file = db_file
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.use_cosine = use_cosine

        assert use_cuda == False, 'AbstractRetrieverParquet does not support cuda'
        assert use_cosine == True, 'AbstractRetrieverParquet only supports cosine'

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        self.connection = self._connect_db()

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
        inputs = self.tokenizer(query, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        query_vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return query_vector.astype('float32')
    
    def search(self, query, k=10, num_records=None):
        query_vector = self.embed_query(query)

        # Read the Parquet file using Dask with specified chunk size
        df = dd.read_parquet(self.s3_path, 
                             storage_options={'anon': False}, 
                             chunksize=self.chunk_size)

        if num_records:
            df = df.head(num_records, compute=False)

        # Create meta data for the output DataFrame without the doc_vector column
        meta = df._meta.drop(columns=['doc_vector']).assign(cosine_similarity=np.float64())


        # Wrap the Dask computation with a progress bar
        with ProgressBar():
            # Compute the top-k rows across all chunks
            top_k_chunks = df.map_partitions(_get_top_k, 
                                             query_vector=query_vector, 
                                             k=k,
                                             meta=meta).compute()

        # Combine the results and get the overall top-k
        top_k_rows = top_k_chunks.nlargest(k, 'cosine_similarity')

        pmids = list(top_k_rows['doc_id'])
        similarities = list(top_k_rows['cosine_similarity'])
        documents = self._fetch_document_info(pmids)

        return pmids, similarities, documents


