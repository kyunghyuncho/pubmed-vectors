import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

BATCH_SIZE=100000

def quantize_vector(vector, min_val, max_val):
    # Normalize to [0, 1]
    normalized = (vector - min_val) / (max_val - min_val)
    # Scale to [-128, 127]
    quantized = normalized * 255 - 128
    return quantized.astype(np.int8)

def process_chunk(chunk, min_val, max_val):
    # Convert the chunk to a pandas DataFrame
    df = chunk.to_pandas()

    # Assuming doc_vector is stored as a list in each row
    doc_vectors = np.vstack(df['doc_vector'].values)

    # Quantize the vectors
    quantized_vectors = np.apply_along_axis(quantize_vector, 1, doc_vectors, min_val, max_val)

    # Replace the float32 vectors with int8 vectors in the DataFrame
    df['doc_vector'] = list(quantized_vectors)

    # Convert the DataFrame back to a PyArrow table
    new_chunk = pa.Table.from_pandas(df)
    return new_chunk

def get_min_max(file_path, batch_size=BATCH_SIZE):
    min_val, max_val = None, None
    reader = pq.ParquetFile(file_path)
    num_batches = reader.metadata.num_row_groups
    with tqdm(total=num_batches, desc='Calculating min/max') as pbar:
        for batch in reader.iter_batches(batch_size=batch_size):
            df = pa.Table.from_batches([batch]).to_pandas()
            doc_vectors = np.vstack(df['doc_vector'].values)
            batch_min, batch_max = doc_vectors.min(), doc_vectors.max()
            if min_val is None or batch_min < min_val:
                min_val = batch_min
            if max_val is None or batch_max > max_val:
                max_val = batch_max
            pbar.update(1)
    return float(min_val), float(max_val)

def save_min_max(min_val, max_val, file_path):
    data = {'min_val': min_val, 'max_val': max_val}
    with open(file_path, 'w') as f:
        json.dump(data, f)

def load_min_max(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['min_val'], data['max_val']

def quantize_parquet(input_path, output_path, min_max_file, batch_size=BATCH_SIZE):
    # Check if min_max file exists
    try:
        min_val, max_val = load_min_max(min_max_file)
    except FileNotFoundError:
        min_val, max_val = get_min_max(input_path, batch_size)
        save_min_max(min_val, max_val, min_max_file)
    
    reader = pq.ParquetFile(input_path)
    writer = None
    num_batches = reader.metadata.num_row_groups

    with tqdm(total=num_batches, desc='Quantizing') as pbar:
        for batch in reader.iter_batches(batch_size=batch_size):
            chunk = pa.Table.from_batches([batch])
            new_chunk = process_chunk(chunk, min_val, max_val)

            if writer is None:
                writer = pq.ParquetWriter(output_path, new_chunk.schema)

            writer.write_table(new_chunk)
            pbar.update(1)

    if writer:
        writer.close()

# File paths
input_parquet_file = 'pubmed_embeddings.parquet'
output_parquet_file = 'pubmed_embeddings_int8.parquet'
min_max_file = 'min_max_values.json'

# Quantize the parquet file
quantize_parquet(input_parquet_file, output_parquet_file, min_max_file)


