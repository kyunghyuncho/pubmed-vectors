import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import h5py
from tqdm import tqdm

# Define your file paths
hdf5_file = 'pubmed_embeddings.h5'
parquet_file = 'pubmed_embeddings.parquet'

# Open the HDF5 file
with h5py.File(hdf5_file, 'r') as hdf:
    # Get the datasets you want to convert
    doc_ids = hdf['doc_ids']
    doc_vectors = hdf['doc_vectors']

    # Check that the datasets have compatible shapes
    assert doc_ids.shape[0] == doc_vectors.shape[0], "The number of doc_ids must match the number of doc_vectors"

    # Define the chunk size
    chunk_size = 100_000  # Adjust the chunk size based on your memory constraints

    # Initialize the Parquet writer
    parquet_writer = None

    # Initialize the progress bar
    num_chunks = (doc_ids.shape[0] + chunk_size - 1) // chunk_size  # Calculate the total number of chunks
    pbar = tqdm(total=num_chunks, desc="Processing chunks", unit="chunk")

    # Read the data in chunks and write to Parquet
    for i in range(0, doc_ids.shape[0], chunk_size):
        # Read a chunk of data
        chunk_ids = doc_ids[i:i+chunk_size]
        chunk_vectors = doc_vectors[i:i+chunk_size]
        
        # Combine the chunks into a single DataFrame
        chunk_df = pd.DataFrame({
            'doc_id': chunk_ids,
            'doc_vector': list(chunk_vectors)
        })

        # Convert the DataFrame to a PyArrow Table
        chunk_table = pa.Table.from_pandas(chunk_df)

        # Write the chunk to Parquet
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(parquet_file, chunk_table.schema)
        parquet_writer.write_table(chunk_table)

        # Update the progress bar
        pbar.update(1)

    # Close the Parquet writer
    if parquet_writer:
        parquet_writer.close()

    # Close the progress bar
    pbar.close()
