import numpy as np
import struct

# Function to write int64 and numpy array to a binary file
def write_pair_to_file(file, id_val, np_array):
    # Encode the string to bytes
    id_bytes = id_val.encode('utf-8')
    id_length = len(id_bytes)
    # Write the length of the string
    file.write(struct.pack('i', id_length))
    # Write the string
    file.write(id_bytes)

    # Write the shape of the numpy array
    file.write(struct.pack('i', np_array.shape[0]))
    # Write the numpy array data
    file.write(np_array.astype('float16').tobytes())

# Function to read int64 and numpy array from a binary file
def read_pair_from_file(file):
    # Read the length of the string
    id_length = struct.unpack('i', file.read(4))[0]
    # Read the string
    id_bytes = file.read(id_length)
    id_val = id_bytes.decode('utf-8')

    # Read the shape of the numpy array
    rows = struct.unpack('i', file.read(4))[0]
    # Read the numpy array data
    np_array = np.frombuffer(file.read(rows * 2), dtype=np.float16).reshape(rows)

    return id_val, np_array


