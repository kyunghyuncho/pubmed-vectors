import numpy as np
import struct

# Function to write int64 and numpy array to a binary file
def write_pair_to_file(file, id_val, np_array):
    # Write int64
    file.write(struct.pack('q', id_val))
    # Write the shape of the numpy array
    file.write(struct.pack('i', np_array.shape[0]))
    file.write(struct.pack('i', np_array.shape[1]))
    # Write the numpy array data
    file.write(np_array.tobytes())

# Function to read int64 and numpy array from a binary file
def read_pair_from_file(file):
    # Read int64
    id_val = struct.unpack('q', file.read(8))[0]
    # Read the shape of the numpy array
    rows = struct.unpack('i', file.read(4))[0]
    cols = struct.unpack('i', file.read(4))[0]
    # Read the numpy array data
    np_array = np.frombuffer(file.read(rows * cols * 8), dtype=np.float64).reshape((rows, cols))
    return id_val, np_array


