from language_modeling_is_compression.compressors import language_model
import random


filename = 'seq_file'
with open(filename, 'rb') as f:
    data = f.read()
length = len(data)
compressed_data, num_padded_bits = language_model.compress(
  data.decode(),
  return_num_padded_bits=True,
  use_slow_lossless_compression=True,
)
decompressed_data = language_model.decompress(
  compressed_data,
  num_padded_bits=num_padded_bits,
  uncompressed_length=length,
)
assert data == decompressed_data
print('compression rate', len(compressed_data) / len(data))