from lmic.compressors import language_model
import random

filename = 'seq_file'
with open(filename, 'rb') as f:
    data = f.read()
compressed_data, num_padded_bits = language_model.compress(
  data,
  return_num_padded_bits=True,
  use_slow_lossless_compression=True,
)
print(compressed_data)
decompressed_data = language_model.decompress(
  compressed_data,
  num_padded_bits=num_padded_bits,
  uncompressed_length=len(data),
)
assert data == decompressed_data
print('compression rate', len(compressed_data) / len(data))