
import os

# size is in bytes
with open('rand_file.bin', 'wb') as fout:
    fout.write(os.urandom(1539626))