import  math
import numpy as np
import re
import subprocess
from gc_maps import dic_13b_to_7q as gc_dict_13_7
from gc_maps import dic_15b_to_8q as gc_dict_15_8
import galois
import os
import random


import time

matrix_numpy_file='H_matrix_identity.npy'

# Load the NumPy array from the file
H_numpy = np.load(open(matrix_numpy_file, 'rb'))
GF4 = galois.GF(2 ** 2)
# Create a Galois field matrix using the loaded NumPy array
H = GF4(H_numpy)

# Convert to NumPy array
H_array = np.array(H)

# Check the shape of the NumPy array (and, consequently, the Galois field matrix)
matrix_shape = H_array.shape
print("Matrix shape:", matrix_shape)