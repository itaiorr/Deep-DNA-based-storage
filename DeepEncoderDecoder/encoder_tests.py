
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

dict_quat_to_binary = {'A': '00', 'C': '01', 'G': '10', 'T': '11'}

dict_binary_to_quat = {'00': 'A', '01': 'C', '10': 'G', '11': 'T'}




class encoder:
    # the encoder is initialize with the fraction of correctable erasures
    # the fraction of correctable substitutions
    # the fraction of not - correctable substitutions
    # TODO: Complete the params
    def __init__(self, erasure_fraction, substitution_fraction, almost_correct_fraction):
        # Constants
        self.block_size = 55000
        self.RS_block_size = 65535
        self.number_of_RS_symbols_in_a_row = 13
        self.information_bits_in_long_row = 16*13+2*15 # 238 # Todo, functions of the redundancy
        self.information_bits_in_short_row = 16*13 #208 # Todo, functions of the redundancy

        # Calculation of the number of rows from the fraction that was given by the user.
        self.erasure_rows = math.ceil(self.block_size * erasure_fraction)
        self.substitution_rows = math.ceil(self.block_size*substitution_fraction)
        self.almost_correct_rows = math.ceil(self.block_size*almost_correct_fraction)

        # Calculation of the redundancy
        self.tp_redundancy = self.erasure_rows + 2 * (self.substitution_rows+self.almost_correct_rows)
        self.rs_redundancy = self.erasure_rows+2*self.substitution_rows
        # Calculation of the long and short rows
        self.number_of_long_rows = self.block_size - self.tp_redundancy
        self.number_of_short_rows = self.block_size -self.number_of_long_rows - self.rs_redundancy



        self.information_bits_in_block = self.number_of_long_rows * self.information_bits_in_long_row +\
            self.number_of_short_rows*self.information_bits_in_short_row

    def replace_all(self, text, dic):
        for i, j in dic.items():
            text = text.replace(i, j)
        return text

    def get_statistics(self):
        self.total_ecc_redudancy_bits = (self.information_bits_in_short_row*self.rs_redundancy+\
                                    (self.information_bits_in_long_row-self.information_bits_in_short_row)*self.tp_redundancy)/\
                                   (self.information_bits_in_long_row*self.block_size)
        self.total_redudancy_bits = 1-self.information_bits_in_block/(self.block_size*128*2)
        self.information_bits_in_block_in_MB = self.information_bits_in_block/(8*1024*1024)
        print(self.information_bits_in_block_in_MB)
        print(self.information_bits_in_block)
        print(self.number_of_short_rows)
        print(self.block_size-self.number_of_short_rows-self.number_of_long_rows)
        print(self.tp_redundancy)
        print(self.rs_redundancy)
        #exit(0)

    def calculate_padding(self, input_bits):
        # TODO: math
        self.padding_size= self.information_bits_in_block - input_bits - 1
        return

    def covert_data_to_block_format(self, file_name):
        global dict_binary_to_quat
        # reading the data in bytes
        with open(file_name, "rb") as f:
            numpy_data = list(np.fromfile(f, np.dtype('B')))  # bytes in ascii
            x = list(map(lambda y: f'{y:08b}', numpy_data))  #bytes in bits
        binary_str_data = ''.join(x)
        print(len(binary_str_data))
        self.calculate_padding(len(binary_str_data))
        print(self.padding_size)
        if self.padding_size < 1 :
            print("ERROR - file is too big")
            exit(0)
        binary_str_data = '1' +binary_str_data
        binary_str_data =  '0' * self.padding_size +binary_str_data

        list_data = []
        ind = 0
        ## TODO :: Check weather this binary transformation is..
        for i in range(self.number_of_long_rows):
            data = ''
            for j in range(0, self.information_bits_in_long_row, 2):
                #print(binary_str_data[ind:ind + 2])
                data += dict_binary_to_quat.get(binary_str_data[ind:ind + 2])
                #data += binary_str_data[ind:ind + 2]
                ind += 2
            list_data.append(data)
        for i in range(self.number_of_short_rows):
            data = ''
            for j in range(0, self.information_bits_in_short_row, 2):
                data += dict_binary_to_quat.get(binary_str_data[ind:ind + 2])
                #data += binary_str_data[ind:ind + 2]
                ind += 2
            list_data.append(data)
        self.data_padded = list_data




    def covert_data_to_binary_for_c1_encoding(self):
        #global RS_block_size
        global dict_quat_to_binary
        data = self.data_padded
        f=bytearray()
        for ind in range(0, self.number_of_RS_symbols_in_a_row):
            block_ind = 8 * ind
            block = ''
            for line_ind in range(self.number_of_long_rows+self.number_of_short_rows):
                block += data[line_ind][block_ind:block_ind + 8]

            block = self.replace_all(block, dict_quat_to_binary)

            block_size_in_words = len(block) / 16

            padding_size_in_words = int(self.RS_block_size - self.rs_redundancy - block_size_in_words)



            block += '0' * padding_size_in_words * 16
            b = block
            array = bytearray(int(b[x:x + 8], 2) for x in range(0, len(b), 8))
            f += array

        file_output = open("c1_encoding_in.bin", "wb")
        file_output.write(f)

        return

    def perform_c1(self):
        # Todo: MOVE to the INIT Func??
        hfile = open("schifracopy/CommonDefinitions.h", "w")
        hfile.write("auto constexpr t1 = "+str(int((self.tp_redundancy - self.rs_redundancy)/2)) +";\n")
        hfile.write("auto constexpr t2 = " + str(self.rs_redundancy) + ";\n")
        hfile.write("auto constexpr M_orig = " + str(self.number_of_long_rows) + ";\n")
        hfile.close()
        os.system("g++-7 -ansi -pedantic-errors -Wall -Wextra -Wno-long-long -Wno-unused-variable -O3 -o schifracopy/c1_dvir schifracopy/C1_encoding.cpp -std=c++17 -lm")
        subprocess.call(["./schifracopy/c1_dvir"])
        return

    def convert_binary_from_c1_encoding(self):

        global dict_binary_to_quat
        with open("c1_encoding_out.bin", "rb") as f:
            numpy_data = list(np.fromfile(f, np.dtype('B')))
            x = list(map(lambda y: f'{y:08b}', numpy_data))
        binary_str_data = ''.join(x)
        tmp = {}
        total_block_len_in_bits = self.block_size*16 #TODO fix it

        for i in range(self.block_size):
            tmp.update({i: ''})

        for n in range(self.number_of_RS_symbols_in_a_row):
            current = binary_str_data[n * total_block_len_in_bits: (n + 1) * total_block_len_in_bits]
            ind = 0
            for i in range(0, self.number_of_long_rows+self.number_of_short_rows):
                tmp.update({i: tmp.get(i) + current[ind: ind + 16]})
                ind += 16
            for i in range(0, self.rs_redundancy):
                tmp.update({self.number_of_long_rows+self.number_of_short_rows + i:
                                tmp.get(self.number_of_long_rows+self.number_of_short_rows + i) + current[ind: ind + 16]})
                ind += 16

        #with open('data.txt', 'r') as f:
        data = self.data_padded
        self.data_coded_with_c1=[]
        #with open('data_after_c1_encoding.txt', 'w') as f:
        for i in range(self.block_size):
            bin_str = tmp.get(i)

            acgt_str = ''
            for j in range(0, len(bin_str), 2):
                acgt_str += dict_binary_to_quat.get(bin_str[j:j + 2])
            if i <self.number_of_long_rows:#+self.number_of_short_rows: TODO: fix that?
                acgt_str += data[i][self.number_of_RS_symbols_in_a_row * 8:len(data[i])]
            #f.write(acgt_str + '\n')

            self.data_coded_with_c1.append(acgt_str)
        return

    def encode_gc_content_after_c1(self):
        global dict_quat_to_binary
        isFirst=True
        #input_file_path = 'data_after_c1_encoding.txt' if isFirst else 'data_after_c1_decoding.txt'
        output_file_path = 'data_after_gc_encoding.txt' if isFirst else 'data_after_gc_encoding_2.txt'
        min_val = 1
        max_val = -1

        data=self.data_coded_with_c1
        self.data_coded_with_c1_gc=[]
        #with open(output_file_path, 'w') as f2:
        line_ind = 0
        for strand in data:
            encoded_strand = ''
            encoded_strand_len = 0
            gc_counter = 0
            strand = strand.strip()
            strand = self.replace_all(strand, dict_quat_to_binary)
            x = gc_dict_13_7.get(strand[0:13])
            if type(x) is tuple:
                encoded_strand += x[0]
                gc_counter += x[0].count('G') + x[0].count('C')
            else:
                encoded_strand += x
                gc_counter += x.count('G') + x.count('C')
            encoded_strand_len += 7

            for i in range(13, self.information_bits_in_short_row, 13):
                x = gc_dict_13_7.get(strand[i:i + 13])
                if type(x) is tuple:
                    x0_count = x[0].count('G') + x[0].count('C')
                    x1_count = x[1].count('G') + x[1].count('C')
                    if abs(((x0_count + gc_counter) / (encoded_strand_len + 7)) - 0.5) < abs(
                            ((x1_count + gc_counter) / (encoded_strand_len + 7)) - 0.5):
                        encoded_strand += x[0]
                        gc_counter += x0_count
                    else:
                        encoded_strand += x[1]
                        gc_counter += x1_count
                else:
                    encoded_strand += x
                    gc_counter += (x.count('G') + x.count('C'))
                encoded_strand_len += 7

            if isFirst:
                if line_ind < self.number_of_long_rows:
                    for i in range(self.information_bits_in_short_row, self.information_bits_in_long_row, 15):
                        x = gc_dict_15_8.get(strand[i:i + 15])

                        if type(x) is tuple:
                            x0_count = x[0].count('G') + x[0].count('C')
                            x1_count = x[1].count('G') + x[1].count('C')
                            if abs(((x0_count + gc_counter) / (encoded_strand_len + 8)) - 0.5) < abs(
                                    ((x1_count + gc_counter) / (encoded_strand_len + 8)) - 0.5):
                                encoded_strand += x[0]
                                gc_counter += x0_count
                            else:
                                encoded_strand += x[1]
                                gc_counter += x1_count
                        else:
                            encoded_strand += x
                            gc_counter += (x.count('G') + x.count('C'))
                        encoded_strand_len += 8

            min_val = min(gc_counter / encoded_strand_len, min_val)
            max_val = max(gc_counter / encoded_strand_len, max_val)
            line_ind += 1
            #f2.write(encoded_strand + '\n')
            self.data_coded_with_c1_gc.append(encoded_strand)
            #print("min val: " + str(min_val))
            #print("max val: " + str(max_val))
        return

    def convert_data_to_binary_in_format_for_c2_encoding(self, matrix_numpy_file='H_matrix_identity.npy'):
        #with open('data_after_gc_encoding.txt', 'r') as f:
        data = self.data_coded_with_c1_gc
        c2_redundency=self.tp_redundancy
        GF4 = galois.GF(2 ** 2)
        #loading the matrix H
        H=GF4(np.load(open(matrix_numpy_file, 'rb')))
        s1 = ''
        s2 = ''
        self.data_to_c2_encoder = bytearray()
        with open('c2_encoding_in.bin', 'wb') as f:
            for line_ind in range(self.block_size - self.tp_redundancy): # number of information rows in the syndrome column
                block = data[line_ind].strip()
                block = self.replace_all(block, {'A': '0 ', 'C': '1 ', 'G': '2 ', 'T': '3 '})
                vec = GF4(np.fromstring(block, dtype=int, sep=' '))
                syndrome = H @ vec
                tmp = "".join(map(str, np.array(syndrome)))
                s1 += tmp[0:8]
                s2 += tmp[8:16]

            s1 = self.replace_all(s1, {'0': '00', '1': '01', '2': '10', '3': '11'})
            s2 = self.replace_all(s2, {'0': '00', '1': '01', '2': '10', '3': '11'})


            block_size_in_words = len(s1) / 16
            padding_size_in_words = int(self.RS_block_size - self.tp_redundancy - block_size_in_words)
            s1 += '0' * 16 * padding_size_in_words
            s2 += '0' * 16 * padding_size_in_words
            b = s1
            array = bytearray(int(b[x:x + 8], 2) for x in range(0, len(b), 8))
            f.write(array)
            self.data_to_c2_encoder+=array
            b = s2
            array = bytearray(int(b[x:x + 8], 2) for x in range(0, len(b), 8))
            f.write(array)
            self.data_to_c2_encoder += array

        return

    def perform_c2(self):
        os.system("g++-7 -ansi -pedantic-errors -Wall -Wextra -Wno-long-long -Wno-unused-variable -O3 -o schifracopy/c2_dvir schifracopy/C2_encoding.cpp -std=c++17 -lm")
        subprocess.call(["./schifracopy/c2_dvir"])
        return

    def convert_binary_from_c2_encoding(self, matrix_numpy_file='H_matrix_identity.npy'):
        dict_2 = {'00': 'A', '01': 'C', '10': 'G', '11': 'T'}

        with open("c2_encoding_out.bin", "rb") as f:
            numpy_data = list(np.fromfile(f, np.dtype('B')))
            x = list(map(lambda y: f'{y:08b}', numpy_data))

        GF4 = galois.GF(2 ** 2)
        H=GF4(np.load(open(matrix_numpy_file, 'rb')))

        cols_to_delete = [col for col in range(128-16, 128)] # todo: fix that

        sub_H = np.delete(H, cols_to_delete, 1)
        binary_str_data = ''.join(x)
        tmp = {}
        total_block_len_in_bits = self.RS_block_size * 16
        fec_start_position = total_block_len_in_bits - (self.tp_redundancy * 16)

        for i in range(self.tp_redundancy):
            tmp.update({i: ''})

        for n in range(2):
            current = binary_str_data[n * total_block_len_in_bits: (n + 1) * total_block_len_in_bits]
            ind = int(fec_start_position)
            for i in range(0, self.tp_redundancy):
                tmp.update({i: tmp.get(i) + current[ind: ind + 16]})
                ind += 16

        data=self.data_coded_with_c1_gc

        with open('data_after_c2_encoding.txt', 'w') as f:
            for i in range(self.number_of_long_rows):
                f.write(data[i]+"\n")

            for i in range(self.tp_redundancy):
                bin_str = tmp.get(i)
                acgt_str = ''
                for j in range(0, len(bin_str), 2):
                    acgt_str += dict_2.get(bin_str[j:j + 2])

                x = data[self.number_of_long_rows + i].strip()
                x = self.replace_all(x, {'A': '0 ', 'C': '1 ', 'G': '2 ', 'T': '3 '})
                sub_s = GF4(np.fromstring(x, dtype=int, sep=' '))
                m = sub_H @ sub_s
                real_syndrome = acgt_str
                real_syndrome = self.replace_all(real_syndrome, {'A': '0 ', 'C': '1 ', 'G': '2 ', 'T': '3 '})
                t = GF4(np.fromstring(real_syndrome, dtype=int, sep=' '))
                b = np.array(np.add(t, m))
                ans = ''.join(map(str, b))
                ans = self.replace_all(ans, {'0': 'A', '1': 'C', '2': 'G', '3': 'T'})

                f.write(data[self.number_of_long_rows + i].strip() + ans + '\n')

    # This function gets informations bits,
    # partition them to chunks of fixed size and encodes them into blocks/matrices.
    def encode_file(self, file_name):
        # TODO : math to check how many bits in the file,
        #  add 1 in the begining and then padding_size of zeroes
        return

    # This function encodes a block=matrix of fixed size 55,000 x 128
    # The input is segmented binary information of specific size from the user
    # The output is one encoded matrix of size TODO:
    # TODO: Fix the parameters
    def encode_block(bitstream):
        # cpp code
        return

    def gc_row_covert_bits_to_quad(self, row):
        # First XX substrings are coverted 13b to 7q, last is 15b to 8q
        # TODO: fix the numbers here
        return

def get_indices(indices_file ="indices.txt"):
    with open(indices_file, 'r') as f:
        data = f.readlines()
    valid_indices = []
    for line in data:
        valid_indices.append(line.strip())
    valid_indices.sort()
    return valid_indices

def generate_indices(file_1 = True, encoded_file = 'data_after_c2_encoding.txt', indices_file ="indices.txt"):
    global indices_exists
    with open(encoded_file, 'r') as f:
        data = f.readlines()
        number_of_lines=len(data)
    if file_1:
        c1='A'
        c2 = 'C'
    else:
        c1 = 'G'
        c2 = 'T'

    indices = get_indices()
    my_indices = indices[:number_of_lines]
    my_indices.sort()
    random_lines_real_indices = list(range(number_of_lines))
    random_lines_real_indices.sort()
    results = {}
    #    c1 = 'A'
    #    c2 = 'C'
    with open('data_after_indices.txt', 'w') as f:
        for counter, index in enumerate(random_lines_real_indices):
            line = data[index].strip()
            index_candidate = my_indices[counter]
            tmpA = index_candidate + c1 + line
            tmpC = index_candidate + c2 + line
            found = False
            tmpA2 = tmpA[:124]
            tmpC2 = tmpC[:124]
            if 'AAAAA' not in tmpA2 and 'CCCCC' not in tmpA2 and 'GGGGG' not in tmpA2 and 'TTTTT' not in tmpA2:
                if 'AAAAA' not in tmpC2 and 'CCCCC' not in tmpC2 and 'GGGGG' not in tmpC2 and 'TTTTT' not in tmpC2:
                    found = True
                    t = random.choice([tmpA, tmpC])
                    f.write(t + '\n')
                else:
                    found = True
                    f.write(tmpA + '\n')
            elif 'AAAAA' not in tmpC2 and 'CCCCC' not in tmpC2 and 'GGGGG' not in tmpC2 and 'TTTTT' not in tmpC2:
                found = True
                f.write(tmpC + '\n')

#            if found:
#                if not indices_exists:
#                    results.update({index_candidate: index})
#            else:
#                print("bugggg!!!")
    print("done!")


# input:
# main function - this function builds the decoder from the given parameters - fraction of correctable erasures, correctable subsitutitions, and correctable "almost" correct sub (up to 3 errors)
#                   the values should be number between 0 to 1, where the sum of them is strictly smaller than 1.
# file_to_encode- is the path of the  file we want to encode
# matrix_numpy_file - matrix of the TP code - should be a .npy file
# is indexer - true if you want indices or false
# is primer - future? # todo: discuss it with Daniella/Itai PRIMERS for illumina adapter
# note that this function assumes g++-7 is installed on the machine

# output:
# TODO : fix things that it will be automated.
# the encoded file without is in "data_after_c2_encoding.txt"
# the encoded file with indice is in "data_after_indices.txt
# erasure_fraction=0.04, substitution_fraction=0.0075, almost_correct_fraction=0.016
def encode(erasure_fraction=0.04, substitution_fraction=0.0075, almost_correct_fraction=0.016,
           file_to_encode="/Users/omersabary/Dropbox/FilesToDecode/Armstrong/armstrong/DeepDNA-File1.zip",
           matrix_numpy_file='H_matrix_identity.npy', is_indexer=True, is_primer=True):
    st = time.time()

    End = encoder(erasure_fraction, substitution_fraction, almost_correct_fraction)
    End.get_statistics()
    End.covert_data_to_block_format(file_to_encode)

    End.covert_data_to_binary_for_c1_encoding()
    End.perform_c1()
    End.convert_binary_from_c1_encoding()
    End.encode_gc_content_after_c1()
    exit(0)

    End.convert_data_to_binary_in_format_for_c2_encoding(matrix_numpy_file)

    End.perform_c2()

    End.convert_binary_from_c2_encoding(matrix_numpy_file)

    # TODO: add indexer to the pipeline :) right now the set of indices is pre-made
    # The set of indices
    end = time.time()
    print(end-st)
    if is_indexer:
        generate_indices(file_1=False)


#encode(file_to_encode="/Users/omersabary/Desktop/DaniellaDvir/test_finale.nosync/two_files.nosync/rand_file.bin")
encode(file_to_encode="/Users/omersabary/Desktop/DaniellaDvir/test_finale/rand_file/random_file/rand_file.bin", is_indexer=True)

#generate_indices(encoded_file="/Users/omersabary/Desktop/DaniellaDvir/test_finale.nosync/Submission-Martine/PilotPool/pilot_strands.txt")