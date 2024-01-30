
import time
import pickle
import math
import galois
import numpy as np
import subprocess
from gc_maps import dic_13b_to_7q as gc_dict_13_7
from gc_maps import dic_15b_to_8q as gc_dict_15_8
from gc_maps import dic_7q_to_13b as reversed_gc_dict_13_7
from gc_maps import dic_8q_to_15b as reversed_gc_dict_15_8
import os
from tqdm import tqdm
import json
from indices_dic import ind_to_num
from indices_dic import num_to_ind
import filecmp


def hamming_distance(string1, string2):
    return sum(c1 != c2 for c1, c2 in zip(string1, string2))

def create_some_noise_omer(M, t1, t2):
    RS_block_size=65535
    with open('/Users/omersabary/Documents/test_omer_dvir/line_erasure.txt', 'r') as f0:
        erasures_lines_read = f0.readlines()
    erasures_lines = [int(x.strip()) for x in erasures_lines_read]
    erasures_lines.sort()
    with open('c1_erasures_locations.txt', 'w') as f1:
        with open('c2_erasures_locations.txt', 'w') as f2:
            tmp1 = erasures_lines
            tmp1 = list(map(lambda x: x if x < M else x + (RS_block_size - (M + 2 * t1 + t2)), tmp1))
            tmp1.sort()
            for line_ind in tmp1:
                f2.write(str(line_ind) + '\n')

            tmp2 = erasures_lines
            tmp2 = list(map(lambda x: x if x < M + 2 * t1 else x + (RS_block_size - (M + 2 * t1 + t2)), tmp2))
            tmp2.sort()
            for line_ind in tmp2:
                f1.write(str(line_ind) + '\n')
    return erasures_lines

class decoder:
    # the decoder is initialize with the fraction of correctable erasures
    # the fraction of correctable substitutions
    # the fraction of not - correctable substitutions
    # TODO: Complete the params
    def __init__(self, erasure_fraction, substitution_fraction, almost_correct_fraction):
        # Constants
        self.block_size = 55000
        self.RS_block_size = 65535
        self.number_of_bases_in_long_row = 128
        self.number_of_bases_in_short_row = 112

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

    def read_data_to_decode(self, data_file):
        with open(data_file, 'r') as f:
            self.data = f.readlines()
        self.erasures_lines = []

    def erasure_line(self, erasure_line_file):
        with open(erasure_line_file, 'r') as f0:
            erasures_lines_read = f0.readlines()
        self.erasures_lines = [int(x.strip()) for x in erasures_lines_read]
        self.erasures_lines.sort()

    # Input - data + matrix H
    # the function coverts the data to binray form for c_2 decoding
    # the function also uses the matrix H to calculate the phantom syndrome row
    # the function creates the file c2_decoding_in.bin with the phantom syndrom for c_2 decoding
    def convert_data_to_binary_in_format_for_c2_decoding(self, matrix_numpy_file):
        GF4 = galois.GF(2 ** 2)
        H=GF4(np.load(open(matrix_numpy_file, 'rb')))
        s1 = ''
        s2 = ''
        self.data_for_c2 = bytearray()
        with open('c2_decoding_in.bin', 'wb') as f:
            for line_ind in range(self.number_of_long_rows):
                block = self.data[line_ind].strip()
                block = self.replace_all(block, {'A': '0 ', 'C': '1 ', 'G': '2 ', 'T': '3 '})
                vec = GF4(np.fromstring(block, dtype=int, sep=' '))
                syndrome = H @ vec
                tmp = "".join(map(str, np.array(syndrome)))
                s1 += tmp[0:8]
                s2 += tmp[8:16]

            # add padding
            block_size_in_words = len(s1) / 8
            padding_size_in_words = int(self.RS_block_size - self.tp_redundancy - block_size_in_words)
            s1 += '0' * 8 * padding_size_in_words
            s2 += '0' * 8 * padding_size_in_words

            for line_ind in range(self.number_of_long_rows, self.number_of_long_rows + self.tp_redundancy):
                block = self.data[line_ind].strip()
                block = self.replace_all(block, {'A': '0 ', 'C': '1 ', 'G': '2 ', 'T': '3 '})
                vec = GF4(np.fromstring(block, dtype=int, sep=' '))
                syndrome = H @ vec
                tmp = "".join(map(str, np.array(syndrome)))
                s1 += tmp[0:8]
                s2 += tmp[8:16]

            s1 = self.replace_all(s1, {'0': '00', '1': '01', '2': '10', '3': '11'})
            s2 = self.replace_all(s2, {'0': '00', '1': '01', '2': '10', '3': '11'})

            b = s1
            array = bytearray(int(b[x:x + 8], 2) for x in range(0, len(b), 8))
            f.write(array)
            self.data_for_c2 += array
            b = s2
            array = bytearray(int(b[x:x + 8], 2) for x in range(0, len(b), 8))
            f.write(array)
            self.data_for_c2 += array

    # C_2 decoder is compiled and invoked on the phantom syndromes
    def perform_c2_dec(self):
        ## todo: maybe move to the initialization?
        hfile = open("schifracopy/CommonDefinitions.h", "w")
        hfile.write("auto constexpr t1 = " + str(int((self.tp_redundancy - self.rs_redundancy) / 2)) + ";\n")
        hfile.write("auto constexpr t2 = " + str(self.rs_redundancy) + ";\n")
        hfile.write("auto constexpr M_orig = " + str(self.number_of_long_rows) + ";\n")
        hfile.close()
        os.system("g++-7 -ansi -pedantic-errors -Wall -Wextra -Wno-long-long -Wno-unused-variable -O3 -o schifracopy/c2_dvir_dec schifracopy/C2_decoding.cpp -std=c++17 -lm")

        subprocess.call(["./schifracopy/c2_dvir_dec"])
        return

    # this function uses the syndrome dict to compute them after c_2 correction
    # the function corrects syndromes with less errors
    def get_syndromes_dict_after_c2_decoding(self):
        dict_3 = {'00': '0 ', '01': '1 ', '10': '2 ', '11': '3 '}
        self.syndrome_phantom_vec = {}
        with open("c2_decoding_out.bin", "rb") as f:
            numpy_data = list(np.fromfile(f, np.dtype('B')))
            x = list(map(lambda y: f'{y:08b}', numpy_data))
        binary_str_data = ''.join(x)
        tmp = {}
        total_block_len_in_bits = self.RS_block_size * 16
        fec_start_position = total_block_len_in_bits - (self.tp_redundancy * 16)

        for i in range(self.number_of_long_rows + self.tp_redundancy):
            tmp.update({i: ''})

        for n in range(2):
            ind = 0
            current = binary_str_data[n * total_block_len_in_bits: (n + 1) * total_block_len_in_bits]
            for i in range(self.number_of_long_rows):
                tmp.update({i: tmp.get(i) + current[ind: ind + 16]})
                ind += 16
            ind = int(fec_start_position)
            for i in range(self.number_of_long_rows, self.number_of_long_rows + self.tp_redundancy):
                tmp.update({i: tmp.get(i) + current[ind: ind + 16]})
                ind += 16

        tmp2 = {}
        for key in tmp:
            my_str = ''
            bin_str = tmp.get(key)
            for j in range(0, len(bin_str), 2):
                my_str += dict_3.get(bin_str[j:j + 2])
            tmp2.update({key: my_str})
        self.syndrome_phantom_vec=tmp2

    # this function loads the syndrome dict table that is used to correct row with 3 or less errors
    def get_syndromes_dict(self, syndrom_table_pkl_path="/Users/omersabary/Dropbox/for_omer/syndromes_dict_identity.pkl"):
        t1 = time.time()
        with open(syndrom_table_pkl_path, "rb") as f:
            syndromes_dict = pickle.load(f)
        t2 = time.time()
        print("syndromes loading time: " + str(t2 - t1))
        self.syn_dic= syndromes_dict

    # This function uses the H matrix to covert the binary data from the phantom syndrome vector
    # This function detect line with 3 or less errors and corrects them
    def convert_binary_from_c2_decoding(self, matrix_numpy_file="H_matrix_identity.npy", era_lines=[], isTest=False):
        dict_5 = {'A': '0 ', 'C': '1 ', 'G': '2 ', 'T': '3 '}
        dict_6 = {'0': 'A', '1': 'C', '2': 'G', '3': 'T'}
        self.noisy_data = self.data
        GF4 = galois.GF(2 ** 2)
        # loading the matrix H
        H = GF4(np.load(open(matrix_numpy_file, 'rb')))
        tmp2 = self.syndrome_phantom_vec


        mul_sub_counted = []
        self.data_after_c2_corrections = []
        #with open('data_after_c2_decoding.txt', 'w') as f:
        for line_ind in range(self.number_of_long_rows + self.tp_redundancy):
            block = self.noisy_data[line_ind].strip()
            block = self.replace_all(block, dict_5)
            vec = GF4(np.fromstring(block, dtype=int, sep=' '))
            syndrome = H @ vec
            syndrome_str = "".join(map(str, np.array(syndrome)))
            if line_ind not in self.erasures_lines and tmp2.get(line_ind).replace(' ', '') != syndrome_str:
                s = GF4(np.fromstring(tmp2.get(line_ind), dtype=int, sep=' '))
                syndrome_error_vec = np.add(GF4(syndrome), s)
                tmp = "".join(map(str, np.array(syndrome_error_vec)))
                if tmp in self.syn_dic:
                    n = self.syn_dic.get(tmp)
                    correct_line = np.add(vec, GF4(n))
                else:
                    mul_sub_counted.append(line_ind)
                    correct_line = vec
            else:
                correct_line = vec
            correct_str = "".join(map(str, np.array(correct_line)))
            correct_str = self.replace_all(correct_str, dict_6)
            if line_ind < self.number_of_long_rows:
                #f.write(correct_str + '\n')
                self.data_after_c2_corrections.append(correct_str)
            else:
                #f.write(correct_str[:128 - 16] + '\n')  #todo : fix that
                self.data_after_c2_corrections.append(correct_str[:128 - 16])

        if isTest:  # Test
            with open('line_subsitution.txt', 'r') as f0:
                mul_sub_lines_read = f0.readlines()
            mul_sub_lines_read = [int(x.strip()) for x in mul_sub_lines_read]
            mul_sub_lines_read.sort()
            mul_sub_counted.sort()
            x = set(mul_sub_lines_read)
            y = set(mul_sub_counted)
            if (y.issubset(x)):
                print("Multiple Subs are identical !! :) :) :) ")
            else:
                print("Multiple Subs error.. !! :) :) :) ")
            print("mul_sub_counted :")
            print(*mul_sub_counted)

            print("mul_sub_lines_read :")
            print(*mul_sub_lines_read)
        return


    # This function decodes the data from GC conent to its binary form before performing the decoding procedure of C1
    def convert_from_gc_content(self, erasures_lines=[]):

        dict_2 = {'00': 'A', '01': 'C', '10': 'G', '11': 'T'}

        data = self.data_after_c2_corrections
        self.data_after_gc_dec = []
        #with open('data_after_gc_decoding.txt', 'w') as f:
        line_ind = 0
        for strand in data:
            if line_ind in self.erasures_lines:
                if line_ind < self.number_of_long_rows:
                    final_decoded_strand = 'A' * int(self.information_bits_in_long_row/2) # todo fix that
                else:
                    final_decoded_strand = 'A' * int(self.information_bits_in_short_row/2) # todo fix
                line_ind += 1
                self.data_after_gc_dec.append(final_decoded_strand)
                #f.write(final_decoded_strand + '\n')
                continue

            # in case of erros, a possible case is that we get a block that is not part of our GC scheme, in this case
            # the block is decode to the all Zeroes block and will be corrected by the outer code as a part of the line
            # correction
            decoded_strand = ''
            final_decoded_strand = ''
            strand = strand.strip()
            for i in range(0, self.number_of_bases_in_short_row, 7):
                tmp = reversed_gc_dict_13_7.get(strand[i:i + 7])
                decoded_strand += tmp if tmp is not None else '0000000000000'
            if line_ind < self.number_of_long_rows:
                for i in range(self.number_of_bases_in_short_row, self.number_of_bases_in_long_row, 8):
                    tmp = reversed_gc_dict_15_8.get(strand[i:i + 8])
                    decoded_strand += tmp if tmp is not None else '000000000000000'
            for i in range(0, len(decoded_strand), 2):
                final_decoded_strand += dict_2.get(decoded_strand[i: i + 2])
            line_ind += 1
            #f.write(final_decoded_strand + '\n')
            self.data_after_gc_dec.append(final_decoded_strand)
        return

    # This function converts the data to its binary form beform performing the decoding procedure with c1
    def convert_data_to_binary_in_format_for_c1_decoding(self):
        data = self.data_after_gc_dec

        self.data_for_c1_dec = bytearray()
        with open('c1_decoding_in.bin', 'wb') as f:
            for ind in range(0, self.number_of_RS_symbols_in_a_row):
                block_ind = 8 * ind
                block = ''
                for line_ind in range(self.number_of_long_rows+self.number_of_short_rows):
                    block += data[line_ind][block_ind:block_ind + 8]
                for line_ind in range(self.number_of_long_rows+self.number_of_short_rows, len(data)):
                    block += data[line_ind][block_ind:block_ind + 8]
                block = self.replace_all(block,{'A': '00', 'C': '01', 'G': '10', 'T': '11'})
                b = block
                array = bytearray(int(b[x:x + 8], 2) for x in range(0, len(b), 8))
                f.write(array)
                self.data_for_c1_dec+=array
        return

    # c_1 is performed on the data
    def perform_c1_dec(self):
        os.system("g++-7 -ansi -pedantic-errors -Wall -Wextra -Wno-long-long -Wno-unused-variable -O3 -o schifracopy/c1_dvir_dec schifracopy/C1_decoding.cpp -std=c++17 -lm")

        subprocess.call("./schifracopy/c1_dvir_dec")
        return

    # input - decoded data from c_1
    # output - converts it to "ACGT" alphabet using the GC-content 
    def convert_binary_from_c1_decoding(self):
        dict_2 = {'00': 'A', '01': 'C', '10': 'G', '11': 'T'}

        with open("c1_decoding_out.bin", "rb") as f:
            numpy_data = list(np.fromfile(f, np.dtype('B')))
            x = list(map(lambda y: f'{y:08b}', numpy_data))
        binary_str_data = ''.join(x)
        tmp = {}
        total_block_len_in_bits = self.block_size * 16

        for i in range(self.number_of_long_rows+self.number_of_short_rows):
            tmp.update({i: ''})

        for n in range(int(self.number_of_RS_symbols_in_a_row)):
            current = binary_str_data[n * total_block_len_in_bits: (n + 1) * total_block_len_in_bits]
            ind = 0
            for i in range(0, self.number_of_long_rows+self.number_of_short_rows):
                tmp.update({i: tmp.get(i) + current[ind: ind + 16]})
                ind += 16

        data_tmp= self.data_after_gc_dec
        multi_sub_indices = []
        self.data_after_c1_dec = []


        for i in range(self.number_of_long_rows+self.number_of_short_rows):
            bin_str = tmp.get(i)
            acgt_str = ''
            for j in range(0, len(bin_str), 2):
                acgt_str += dict_2.get(bin_str[j:j + 2])

            self.data_after_c1_dec.append(acgt_str)
            if not data_tmp[i].startswith(acgt_str):
                multi_sub_indices.append(i)
        print("total: " + str(len(multi_sub_indices)))
        self.mul_sub_indices=multi_sub_indices
        return

    # this function convert back to GC content for the final step of the decoding
    def convert_data_to_gc_content(self):
        dict_1 = {'A': '00', 'C': '01', 'G': '10', 'T': '11'}

        data = self.data_after_c1_dec
        min_val = 1
        max_val = -1
        self.data_after_c1_and_gc_dec = []

        line_ind = 0
        for strand in data:
            encoded_strand = ''
            encoded_strand_len = 0
            gc_counter = 0
            strand = strand.strip()
            strand = self.replace_all(strand, dict_1)
            x = gc_dict_13_7.get(strand[0:13])
            if type(x) is tuple:
                encoded_strand += x[0]
                gc_counter += x[0].count('G') + x[0].count('C')
            else:
                encoded_strand += x
                gc_counter += x.count('G') + x.count('C')
            encoded_strand_len += 7

            for i in range(13, self.information_bits_in_short_row, 13): # todo: fix that
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

            min_val = min(gc_counter / encoded_strand_len, min_val)
            max_val = max(gc_counter / encoded_strand_len, max_val)
            line_ind += 1

            self.data_after_c1_and_gc_dec.append(encoded_strand)
        print("Min GC content val: " + str(min_val))
        print("Max GC content val: " + str(max_val))

    # Corrects the last column using matrix H, after the RS correct of C_1
    def convert_binary_from_c1_decoding_data_B(self, matrix_numpy_file="H_matrix_identity.npy"):
        dict_5 = {'A': '0 ', 'C': '1 ', 'G': '2 ', 'T': '3 '}
        dict_6 = {'0': 'A', '1': 'C', '2': 'G', '3': 'T'}

        # the final conversion to GC content, to ensure that all errors have been corrected
        self.convert_data_to_gc_content()
        tmp2 =self.syndrome_phantom_vec

        data_after_gc = self.data_after_c1_and_gc_dec
        data=self.data_after_c1_dec
        GF4 = galois.GF(2 ** 2)
        H=GF4(np.load(open(matrix_numpy_file, 'rb')))
        quick_dict = {'00': 'A', '01': 'C', '10': 'G', '11': 'T'}
        cols_to_delete = [col for col in range(128-16, 128)]
        sub_H = np.delete(H, cols_to_delete, 1)
        self.data_after_c1_dec_gc_dec_B=[]

        # this function organize the data in its original form for the final decoding step
        for i in range(self.number_of_long_rows+self.number_of_short_rows):
            acgt_str = data[i][:int(self.information_bits_in_short_row/2)].strip() #todo: fix that
            if i < self.number_of_long_rows:
                x = data_after_gc[i][:self.number_of_bases_in_short_row].strip()
                # if i in multi_sub_indices:
                #     print("x_before, len: " + str(len(x)) + " , val: " + str(x))
                # if i in multi_sub_indices:
                #     x = 112*'A'
                x = self.replace_all(x, dict_5)
                sub_s = GF4(np.fromstring(x, dtype=int, sep=' '))
                m = sub_H @ sub_s
                # if i in multi_sub_indices:
                #     print("m, len: "+ str(len(m))+" , val: "+str(m))
                #     print("sub_s, len: " + str(len(sub_s)) + " , val: " + str(sub_s))
                #     print("x, len: " + str(len(x)) + " , val: " + str(x))
                t2 = GF4(np.fromstring(tmp2.get(i), dtype=int, sep=' '))
                b2 = np.array(np.add(t2, m))
                ans2 = ''.join(map(str, b2))
                ans2 = self.replace_all(ans2, dict_6)
                decoded_syndrome2 = ''
                for i in range(0, 16, 8):
                    decoded_syndrome2 += reversed_gc_dict_15_8.get(ans2[i:i + 8])
                for j in range(0, len(decoded_syndrome2), 2):
                    acgt_str += quick_dict.get(decoded_syndrome2[j:j + 2])

            self.data_after_c1_dec_gc_dec_B.append(acgt_str)

    # the function conver the data to binary form
    def convert_result_to_binary(self, out_put_file_path="output_file.bin"):
        data = self.data_after_c1_dec_gc_dec_B
        s1 = ''
        for line in data:
            block = line.strip()
            block = self.replace_all(block, {'A': '00', 'C': '01', 'G': '10', 'T': '11'})
            s1 += block
        index_first = s1.find('1')
        b = s1[index_first+1:]
        array = bytearray(int(b[x:x + 8], 2) for x in range(0, len(b), 8))

        with open(out_put_file_path, "wb") as fout:
            fout.write(array)

    # todo : can itay send us the numpy array directly?

    # returns the predicted sequence from the probs
    def get_pred(self, pred_probs=[], length=128):
        intToDNA = {'A': "A", 0: "A", 'C': "C", 1: "C", 'G': "G", 2: "G", 'T': "T", 3: "T"}
        pred = ""
        for k in range(length):
            pred_index = [pred_probs[j][k] for j in [0, 1, 2, 3]]
            pred_val = np.argmax(pred_index)  # , key=map_index)
            pred = pred + intToDNA[pred_val]
        return pred

    # returns the product of the probs to measure the confidence of the DNN
    def probs_product(self, pred_probs, length=128):
        prod = 1
        for k in range(length):
            pred_index = [pred_probs[j][k] for j in [0, 1, 2, 3]]
            prod = prod * np.max(pred_index)
        return prod

    def prob_mean(self, pred_probs, length):
        sum = 0
        for k in range(length):
            pred_index = [pred_probs[j][k] for j in [0, 1, 2, 3]]
            sum = sum + np.max(pred_index)
        count = sum / length
        return count

    def get_labels_from_cpl(self, res_files_path, confidence_level):
        #res_files_path = "/Users/omersabary/Documents/dataNanopore-pilotSep11.nosync/full_omer_alex_dec14/Dec14-Combined/res_CPL/"
        f_fail = open(res_files_path+"output-results-fail.txt", "r")
        f_success = open(res_files_path+"output-results-success.txt", "r")
        f_fail_pred = f_fail.readlines()
        f_success_pred = f_success.readlines()
        predictions = f_fail_pred[2::5] +f_success_pred[2::5]
        predictions = [pred.rstrip() for pred in predictions if len(pred.rstrip())==140]
        predictions = [pred.rstrip() for pred in predictions if pred[11] in {'A','C'}]
        obs_indices= [pred[:11] for pred in predictions]
        obs_indices = [ind for ind in obs_indices if ind in ind_to_num.keys()]
        obs_numerical_indices = [ind_to_num[ind] for ind in obs_indices]

        era_lines = [] #erased_numerice_index
        self.data=[]
        for i in tqdm(range(self.block_size)):
            if i not in obs_numerical_indices:
                era_lines.append(i)
                self.data.append("A" * 128)
            else:
                i_ind = obs_numerical_indices.index(i)
                if len(predictions[i_ind][12:])!=128:
                    self.data.append("A" * 128)
                    era_lines.append(i)
                else:
                    self.data.append(predictions[i_ind][12:])

                #print(len(predictions[i_ind][12:]))
        print(len(era_lines))
        self.erasures_lines = era_lines
        #exit(0)

    # todo: parallerization ??
    # input - path of inference file and confidence level
    # output - organized data for the decoding step and list of erased lines
    def get_labels_from_inf_files(self, inf_files_path, confidence_level = 0.25, file_1=True):

        pred_files = [f for f in os.listdir(inf_files_path)]
        print(len(pred_files))

        pred_files = [f for f in pred_files if f[-6] in {'G','T'}]
        print(len(pred_files))
        pred_files_sort =  sorted(pred_files)

        i = 0
        era_lines = [] #erased_numerice_index
        self.data=[]
        conf_filter = 0
        conf_diff = 0
        for json_file in tqdm(pred_files_sort):
            path = inf_files_path + '/' + json_file
            with open(path) as f:
                inf_data = json.load(f)
                inf_index = inf_data["index"][:-1] #todo:fix!!
                numeric_index = ind_to_num[inf_index]

                if numeric_index > i :
                    for j in range(i, numeric_index):
                        era_lines.append(j)
                        self.data.append("A"*128)
                    i = numeric_index
                confidence_func = (self.prob_mean(pred_probs=inf_data["pred_probs"],
                                             length=self.number_of_bases_in_long_row))**(2* len(inf_data["noisy_copies"]))
                if confidence_func < confidence_level or len(inf_data["noisy_copies"])<=4:
                    conf_filter = conf_filter+1
                    era_lines.append(i)
                    self.data.append("A" * 128)
                else:
                    pred = self.get_pred(pred_probs=inf_data["pred_probs"], length=self.number_of_bases_in_long_row)
                    self.data.append(pred)

                i=i+1


        for j in range(i, self.block_size):
            era_lines.append(j)
            self.data.append("A" * 128)
        self.erasures_lines = era_lines
        print(len(era_lines))
        print("conf_filtered:" +str(conf_filter))
        print("conf diff: "+ str(conf_diff))
        return

    ## inference from both CPL and DNN filter size
    def get_labels_from_inf_files_with_cpl_filter_size(self, inf_files_path, confidence_level = 0.7,
                                           confidence_cluster_size=4,
                                           res_cpl_path="",  file_char={'A', 'C'}):

        f_fail = open(res_cpl_path + "output-results-fail.txt", "r")
        f_success = open(res_cpl_path + "output-results-success.txt", "r")
        f_fail_pred = f_fail.readlines()
        f_success_pred = f_success.readlines()
        predictions = f_fail_pred[2::5] + f_success_pred[2::5]
        predictions = [pred.rstrip() for pred in predictions if pred[11] in file_char]
        obs_indices = [pred[:11] for pred in predictions]

        pred_files = [f for f in os.listdir(inf_files_path)]
        print(len(pred_files))

        pred_files = [f for f in pred_files if f[-6] in file_char]
        print(len(pred_files))
        pred_files_sort =  sorted(pred_files)

        i = 0
        era_lines = [] #erased_numerice_index
        self.data=[]
        conf_filter = 0
        conf_diff = 0
        cpl_correct = 0
        cpl_wrong = 0
        for json_file in tqdm(pred_files_sort):
            path = inf_files_path + '/' + json_file
            with open(path) as f:
                inf_data = json.load(f)
                inf_index = inf_data["index"][:-1]
                numeric_index = ind_to_num[inf_index]

                if numeric_index > i :
                    for j in range(i, numeric_index):
                        era_lines.append(j)
                        self.data.append("A"*128)
                    i = numeric_index
                confidence_func = (self.prob_mean(pred_probs=inf_data["pred_probs"],
                                             length=self.number_of_bases_in_long_row))**(2*len(inf_data["noisy_copies"]))
                if len(inf_data["noisy_copies"]) <= confidence_cluster_size:
                    conf_filter = conf_filter + 1
                    era_lines.append(i)
                    self.data.append("A" * 128)
                elif confidence_func <=confidence_level:
                    idx = num_to_ind[i]
                    if idx in obs_indices:
                        i_ind = obs_indices.index(idx)
                        if len(predictions[i_ind][12:]) != 128:
                            cpl_wrong = cpl_wrong + 1
                            self.data.append("A" * 128)
                            era_lines.append(i)
                        else:
                            cpl_correct = cpl_correct + 1
                            self.data.append(predictions[i_ind][12:])
                    else:
                        cpl_wrong = cpl_wrong + 1
                        self.data.append("A" * 128)
                        era_lines.append(i)
                else:
                    pred = self.get_pred(pred_probs=inf_data["pred_probs"], length=self.number_of_bases_in_long_row)
                    self.data.append(pred)

                i=i+1


        for j in range(i, self.block_size):
            era_lines.append(j)
            self.data.append("A" * 128)
        self.erasures_lines = era_lines
        print(len(era_lines))
        print("conf_filtered:" +str(conf_filter))
        print("conf diff: "+ str(conf_diff))
        print("cpl correct:"+str(cpl_correct))
        print("cpl wrong:"+str(cpl_wrong))
        return


    ## inference from both CPL and DNN
    def get_labels_from_inf_files_with_cpl(self, inf_files_path, confidence_level = 0.35,
                                           confidence_cluster_size=4,
                                           res_cpl_path="",  file_char={'A', 'C'}):

        f_fail = open(res_cpl_path + "output-results-fail.txt", "r")
        f_success = open(res_cpl_path + "output-results-success.txt", "r")
        f_fail_pred = f_fail.readlines()
        f_success_pred = f_success.readlines()
        predictions = f_fail_pred[2::5] + f_success_pred[2::5]
        predictions = [pred.rstrip() for pred in predictions if pred[11] in file_char]
        obs_indices = [pred[:11] for pred in predictions]


        pred_files = [f for f in os.listdir(inf_files_path)]
        print(len(pred_files))

        pred_files = [f for f in pred_files if f[-6] in file_char]
        print(len(pred_files))
        pred_files_sort =  sorted(pred_files)

        i = 0
        era_lines = [] #erased_numerice_index
        self.data=[]
        new_erasures = 0
        conf_filter = 0
        conf_diff = 0
        cpl_correct = 0
        cpl_wrong = 0
        for json_file in tqdm(pred_files_sort):
            path = inf_files_path + '/' + json_file
            with open(path) as f:
                inf_data = json.load(f)
                inf_index = inf_data["index"][:-1] #todo:fix!!
                numeric_index = ind_to_num[inf_index]

                if numeric_index > i :
                    for j in range(i, numeric_index):
                        era_lines.append(j)
                        self.data.append("A"*128)
                    i = numeric_index
                confidence_func = (self.prob_mean(pred_probs=inf_data["pred_probs"],
                                             length=self.number_of_bases_in_long_row))**(2*len(inf_data["noisy_copies"]))
                if confidence_func <=confidence_level:
                    conf_filter = conf_filter + 1
                    if len(inf_data["noisy_copies"])<=confidence_cluster_size:
                        new_erasures = new_erasures+1
                        era_lines.append(i)
                        self.data.append("A" * 128)
                    else:
                        idx = num_to_ind[i]
                        if idx in obs_indices:
                            i_ind = obs_indices.index(idx)
                            if len(predictions[i_ind][12:]) != 128:
                                cpl_wrong = cpl_wrong + 1
                                self.data.append("A" * 128)
                                era_lines.append(i)
                            else:
                                cpl_correct = cpl_correct + 1
                                self.data.append(predictions[i_ind][12:])
                        else:
                            cpl_wrong = cpl_wrong + 1
                            self.data.append("A" * 128)
                            era_lines.append(i)
                            #print(idx + str(" is not in obs_indices"))
                        #print("conf fil  cpl")

                else:
                    pred = self.get_pred(pred_probs=inf_data["pred_probs"], length=self.number_of_bases_in_long_row)
                    self.data.append(pred)

                i=i+1


        for j in range(i, self.block_size):
            era_lines.append(j)
            self.data.append("A" * 128)
        self.erasures_lines = era_lines
        f_log.write("number of erasure lines:"+str(len(era_lines))+"\n")
        f_log.write("conf_filtered:" +str(conf_filter)+"\n")
        f_log.write("erased by cpl/conf: "+ str(new_erasures)+"\n")
        f_log.write("cpl correct:"+str(cpl_correct)+"\n")
        f_log.write("cpl wrong:"+str(cpl_wrong)+"\n")

        file_sub = open("/Users/omersabary/Desktop/DaniellaDvir/test_finale.nosync/two_files.nosync/data_after_indices_1.txt", "r")
        counter_correct = 0
        counter_wrong = 0
        for line_data, line_correct in zip(self.data, file_sub.readlines()):
            #print(line_data.rstrip())
            #print(line_correct[12:].rstrip())
            if line_data.rstrip() == line_correct[12:].rstrip():
                counter_correct = counter_correct +1
            else:
                counter_wrong = counter_wrong +1

        f_log.write("total_correct "+ str(counter_correct)+"\n")
        f_log.write("total_wrong "+ str(counter_wrong)+"\n")
        return


    # the function generates file with the numerical indices of the erased for the c_1 and c_2 decoders
    def generate_erasure_files_for_codec(self):
        M = self.number_of_long_rows
        t1 = int(self.number_of_short_rows / 2)
        t2 = self.rs_redundancy
        print("total erasures:" +str(len(self.erasures_lines)))
        with open('c1_erasures_locations.txt', 'w') as f1:
            with open('c2_erasures_locations.txt', 'w') as f2:
                tmp1 = self.erasures_lines
                tmp1 = list(map(lambda x: x if x < M else x + (self.RS_block_size - (M + 2 * t1 + t2)), tmp1))
                tmp1.sort()
                for line_ind in tmp1:
                    f2.write(str(line_ind) + '\n')

                tmp2 = self.erasures_lines
                tmp2 = list(map(lambda x: x if x < M + 2 * t1 else x + (self.RS_block_size - (M + 2 * t1 + t2)), tmp2))
                tmp2.sort()
                for line_ind in tmp2:
                    f1.write(str(line_ind) + '\n')
        return self.erasures_lines




def get_syndromes_dict():
    t1 = time.time()
    with open("/Users/omersabary/Dropbox/for_omer/syndromes_dict_identity.pkl", "rb") as f:
        syndromes_dict = pickle.load(f)
    t2 = time.time()
    print("syndromes loading time: " + str(t2 - t1))
    return syndromes_dict

def convert_key_to_int(syndrome):
    num = 0
    dic_4_to_b={'0': 0, '1': 1, '2': 2, '3': 3}
    lst = []
    for bit in syndrome:
        #print(bit)
        num = (num + (dic_4_to_b[bit])) << 2
    num = num >>2
    return num



# input:
# main function - this function builds the decoder from the given parameters - fraction of correctable erasures, correctable subsitutitions, and correctable "almost" correct sub (up to 3 errors)
#                   the values should be number between 0 to 1, where the sum of them is strictly smaller than 1.
# inf path - is the path of the inference files
# conf_level - the thresholds of the confidence of the decoder - should be a number from 0 to 1
# h_matrix path - matrix of the TP code - should be a .npy file
# syndrom table - pickle file with the possible syndrome - used for the syndrome-based decoding
# note that this function assume g++-7 is installed on the machine

# output:
# the decoded file can be found as "output_file.bin" or at user defined path
def decode(erasure_fraction=0.04, substitution_fraction=0.0075, almost_correct_fraction=0.016,
                      inf_path="/Users/omersabary/Desktop/DaniellaDvir/test_finale/pilot_nanopore.DNAformer_s_v1_L.ce_sl1.size_32.pad_seq/test_omer_5/", conf_level=0.7, conf_size = 4,
                      h_matrix_path="H_matrix_identity.npy",
                      syndrom_table_pkl_path="/Users/omersabary/Dropbox/for_omer/syndromes_dict_identity.pkl",
                      out_put_file_path="output_file.bin", res_cpl_path="/Users/omersabary/PycharmProjects/pythonProject1/res_Omer/SingleRunNov10/res_CPL/",  file_iden ={'A', 'C'}):
    st = time.time()
    Dec = decoder(erasure_fraction, substitution_fraction, almost_correct_fraction)


    Dec.get_labels_from_inf_files_with_cpl(inf_files_path=inf_path, confidence_level= conf_level,
                                           confidence_cluster_size=conf_size,
                                           res_cpl_path=res_cpl_path,
                                           file_char=file_iden)
    #Dec.read_data_to_decode("/Users/omersabary/Documents/test_omer_dvir/errornous.txt")
    Dec.generate_erasure_files_for_codec()
    Dec.convert_data_to_binary_in_format_for_c2_decoding(h_matrix_path)
    Dec.perform_c2_dec()
    Dec.get_syndromes_dict(syndrom_table_pkl_path)
    Dec.get_syndromes_dict_after_c2_decoding()
    Dec.convert_binary_from_c2_decoding()
    Dec.convert_from_gc_content()
    Dec.convert_data_to_binary_in_format_for_c1_decoding()
    Dec.perform_c1_dec()
    Dec.convert_binary_from_c1_decoding()
    Dec.convert_binary_from_c1_decoding_data_B()
    Dec.convert_result_to_binary(out_put_file_path)
    en = time.time()
    print(en - st)
    return

f_log = open("log_run.txt", "w")

## Code parameters that were used in our experiment

erasure_fraction=0.04
substitution_fraction=0.0075
almost_correct_fraction=0.016

## Inference_path
inf_path = "/Users/omersabary/Documents/dataNanopore-pilotSep11/inf_itai_final/Nanopore_single_flowcell_v2"

## CPL resulsts path
CPL_path = "/Users/omersabary/Documents/dataNanopore-pilotSep11/full_omer_alex_hadas/full_poolNov10/Single_Flow_Cell_Nov10_only/res_CPL/"

# Code utilities
h_matrix_path = "H_matrix_identity.npy"
## syndrome table path - used to decoding the syndrome, can be replace by multiplying the matrix.
## the syndrome table can be download from here this link, https://www.dropbox.com/s/rml0dbnvhl5kwet/syndromes_dict_identity.pkl?dl=0
## another link : https://drive.google.com/file/d/1QgJRKgvm8T2MHrwDhyYFveK0CMIY3C-J/view?usp=sharing
## then this path should be replaced.
syndrom_table_pkl_path = "/Users/omersabary/Dropbox/for_omer/syndromes_dict_identity.pkl"

# output file path
out_put_file_path = "out_decoder.bin"

#confidence parameter
conf_filter = 0.7
conf_size = 4

# file identifier
file_identifier = {'G', 'T'}

decode(erasure_fraction=erasure_fraction, substitution_fraction=substitution_fraction, almost_correct_fraction=almost_correct_fraction,
       inf_path = inf_path, res_cpl_path=CPL_path, h_matrix_path=h_matrix_path, syndrom_table_pkl_path=syndrom_table_pkl_path,
       out_put_file_path=out_put_file_path, conf_level=0.7, conf_size=4, file_iden = file_identifier)




