import random
from datetime import datetime
import pickle
from indices_dic import ind_to_num

indices_exists=False


def get_indices():
    with open('indices.txt', 'r') as f:
        data = f.readlines()
    valid_indices = []
    for line in data:
        #if not line.strip().endswith(('AA', 'CC', 'GG', 'TT')):
        valid_indices.append(line.strip())
    valid_indices.sort()
    return valid_indices

def generate_indices():
    global indices_exists
    with open('data_after_c2_encoding.txt', 'r') as f:
        data = f.readlines()
        number_of_lines=len(data)

    if indices_exists:
        indices_dict = parse_indices()
        my_indices = list(indices_dict.keys())
        random_lines_real_indices = list(indices_dict.values())
        c1 = 'G'
        c2 = 'T'
    else:
        indices = get_indices()
        my_indices = indices[:number_of_lines]
        my_indices.sort()
        random_lines_real_indices = list(range(number_of_lines))
        random_lines_real_indices.sort()
        results = {}
        c1 = 'A'
        c2 = 'C'
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

            if found:
                if not indices_exists:
                    results.update({index_candidate: index})
            else:
                print("bugggg!!!")
    print("done!")

    if not indices_exists:
        with open('indices_dictionary.pkl', 'wb') as f:
            pickle.dump(results, f)

    indices_exists = True


def generate_indices_for_all():
    with open('data_after_c2_encoding.txt', 'r') as f:
        data = f.readlines()

    indices = get_indices()
    c1 = 'G'
    c2 = 'T'

    with open('data_after_indices_for_all.txt', 'w') as f:
        for counter, line in enumerate(data):
            index_candidate = indices[counter]
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

            if found:
                if not indices_exists:
                    results.update({index_candidate: index})
            else:
                print("bugggg!!!")
    print("done!")

def parse_indices():
    with open('indices_dictionary.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


def merge_files():
    indices_dict = parse_indices()
    tmp = {}
    index_len = 12
    with open('data_after_indices.txt', 'r') as f:
        data = f.readlines()
    for line in data:
        line = line.strip()
        line_index = line[:index_len]
        line_real_index = indices_dict.get(line_index)
        tmp.update({line_real_index: line[index_len:]})

    with open('data_after_c2_encoding.txt', 'r') as f:
        data = f.readlines()
    with open('data_after_some_noise.txt', 'w') as f:
        for line_ind, line in enumerate(data):
            if line_ind in tmp:
                f.write(tmp.get(line_ind) + '\n')
            else:
                f.write(line)


def add_primers(input_file='data_after_indices.txt', output_file="'data_after_primers.txt'"):
    front_primer = "TCGTCGGCAGCGTCAGATGTGTATAAGAGACAG"
    back_primer = "CTGTCTCTTATACACATCTCCGAGCCCACGAGAC"
    with open(input_file, 'r') as f:
        data = f.readlines()

    with open(output_file, 'w') as f:
        for line in data:
            f.write(front_primer + line.strip() + back_primer + '\n')

#generate_indices()

add_primers(input_file="/Users/omersabary/Desktop/DaniellaDvir/test_finale.nosync/Submission-Martine/MicrosoftPool/Centers.txt",
            output_file="/Users/omersabary/Desktop/DaniellaDvir/test_finale.nosync/Submission-Martine/MicrosoftPool/Centers_with_primers.txt")
exit(0)
f = open("data_after_indices.txt", "r")
lines = f.readlines()
lines = [line[:11] for line in lines]
f_out = open("used_indices_omer.txt", "w")
for line in lines:
    #print(ind_to_num[line.rstrip()])
    f_out.write(line+"\n")