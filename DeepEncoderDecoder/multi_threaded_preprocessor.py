import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def copy_reverse_compliment(cp):
    cp = cp.rstrip()
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    bases = [complement[base] for base in cp]
    return ''.join(bases)[::-1]

def get_insertion_ball(s):
    ins_ball = []
    for i in range(len(s)):
        for inserted_char in {'A', 'C', 'G', 'T'}:
            list_s = list(s)
            list_s.insert(i, inserted_char)
            ins_ball.append(''.join(list_s))
    return ins_ball

def get_substitution_ball(s, substitution_dictionaries):
    sub_ball = []
    for i in range(len(s)):
        curr_char = s[i]
        for replaced_char in substitution_dictionaries[curr_char]:
            list_s = list(s)
            list_s[i] = replaced_char
            sub_ball.append(''.join(list_s))
    return sub_ball

def get_del_ball(s):
    del_ball = []
    for i in range(len(s)):
        list_s = list(s)
        del list_s[i]
        del_ball.append(''.join(list_s))
    return del_ball

def process_line(line, front_primer, back_primer, lib_length):
    substitution_dictionaries = {
        'A': {'C', 'G', 'T'},
        'C': {'A', 'G', 'T'},
        'G': {'A', 'C', 'T'},
        'T': {'A', 'C', 'G'}
    }

    front_primer_rev_com = copy_reverse_compliment(front_primer)
    back_primer_rev_com = copy_reverse_compliment(back_primer)

    front_ind = line.find(front_primer)
    back_ind = line.find(back_primer)
    if front_ind != -1 and back_ind != -1:
        seq_without_primers = line[front_ind+len(front_primer): back_ind]
        if abs(len(seq_without_primers) - lib_length) <= 5:
            return seq_without_primers

    elif front_ind != -1:
        seq_without_primers = line[front_ind+len(front_primer): min(len(line), front_ind+len(front_primer)+lib_length)]
        if abs(len(seq_without_primers) - lib_length) <= 5:
            return seq_without_primers

    elif back_ind != -1:
        seq_without_primers = line[max(0, back_ind - lib_length):back_ind]
        if abs(len(seq_without_primers) - lib_length) <= 5:
            return seq_without_primers

    else:
        line = copy_reverse_compliment(line)
        front_ind = line.find(front_primer)
        back_ind = line.find(back_primer)
        if front_ind != -1 and back_ind != -1:
            seq_without_primers = line[front_ind+len(front_primer): back_ind]
            if abs(len(seq_without_primers) - lib_length) <= 5:
                return seq_without_primers

        elif front_ind != -1:
            seq_without_primers = line[front_ind+len(front_primer): min(len(line), front_ind+len(front_primer)+lib_length)]
            if abs(len(seq_without_primers) - lib_length) <= 5:
                return seq_without_primers

        elif back_ind != -1:
            seq_without_primers = line[max(0, back_ind - lib_length):back_ind]
            if abs(len(seq_without_primers) - lib_length) <= 5:
                return seq_without_primers

    return None

def read_preprocessor(file_input_path='./Calls.fastq', preprocessed_reads_path='./reads_processed.fastq', lib_length=140):
    file_output_without_primers = open(preprocessed_reads_path, "w")
    front_primer = "TAAGAGACAG"
    back_primer = "CTGTCTCTTA"

    if os.path.isfile(file_input_path):
        print(f"The path '{file_input_path}' is a file.")
        file_input_fastq = open(file_input_path, "r")
        lines = file_input_fastq.readlines()[1::4]

        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_line, lines, [front_primer] * len(lines),
                                             [back_primer] * len(lines), [lib_length] * len(lines)),
                                total=len(lines), desc="Processing reads"))

        for result in results:
            if result is not None:
                file_output_without_primers.write(result + "\n")

        file_input_fastq.close()
        file_output_without_primers.close()

    elif os.path.isdir(file_input_path):
        print(f"The path '{file_input_path}' is a folder.")
        for file_name in os.listdir(file_input_path):
            if file_name.endswith(".fastq"):
                file_path = os.path.join(file_input_path, file_name)
                file_input_fastq = open(file_path, "r")
                lines = file_input_fastq.readlines()[1::4]

                with ThreadPoolExecutor() as executor:
                    results = list(tqdm(executor.map(process_line, lines, [front_primer] * len(lines),
                                                     [back_primer] * len(lines), [lib_length] * len(lines)),
                                        total=len(lines), desc="Processing reads"))

                for result in results:
                    if result is not None:
                        file_output_without_primers.write(result + "\n")

                file_input_fastq.close()

    else:
        print(f"The path '{file_input_path}' is neither a file nor a folder, or it doesn't exist.")


    file_output_without_primers.close()

# Running the script
if __name__ == '__main__':
    read_preprocessor()
