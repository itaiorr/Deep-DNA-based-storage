from tqdm import tqdm

def copy_reverse_compliment(cp):
    cp = cp.rstrip()
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    bases = list(cp)
    bases = [complement[base] for base in bases]
    cp=''.join(bases)
    #def reverse_complement(s):
    return (cp[::-1])

def get_insertion_ball(s):
    ins_ball = []
    for i in range(len(s)):
        for inserted_char in {'A', 'C', 'G', 'T'}:
            list_s = list(s)
            list_s.insert(i, inserted_char)
            ins_ball.append(''.join(list_s))
    return ins_ball

def get_substitution_ball(s):
    sub_ball = []
    for i in range(len(s)):
        curr_char=s[i]
        for replaced_char in subsitution_dictionaries[curr_char]:
            list_s = list(s)
            list_s[i] = replaced_char
            sub_ball.append(''.join(list_s))
        #s[i] = curr_char
    return sub_ball

def get_del_ball(s):
    del_ball = []
    for i in range(len(s)):
        #for inserted_char in {'A', 'C', 'G', 'T'}:
        list_s = list(s)
        del list_s[i]
        del_ball.append(''.join(list_s))
    return del_ball


def read_preprocessor(file_input_path='./Calls.fastq', preprocessed_reads_path='./reads_processed.fastq',    lib_length = 140):
    file_input_fastq =open(file_input_path, "r")

    file_output_without_primers= open(preprocessed_reads_path, "w")


    #front_primer = "TCGTCGGCAGCGTCAGATGTGTATAAGAGACAG"
    front_primer ="TAAGAGACAG"
    #front_primer ="AAGAGACAG"

    #back_primer = "CTGTCTCTTATACACATCTCCGAGCCCACGAGAC"
    back_primer = "CTGTCTCTTA"
    #back_primer = "CTGTCTCTT"

    front_primer_rev_com = copy_reverse_compliment(front_primer) #"TAAGAGACAG"
    back_primer_rev_com =  copy_reverse_compliment(back_primer) #"CTGTCTCTTA"

    subsitution_dictionaries={
        'A': {'C', 'G', 'T'},
        'C': {'A', 'G', 'T'},
        'G': {'A', 'C', 'T'},
        'T': {'A', 'C', 'G'}
    }







    count_filter =0
    for line in tqdm(file_input_fastq.readlines()[1::4]):
        front_ind = line.find(front_primer)
        back_ind = line.find(back_primer)
        if front_ind != -1 and back_ind != -1:
            seq_with_primers = line[front_ind:back_ind+len(back_primer)]

            seq_without_primers = line[front_ind+len(front_primer): back_ind]
            if abs(len(seq_without_primers) - lib_length) <=5:
                file_output_without_primers.write(seq_without_primers+"\n")
        elif front_ind != -1:
            seq_without_primers = line[front_ind+len(front_primer): min(len(line),front_ind+len(front_primer)+lib_length)]
            if abs(len(seq_without_primers) - lib_length) <= 5:

                file_output_without_primers.write(seq_without_primers + "\n")
        elif back_ind!=-1:
            seq_without_primers = line[max(0, back_ind - lib_length):back_ind]

            if abs(len(seq_without_primers) - lib_length) <= 5:
                file_output_without_primers.write(seq_without_primers + "\n")

        else:
            line = copy_reverse_compliment(line)
            front_ind = line.find(front_primer)
            back_ind = line.find(back_primer)
            if front_ind != -1 and back_ind != -1:

                seq_without_primers = line[front_ind + len(front_primer): back_ind]
                if abs(len(seq_without_primers) - lib_length) <= 5:
                    file_output_without_primers.write(seq_without_primers + "\n")
            elif front_ind != -1:
                seq_without_primers = line[front_ind + len(front_primer): min(len(line), front_ind + len(front_primer) + lib_length)]
                if abs(len(seq_without_primers) - lib_length) <= 5:
                    file_output_without_primers.write(seq_without_primers + "\n")
            elif back_ind!=-1:
                seq_without_primers = line[max(0, back_ind - lib_length):back_ind]
                if abs(len(seq_without_primers) - lib_length) <= 5:
                    file_output_without_primers.write(seq_without_primers + "\n")
            else:

                count_filter=count_filter+1

    file_input_fastq.close()
    file_output_without_primers.close()
