

def GC_cont(line):
    count =0.0
    for c in line:
        if c=='G' or c=='C':
            count = count +1
    return count/len(line)

f = open("/Users/omersabary/Desktop/DaniellaDvir/test_finale.nosync/Submission-Martine/LargePool/data_after_indices_two_files.txt", "r")
lines = f.readlines()
for line in lines:

    if GC_cont(line) > 65 or GC_cont(line) <35:
        print("lowwww highhhh")


exit(0)
f = open("/Users/omersabary/Desktop/DaniellaDvir/test_finale.nosync/two_files.nosync/data_after_indices_1.txt", "r")
lines = f.readlines()
lines = [line[:12] for line in lines]
f_out = open("used_indices_file_1.txt", "w")
for line in lines:
    #print(ind_to_num[line.rstrip()])
    f_out.write(line+"\n")