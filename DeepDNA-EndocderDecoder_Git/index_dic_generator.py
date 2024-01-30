

f = open("indices.txt", "r")
indices = f.readlines()

num_to_ind = {}
ind_to_num = {}

for i, ind in enumerate(indices):
    num_to_ind[i] = ind.rstrip()
    ind_to_num[ind.rstrip()] = i

print(num_to_ind)
print(ind_to_num)

output_file = open("indices_dic.py", "w")
output_file.write("num_to_ind={\n")
for key in num_to_ind.keys():
    output_file.write(str(key) + ": '" +str(num_to_ind[key]+"', "))
output_file.write("}\n")
output_file.write("\n \n")

output_file.write("ind_to_num={\n")
for key in ind_to_num.keys():
    output_file.write("'"+str(key) + "': " +str(ind_to_num[key])+", ")
output_file.write("}\n")

