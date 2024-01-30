import editdistance
from tqdm import tqdm


f = open("indices.txt", "r")
lines = f.readlines()
print(lines)

min_edit_distance = 12
min_i = 0
min_j = 1
for i in tqdm(range(len(lines))):
    for j in range(i+1, len(lines)):
        dis = editdistance.distance(lines[i].rstrip(), lines[j].rstrip())
        if dis < min_edit_distance:
            print(dis)
            min_edit_distance= dis
            min_i = i
            min_j = j
print("minimum")
print(dis)
print(min_i)
print(min_j)