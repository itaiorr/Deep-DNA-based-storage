from gc_maps import dic_13b_to_7q as gc_dict_13_7
from gc_maps import dic_15b_to_8q as gc_dict_15_8

dic_count = {}
for key in gc_dict_13_7.keys():
    if type(gc_dict_13_7[key]) is tuple:
        x0 = gc_dict_13_7[key][0]
        x1 = gc_dict_13_7[key][1]
        gc_counter0 = x0.count('G') + x0.count('C')
        gc_counter1 = x1.count('G') + x1.count('C')
        if gc_counter0 > gc_counter1:
            tmp = gc_counter0
            gc_counter0 = gc_counter1
            gc_counter1 = tmp
        if str(gc_counter0)+"_"+str(gc_counter1) in dic_count.keys():
            dic_count[str(gc_counter0)+"_"+str(gc_counter1)] = dic_count[str(gc_counter0)+"_"+str(gc_counter1)] +1
        else:
            dic_count[str(gc_counter0)+"_"+str(gc_counter1)] = 1
    else:
        x0 = gc_dict_13_7[key]
        gc_counter0 = x0.count('G') + x0.count('C')
        if str(gc_counter0) in dic_count.keys():
            dic_count[str(gc_counter0)] =   dic_count[str(gc_counter0)] + 1
        else:
            dic_count[str(gc_counter0)] =1

print(dic_count)



