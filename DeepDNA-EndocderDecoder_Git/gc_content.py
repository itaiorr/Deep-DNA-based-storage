import re
import itertools


################################# GC CONTENT ################################################3
def generate_without_homopolymers(length=12):
    # homopolyner is a sequence of a same symbol in length 5 or higher
    ans = []
    alphabet = "ACGT"
    homopolymer_re = r'A{5,%d}' % length + '|' + r'C{5,%d}' % length + '|' + r'G{5,%d}' % length + '|' + r'T{5,%d}' % length
    for index, item in enumerate(itertools.product(alphabet, repeat=length)):
        elem = "".join(item)
        if len(re.findall(homopolymer_re, elem)) == 0:
            if not elem.startswith(('AAA', 'CCC', 'GGG', 'TTT')) and not elem.endswith(('AAA', 'CCC', 'GGG', 'TTT')):
                ans.append(elem)
    return ans


def generate_GC_conent_groups(length):
    gc_count_dict = {}
    for i in range(length + 1):
        gc_count_dict.update({i: []})
    valid_words = generate_without_homopolymers(length)
    for elem in valid_words:
        GC_count = elem.count('G') + elem.count('C')
        gc_count_dict[GC_count].append(elem)
    return gc_count_dict


def create_gc_content_map_15_8(gc_map, length):
    gc_dict = {}
    reversed_gc_dict = {}
    counter_3_5 = 0
    counter_4 = 0
    counter_2_6 = 0

    gc_map_2 = gc_map.get(2)
    GC_2_len = len(gc_map_2)
    gc_map_3 = gc_map.get(3)
    GC_3_len = len(gc_map_3)
    gc_map_4 = gc_map.get(4)
    GC_4_len = len(gc_map_4)
    gc_map_5 = gc_map.get(5)
    GC_5_len = len(gc_map_5)
    gc_map_6 = gc_map.get(6)
    GC_6_len = len(gc_map_6)
    alphabet = [0, 1]
    for item in itertools.product(alphabet, repeat=length):
        str_item = ''.join(map(str, item))
        if counter_4 < GC_4_len:
            gc_dict.update({str_item: gc_map_4[counter_4]})
            reversed_gc_dict.update({gc_map_4[counter_4]: str_item})
            counter_4 += 1
            continue
        elif counter_3_5 < GC_3_len:
            gc_dict.update({str_item: (gc_map_3[counter_3_5], gc_map_5[counter_3_5])})
            reversed_gc_dict.update({gc_map_3[counter_3_5]: str_item})
            reversed_gc_dict.update({gc_map_5[counter_3_5]: str_item})
            counter_3_5 += 1
            continue
        else:
            gc_dict.update({str_item: (gc_map_2[counter_2_6], gc_map_6[counter_2_6])})
            reversed_gc_dict.update({gc_map_2[counter_2_6]: str_item})
            reversed_gc_dict.update({gc_map_6[counter_2_6]: str_item})
            counter_2_6 += 1
            continue
    return gc_dict, reversed_gc_dict


def create_gc_content_map_13_7(gc_map, lenght):
    gc_dict = {}
    reversed_gc_dict = {}
    counter_1 = 0
    counter_2 = 0
    counter_3 = 0
    counter_4 = 0
    counter_5 = 0
    counter_6 = 0
    gc_map_1 = gc_map.get(1)
    GC_1_len = len(gc_map_1)
    gc_map_2 = gc_map.get(2)
    GC_2_len = len(gc_map_2)
    gc_map_3 = gc_map.get(3)
    GC_3_len = len(gc_map_3)
    gc_map_4 = gc_map.get(4)
    GC_4_len = len(gc_map_4)
    gc_map_5 = gc_map.get(5)
    GC_5_len = len(gc_map_5)
    gc_map_6 = gc_map.get(6)
    GC_6_len = len(gc_map_6)
    alphabet = [0, 1]
    for item in itertools.product(alphabet, repeat=lenght):
        str_item = ''.join(map(str, item))
        if counter_3 < GC_3_len:
            if counter_1 < GC_1_len:
                gc_dict.update({str_item: (gc_map_3[counter_3], gc_map_1[counter_1])})
                reversed_gc_dict.update({gc_map_3[counter_3]: str_item})
                reversed_gc_dict.update({gc_map_1[counter_1]: str_item})
                counter_1 += 1
                counter_3 += 1
                continue
            elif counter_5 < GC_5_len:
                gc_dict.update({str_item: (gc_map_3[counter_3], gc_map_5[counter_5])})
                reversed_gc_dict.update({gc_map_3[counter_3]: str_item})
                reversed_gc_dict.update({gc_map_5[counter_5]: str_item})
                counter_5 += 1
                counter_3 += 1
                continue
            else:
                gc_dict.update({str_item: gc_map_3[counter_3]})
                reversed_gc_dict.update({gc_map_3[counter_3]: str_item})
                counter_3 += 1
                continue
        else:
            if counter_2 < GC_2_len:
                gc_dict.update({str_item: (gc_map_4[counter_4], gc_map_2[counter_2])})
                reversed_gc_dict.update({gc_map_4[counter_4]: str_item})
                reversed_gc_dict.update({gc_map_2[counter_2]: str_item})
                counter_2 += 1
                counter_4 += 1
                continue
            elif counter_6 < GC_6_len:
                gc_dict.update({str_item: (gc_map_4[counter_4], gc_map_6[counter_6])})
                reversed_gc_dict.update({gc_map_4[counter_4]: str_item})
                reversed_gc_dict.update({gc_map_6[counter_6]: str_item})
                counter_6 += 1
                counter_4 += 1
                continue
            else:
                gc_dict.update({str_item: gc_map_4[counter_4]})
                reversed_gc_dict.update({gc_map_4[counter_4]: str_item})
                counter_4 += 1
    return gc_dict, reversed_gc_dict

gc_map_13_7 = generate_GC_conent_groups(7)
gc_dict_13_7, reversed_gc_dict_13_7 = create_gc_content_map_13_7(gc_map_13_7, 13)


gc_map_15_8 = generate_GC_conent_groups(8)
gc_dict_15_8, reversed_gc_dict_15_8 = create_gc_content_map_15_8(gc_map_15_8, 15)

f = open("dic_15_8.txt", "w")
f.write("{\n")
for key in gc_dict_15_8.keys():
    if str(gc_dict_15_8[key])[0]!='(':
        f.write("'" + str(key) + "' : '" + str(gc_dict_15_8[key]) + "',\n")
    else:
        f.write("'" + str(key) + "' : " + str(gc_dict_15_8[key]) + ",\n")

f.write("}\n")

#(gc_dict_15_8)
