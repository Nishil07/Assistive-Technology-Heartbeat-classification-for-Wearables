import os
path_dir = "C:/Users/Eshika/Downloads/MLII"
all_pats = []
for i in os.listdir("C:/Users/Eshika/Downloads/MLII"):
    pat_id = os.listdir(path_dir + "/" +i)
    pat_id = [all_pats.append(x[0:3]) for x in pat_id]

all_pats = sorted(set(all_pats))
for i in all_pats:
    col_count = 0
    row_count = 0
    col_names = []
    for j in os.listdir("C:/Users/Eshika/Downloads/MLII"):
        flag = 0
        mat_count = 0
        for x in os.listdir(path_dir + "/" +j):
            if x.startswith(i):
                flag = 1
                mat_count+=1
        if flag == 1:
            col_names.append(j)
            col_count += 1
            if row_count < 3600*mat_count:
                row_count = 3600*mat_count
    
    print(str(i) + " " + str(col_count) + " " + str(row_count) + " " + str(col_names))
