import os

# path = "C:\\Users\\Administrator\\Desktop\\result\\GE\\GE_point_txt"
# save_path = "C:\\Users\\Administrator\\Desktop\\result\\GE\\"
path = "C:\\Users\\Administrator\\Desktop\\result\\PHLIPHS\\PHLIPHS_point_txt"
save_path = "C:\\Users\\Administrator\\Desktop\\result\\PHLIPHS\\"

for file in os.listdir(path):
    file_path = os.path.join(path, file)
    with open(file_path, "r") as f:
        data = f.readlines()[0]
    with open(save_path + "point.txt", "a+") as ff:
        ff.write(data)
        pass