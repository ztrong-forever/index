import numpy as np
import os


def save_std(path, dirs_list, *args):

    path = path
    precision_dict, recall_dict, f1_score_dict = args

    for key, value in precision_dict.items():
        if not os.path.exists(key):
            os.mkdir(key)
        dir_path = key + "\\std.txt"
        std = np.var(value[:3])
        mean = value[-1]
        with open(dir_path, "a") as f:
            f.write("precision_mean:" + str(mean) + "\n")
            f.write("precision_std:" + str(std) + "\n")

    for key, value in recall_dict.items():
        if not os.path.exists(key):
            os.mkdir(key)
        dir_path = key + "\\std.txt"
        std = np.var(value[:3])
        mean = value[-1]
        with open(dir_path, "a") as f:
            f.write("recall_mean:" + str(mean) + "\n")
            f.write("recall_std:" + str(std)+ "\n")

    for key, value in f1_score_dict.items():
        if not os.path.exists(key):
            os.mkdir(key)
        dir_path = key + "\\std.txt"
        std = np.var(value[0:3])
        mean = value[-1]
        with open(dir_path, "a") as f:
            f.write("f1_score_mean:" + str(mean) + "\n")
            f.write("f1_score_std:" + str(std)+ "\n")

    for dirs in dirs_list:
        if not os.path.exists(key):
            os.mkdir(key)
        dir_list1 = dirs.split("\\")
        dir_list2 = "\\".join(dir_list1[:3]) + "\\std.txt"
        with open(dir_list2, "r") as f:
            file = f.read()
        with open(path + "all_index.txt", "a") as f:
            f.write("*************************    " + dir_list1[2] + "    *************************" + "\n" +
                    file + "\n" + "\n" + "\n")


def load_file(loadfile_path):

    dirs_list = []
    loadfile_path = loadfile_path

    for root, dirs, files in os.walk(loadfile_path):

        if files:
            dirs_list.append(os.path.join(root, files[0]))

    return dirs_list


def read_file(dirs_path):

    precision_dict = {}
    recall_dict = {}
    f1_score_dict = {}

    for dir_path in dirs_path:

        file_name = dir_path.split("\\")[2]

        with open(dir_path, "r") as f:

            files = f.read()
            precision_list = []
            recall_list = []
            f1_score_list = []
            files = files.replace("\n", '').strip().split(" ")
            files = [file for file in files if file]

            for index, file in enumerate(files):

                if file == "point":
                    precision = float(files[index+1])
                    precision_list.append(precision)
                    recall = float(files[index+2])
                    recall_list.append(recall)
                    f1_score = float(files[index+3])
                    f1_score_list.append(f1_score)

                elif file == "slip" and files[index+1] not in "disappeared":
                    precision = float(files[index + 1])
                    precision_list.append(precision)
                    recall = float(files[index + 2])
                    recall_list.append(recall)
                    f1_score = float(files[index + 3])
                    f1_score_list.append(f1_score)

                elif file == "disappeared":
                    precision = float(files[index + 1])
                    precision_list.append(precision)
                    recall = float(files[index + 2])
                    recall_list.append(recall)
                    f1_score = float(files[index + 3])
                    f1_score_list.append(f1_score)

                elif file == "macro":
                    precision_mean = float(files[index + 2])
                    precision_list.append(precision_mean)
                    recall_mean = float(files[index + 3])
                    recall_list.append(recall_mean)
                    f1_score_mean = float(files[index + 4])
                    f1_score_list.append(f1_score_mean)

            precision_dict[file_name] = precision_list
            recall_dict[file_name] = recall_list
            f1_score_dict[file_name] = f1_score_list

    # print(precision_dict, "\n", recall_dict, "\n", f1_score_dict)
    return precision_dict, recall_dict, f1_score_dict


def main():

    path = ".\\file\\"
    dirs_list = load_file(path)
    precision_dict, recall_dict, f1_score_dict = read_file(dirs_list)
    save_std(path, dirs_list, precision_dict, recall_dict, f1_score_dict)
    # print(dirs_list)




if __name__ == "__main__":

    main()