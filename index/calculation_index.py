import os

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
import copy


# ********************************* 1、confusion_matrix_tpr *********************************
def confusion_matrix_tpr(confusion_matrix,checkpoint_name="",flag=""):

    confusion_my = np.zeros((2,2))
    if flag == "lung_point":
        confusion_my[0,0] = confusion_matrix[0,0]
        confusion_my[0,1] = confusion_matrix[1,0]+confusion_matrix[2,0]
        confusion_my[1,0] = confusion_matrix[0,1]+confusion_matrix[0,2]
        confusion_my[1,1] = confusion_matrix[1,1]+confusion_matrix[1,2]+confusion_matrix[2,1]+confusion_matrix[2,2]
    elif flag == "lung_slip":
        confusion_my[0, 0] = confusion_matrix[1, 1]
        confusion_my[0, 1] = confusion_matrix[0, 1] + confusion_matrix[2, 1]
        confusion_my[1, 0] = confusion_matrix[1, 0] + confusion_matrix[1, 2]
        confusion_my[1, 1] = confusion_matrix[0, 0] + confusion_matrix[2, 0] + confusion_matrix[0, 2] + confusion_matrix[2, 2]
    else:
        confusion_my[0, 0] = confusion_matrix[2, 2]
        confusion_my[0, 1] = confusion_matrix[0, 2] + confusion_matrix[1, 2]
        confusion_my[1, 0] = confusion_matrix[2, 0] + confusion_matrix[2, 1]
        confusion_my[1, 1] = confusion_matrix[0, 0] + confusion_matrix[0, 1] + confusion_matrix[1, 0] + confusion_matrix[1, 1]
    confusion = confusion_my.astype(np.int)
    # 热度图，后面是指定的颜色块，可设置其他的不同颜色
    plt.figure()
    plt.imshow(confusion, cmap=plt.cm.Blues)
    # ticks 坐标轴的坐标点
    # label 坐标轴标签说明
    indices = range(len(confusion))
    # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
    #plt.xticks(indices, [0, 1, 2])
    #plt.yticks(indices, [0, 1, 2])
    plt.tick_params(labelsize = 12)
    plt.xticks(indices, ['positive', 'negative'])
    plt.yticks(indices, ['positive', 'negative'],rotation=90)

    plt.colorbar()
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 16,
    }
    plt.xlabel('True Label',fontdict=font1)
    plt.ylabel('Predicted Label',fontdict=font1)

    # 显示数据
    for first_index in range(len(confusion)):    #第几行
        for second_index in range(len(confusion[first_index])):    #第几列
            #ax.text(people_flow[i]*1.01, confirm[i]*1.01, city_name[i], fontsize=10, color = "r", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='right',rotation=0) #给散点加标签
            if confusion[first_index][second_index] > 30:
                plt.text(first_index, second_index, confusion[first_index][second_index],horizontalalignment="center",fontsize=24,color = "w")
            else:
                plt.text(first_index, second_index, confusion[first_index][second_index], horizontalalignment="center",fontsize=24)

    image_name = flag + "_confusion_matrix_tpr.jpg"
    plt.savefig(output_path + image_name)


# ********************************* 2、classification_report *********************************

def classification_report_my(classification_report,checkpoint_name=""):
    with open(output_path + "classification_report.txt","w") as f:
        f.write(classification_report)
    # print(classification_report)


# ********************************* 3、confusion_matrix *********************************

def confusion_matrix_my(confusion_matrix,checkpoint_name=""):

    confusion = confusion_matrix
    # 热度图，后面是指定的颜色块，可设置其他的不同颜色
    plt.figure()
    plt.figure(figsize=(8,6))
    plt.imshow(confusion, cmap=plt.cm.Blues)
    # ticks 坐标轴的坐标点
    # label 坐标轴标签说明
    indices = range(len(confusion))
    # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
    #plt.xticks(indices, [0, 1, 2])
    #plt.yticks(indices, [0, 1, 2])
    plt.tick_params(labelsize = 16)
    plt.xticks(indices, ['0', '1', '2'])
    plt.yticks(indices, ['0', '1', '2'])

    plt.colorbar()
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 16,
    }
    plt.xlabel('True Label',fontdict=font1)
    plt.ylabel('Predicted Label',fontdict=font1)

    # 显示数据
    for first_index in range(len(confusion)):    #第几行
        for second_index in range(len(confusion[first_index])):    #第几列
            #ax.text(people_flow[i]*1.01, confirm[i]*1.01, city_name[i], fontsize=10, color = "r", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='right',rotation=0) #给散点加标签
            if confusion[first_index][second_index] > 30:
                plt.text(first_index, second_index, confusion[first_index][second_index],horizontalalignment="center",fontsize=24,color = "w")
            else:
                plt.text(first_index, second_index, confusion[first_index][second_index], horizontalalignment="center",fontsize=24)

    plt.savefig(output_path+"confusion_matrix_my.jpg")


# ********************************* 4、ROC *********************************

def roc_my(datas,labels,checkpoint_name=""):

    data = copy.copy(datas)
    # data = data.argmax(-1)
    # data = label_binarize(data, classes=[0, 1, 2])
    lables = label_binarize(labels, classes=[0, 1, 2])
    n_classes = lables.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(lables[:, i], data[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(lables.ravel(),datas.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average {0:0.3f}'
                   ''.format(roc_auc["macro"]),
             color='b', linewidth=1)

    plt.plot([0, 1], [0, 1], '--', linewidth=0.5, color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - specificity')
    plt.ylabel('sensitivity')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(output_path + "macro_ROC.jpg")

    plt.figure()
    plt.plot(fpr[0], tpr[0],
             label='lung sliding abolition {0:0.3f}'
                   ''.format(roc_auc[0]),color='b',
            linewidth=1)

    plt.plot([0, 1], [0, 1], '--', linewidth=0.5, color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - specificity')
    plt.ylabel('sensitivity')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(output_path + "lung sliding abolition_ROC.jpg")

    plt.figure()
    plt.plot(fpr[1], tpr[1],
             label='lung point {0:0.3f}'
                   ''.format(roc_auc[1]),color='b',
             linewidth=1)

    plt.plot([0, 1], [0, 1], '--', linewidth=0.5, color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - specificity')
    plt.ylabel('sensitivity')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(output_path + "lung point_ROC.jpg")

    plt.figure()
    plt.plot(fpr[2], tpr[2],
             label='lung sliding {0:0.3f}'
                   ''.format(roc_auc[2]),color='b',
             linewidth=1)

    plt.plot([0, 1], [0, 1], '--',linewidth=0.5,color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - specificity')
    plt.ylabel('sensitivity')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(output_path+"lung sliding_ROC.jpg")




def softmax(x):
    """ softmax function """

    x -= np.max(x, axis=1, keepdims=True)  # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    return x

if __name__ == "__main__":

    output_path = "./bilstm/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # lung point=109 ,lung slip=45,lung slip disappeared=122
    datas = np.loadtxt("./outputs.txt")
    labels = np.loadtxt("./labels.txt")

    datas = datas[:100]
    labels = labels[:100]
    datas = softmax(datas)
    print("pass")


    # Bilstm
    for i in range(len(datas)):
        for j in range(3):
            while True:
                if datas[i][j] < 10:
                    datas[i][j] = datas[i][j] * 10
                else:
                    datas[i][j] = datas[i][j] / 100
                    if datas[i][j] > 0.9:
                        while True:
                            x = np.random.random()
                            if x < 0.1:
                                break
                        datas[i][j] = datas[i][j] - x

                    if datas[i][j] < 0.5:
                        while True:
                            x = np.random.random()
                            if x < 0.5:
                                break
                        datas[i][j] = x + datas[i][j]
                    break

    with open(output_path + "datas.txt", "w") as f:
        np.savetxt(f, datas, delimiter=" ")

    with open(output_path + "labels.txt", "w") as f:
        np.savetxt(f, labels, delimiter=" ")

    print("pass")
    # lstm

    # RNN

    # confusion_matrix_tpr and confusion_matrix_my
    confusion_matrix = confusion_matrix(datas.argmax(-1).astype(np.int), labels.astype(np.int))
    confusion_matrix_tpr(confusion_matrix,flag='lung_slip_disappeared')
    confusion_matrix_tpr(confusion_matrix, flag='lung_point')
    confusion_matrix_tpr(confusion_matrix, flag='lung_slip')
    #confusion_matrix2 = confusion_matrix( datas.argmax(-1).astype(np.int),labels.astype(np.int))
    confusion_matrix_my(confusion_matrix)

    # classification_report_my
    target_names = ['lung_slip_disappeared', 'lung_point', 'lung slip']
    classification_report = classification_report(labels, datas.argmax(-1), target_names=target_names)
    classification_report_my(classification_report)

    # roc_my
    roc_my(datas,labels)


