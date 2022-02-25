import cv2
import os

# mask_path = "C:\\Users\\Administrator\\Desktop\\result\\PHLIPHS\\PHLIPHS_mask_save\\"
# image_path = "C:\\Users\\Administrator\\Desktop\\result\\PHLIPHS\\PHLIPHS_save\\"

image_path = "C:\\Users\\Administrator\\Desktop\\result\\GE\\GE_save\\"
mask_path = "C:\\Users\\Administrator\\Desktop\\result\\GE\\GE_mask_save\\"


for file in os.listdir(mask_path):
    mask = os.path.join(mask_path, file)
    image = os.path.join(image_path, file)
    mask = cv2.imread(mask,0)
    image = cv2.imread(image)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = 0
    for contour in range(0,len(contours)):
        if len(contours[contour]) > len(contours[cnt]):
            cnt = contour
    cv2.drawContours(image, [contours[cnt]], 0, (255, 255, 255), 1)
    # 打开画了轮廓之后的图像
    cv2.imshow('mask', image)
    cv2.waitKey()
    #cv2.destroyWindow()
