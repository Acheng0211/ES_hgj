# opencv 图像的基本运算
 
# 导入库
import numpy as np
import argparse
import cv2
 
# 构建参数解析器
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 
# 加载图像
image = cv2.imread(args["image"])
cv2.imshow("image loaded", image)
 
# 创建矩形区域，填充白色255
rectangle = np.zeros(image.shape[0:2], dtype="uint8")
cv2.rectangle(rectangle, (360, 348), (660, 570), 255, -1) # 修改这里
cv2.imshow("Rectangle", rectangle)
 
# 创建圆形区域，填充白色255
circle = np.zeros(image.shape[0:2], dtype="uint8")
cv2.circle(circle, (520, 455), 140, 255, -1) # 修改
cv2.imshow("Circle", circle)
 
''' 
# 在此例（二值图像）中，以下的0表示黑色像素值0, 1表示白色像素值255
# 位与运算，与常识相同，有0则为0, 均无0则为1
bitwiseAnd = cv2.bitwise_and(rectangle, circle)
cv2.imshow("AND", bitwiseAnd)
cv2.waitKey(0)
# 非运算，非0为1, 非1为0
bitwiseNot = cv2.bitwise_not(circle)
cv2.imshow("NOT", bitwiseNot)
cv2.waitKey(0)
# 异或运算，不同为1, 相同为0
bitwiseXor = cv2.bitwise_xor(rectangle, circle)
cv2.imshow("XOR", bitwiseXor)
cv2.waitKey(0)
'''
# 或运算，有1则为1, 全为0则为0
bitwiseOr = cv2.bitwise_or(rectangle, circle)
cv2.imshow("OR", bitwiseOr)
cv2.waitKey(0)
# 使用mask
mask = bitwiseOr
cv2.imshow("Mask", mask)
 
# Apply out mask -- notice how only the person in the image is cropped out
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Mask Applied to Image", masked)

key = cv2.waitKey(1) & 0xFF
if key == 27:
    cv2.destroyAllWindows()
