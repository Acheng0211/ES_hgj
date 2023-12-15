import os
import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image

def take_photo(color_frame, photo_path):
    color_image = np.asanyarray(color_frame.get_data())
    cv2.imwrite(photo_path, color_image)
    print(f"照片已保存到: {photo_path}")

def getMask(raw_photo_path):
    image = Image.open(raw_photo_path)
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    width, height = image.size
    instance_mask = Image.new('L', (width, height), 0)

    for x in range(width):
        for y in range(height):
            pixel = image.getpixel((x, y))
            if pixel[3] != 0:  # 判断像素的 alpha 值是否不为 0
                instance_mask.putpixel((x, y), 255)  # 将掩码像素设置为白色（255）

    # # 加载原始照片
    # raw_photo = cv2.imread(raw_photo_path)
    # # 创建实例分割器
    # instance_segmenter = cv2.dnn_SegmentationModel('deeplabv3_mnv2_pascal_train_aug_2018_01_29.pbtxt',
    #                                                'deeplabv3_mnv2_pascal_train_aug_2018_01_29/frozen_inference_graph.pb')
    # # 设置输入
    # instance_segmenter.setInput(cv2.dnn.blobFromImage(raw_photo))
    # # 获取输出
    # instance_mask = instance_segmenter.forward()
    # # 获取实例分割结果
    # instance_mask = instance_mask[0, :, :, 1]
    # 保存实例分割结果
    
    return instance_mask

def getGeoFeature(mask_photo_path):
    
    # 加载实例分割结果
    instance_mask = getMask(mask_photo_path)
    #instance_mask = cv2.imread('instance_mask.png', cv2.IMREAD_GRAYSCALE)
    # 寻找实例轮廓
    contours, hierarchy = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 提取物体的几何特征
    for i, contour in enumerate(contours):
        # 计算面积
        area = cv2.contourArea(contour)
        # 计算边界框
        x, y, w, h = cv2.boundingRect(contour)
        # 打印或保存特征信息
        print(f"Object {i + 1}: Area = {area}, Bounding Box = ({x}, {y}, {w}, {h})")

def main():
    # 创建保存照片的文件夹
    photo_dir = "/home/hgj/ES_ws/photo/raw/"
    if not os.path.exists(photo_dir):
        os.makedirs(photo_dir)

    # 配置相机
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 启动相机
    profile = pipeline.start(config)

    try:
        photo_count = 0
        while True:
            # 等待帧
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # 将帧转换为图像
            color_image = np.asanyarray(color_frame.get_data())
            cv2.imshow('RealSense D435i 实时图像', color_image)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):# 按下空格键拍照
                photo_path = os.path.join(photo_dir, f"RawPhoto_{photo_count}.png")
                take_photo(color_frame, photo_path)
                photo_count += 1
            elif key == 27:  # 按下ESC键退出
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    mask = getMask(photo_dir+"RawPhoto_0.png")
    mask_np = np.array(mask)
    mask_photo_dir = "/home/hgj/ES_ws/photo/mask/"
    # mask.save(mask_photo_dir+"instance_mask.png")
    if not os.path.exists(mask_photo_dir):
        os.makedirs(mask_photo_dir)
    mask_photo_path = os.path.join(mask_photo_dir, "instance_mask.png")
    print(photo_dir+"RawPhoto_0.png")
    print(mask_photo_path)
    cv2.imwrite(mask_photo_path, mask_np)
    # getGeoFeature(photo_path+"RawPhoto_0.png")

if __name__ == '__main__':
    main()
    
