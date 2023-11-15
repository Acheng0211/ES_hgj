import glob

import numpy as np
import cv2
from agnostic_segmentation import agnostic_segmentation

if __name__ == "__main__":
    oj_img_ori = "oj1.jpeg"
    oj_img_path = "/home/hgj/ES_ws/photo/"+oj_img_ori
   
    oj_prediction = agnostic_segmentation.segment_image(oj_img_ori, oj_img_path)
    seg_img = agnostic_segmentation.draw_segmented_image(oj_img_ori, oj_img_path)
    msk_img = agnostic_segmentation.draw_found_masks(oj_img_ori, oj_prediction)
    img = cv2.imread(oj_img_path, flags=True)
    cv2.imshow("demo",img)
    cv2.waitKey(5000)
    #cv2.imwrite("demo.jpg",img)
    print("success")