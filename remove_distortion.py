#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:cong
@time: 2025/01/07 14:03:03
"""

import numpy as np
import cv2


# 加载标定参数
data = np.load("calibration_data.npz")
mtx = data['mtx']
dist = data['dist']
cam_url = "rtsp://admin:jiean300845@172.16.51.93:554/h264/ch1/main/av_stream"
cam_url=0
# 校正摄像头实时图像
cap = cv2.VideoCapture(cam_url)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头数据")
        break

    # 矫正畸变
    h, w = frame.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, new_mtx)

    # 显示结果
    cv2.imshow("Original", frame)
    cv2.imshow("Undistorted", undistorted_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
