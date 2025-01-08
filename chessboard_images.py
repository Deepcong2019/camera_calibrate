#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:cong
@time: 2025/01/06 16:16:32
"""
import numpy as np
import cv2
import os


# 棋盘格参数 (内角点数量，行数-1, 列数-1))
CHESSBOARD_SIZE = (10, 7)
CAPTURE_PATH = './calibration_images'

# 创建保存目录
os.makedirs(CAPTURE_PATH, exist_ok=True)
cam_url = "rtsp://admin:jiean300845@172.16.51.93:554/h264/ch1/main/av_stream"
cam_url = 0
# 打开摄像头
cap = cv2.VideoCapture(cam_url)  # 如果有多个摄像头，可以调整索引 0
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("按 's' 保存图像，按 'q' 退出")
image_count = 0

while True:
    ret, frame = cap.read()
    frame_copy = frame.copy()  # 深拷贝图像
    if not ret:
        print("无法读取摄像头数据")
        break

    # 转灰度显示
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    # 如果检测到角点，绘制棋盘格角点
    if ret:
        cv2.drawChessboardCorners(frame_copy, CHESSBOARD_SIZE, corners, ret)

    # 显示摄像头画面
    cv2.imshow('Camera', frame_copy)

    # 按键控制
    key = cv2.waitKey(1)
    if key == ord('s'):  # 保存图像
        image_count += 1
        image_path = os.path.join(CAPTURE_PATH, f"chessboard_{image_count}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"保存图像: {image_path}")
    elif key == ord('q'):  # 退出
        break

cap.release()
cv2.destroyAllWindows()
