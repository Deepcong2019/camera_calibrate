#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:cong
@time: 2025/01/07 13:45:46
@theme:相机标定的目标是找到相机坐标系（由相机的位置和姿态决定）与世界坐标系（标定板的坐标系）之间的变换关系。标定板提供了一个可靠且统一的参考。
       确定相机的内参和外参；
"""
import cv2
import numpy as np
import glob

# 棋盘格参数 (内角点数量)
CHESSBOARD_SIZE = (10, 7)
SQUARE_SIZE = 20  # 每个格子的真实边长 (单位: 米)

# 为棋盘格的角点在三维空间中构建真实世界坐标系,objp 作为世界坐标系中的参考点，用于计算相机内参和畸变系数
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE  # 乘以每格边长，得到真实世界坐标

objpoints = []  # 3D点
imgpoints = []  # 2D点

# 读取保存的棋盘格图像
images = glob.glob('./calibration_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测角点
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    if ret:
        objpoints.append(objp)
        # cv2.cornerSubPix 是 OpenCV 中的一个函数，用于提高角点检测的精度。它通过亚像素级别的优化来精确定位角点位置。
        # 通常，在检测到角点后，使用该函数可以进一步优化角点的位置
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

        # 显示检测结果
        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# 标定相机,会输出和图片数量一样个数的R和T
(rms, mtx, dist, rvecs, tvecs) = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 打印标定结果
print("相机内参矩阵:\n", mtx)
print("畸变系数:\n", dist)
print("RMS error:", rms)
# print("旋转向量 (Rotation Vectors):\n", rvecs)
# print("平移向量 (Translation Vectors):\n", tvecs)


# RMS (cv2.calibrateCamera 的返回值): 是所有角点的均方根误差（平方和的均值再开平方）。
# total_error 的计算: 是所有图片的平均重投影误差，计算方式不同，但结果通常是相近的。
# 平均重投影误差通常应在 0.1 ~ 1 像素之间，越小越好。
# === 计算标定误差 (RMS) ===
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error

print("平均重投影误差:", total_error / len(objpoints))
# 保存标定参数
np.savez("calibration_data.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
