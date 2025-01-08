#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:cong
@time: 2025/01/08 09:11:24
"""
import numpy as np
import glob
import cv2
import math

#当前验证此算法的标定结果与其他标定基本一致

#1，相机标定获取内参及畸变系数
#角点个数
w = 10
h = 7
b_w = 20  #棋盘格边长20mm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((w * h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
objp = b_w * objp  # 打印棋盘格一格的边长为2.6cm
#print(objp)
obj_points = []  # 存储3D点
img_points = []  # 存储2D点

#images = glob.glob("E:/image/*.png")  # 黑白棋盘的图片路径

def get_image_paths(folder_path):
    # 使用通配符筛选出所有jpg/png图片
    return glob.glob(f"{folder_path}/*.jpg", recursive=True)
    # 如果需要包括其他格式的图片，可以在这里添加，例如：png
    # return glob.glob(f"{folder_path}/**/*.jpg", recursive=True) + \
    #        glob.glob(f"{folder_path}/**/*.png", recursive=True)


# 使用示例
#folder_path = "G:/3dversion/weiziguji/8mm/"  # 替换为你的文件夹路径
folder_path = "./calibration_images/"
images = get_image_paths(folder_path)

size = None
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    if ret:
        obj_points.append(objp)     #世界坐标系中的三维点始终不变
        #此处的winsize（会影响到畸变系数）
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001))
        #寻找棋盘格角点，若是有则进行保存
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)
        cv2.drawChessboardCorners(img, (w, h), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
        cv2.imshow("demo",img)
        cv2.waitKey(500)

# print(obj_points)
# print(img_points)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
result = "摄像机矩阵:\n {}\n 畸变参数:\n {}\n 旋转矩阵:\n {}\n 平移矩阵:\n {}".format(mtx, dist, rvecs, tvecs)
print(result)

# 内参数矩阵、畸变系数
Camera_intrinsic = {"mtx": mtx, "dist": dist, }

#2，获取当前位姿（原点位姿）
obj_points = objp  # 存储3D点
img_points = []  # 存储2D点

for fname in images:
    #_, frame = camera.read()
    frame = cv2.imread(fname)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    if ret:  # 画面中有棋盘格
        img_points = np.array(corners)
        cv2.drawChessboardCorners(frame, (w, h), corners, ret)
        # rvec: 旋转向量 tvec: 平移向量
        _, rvec, tvec = cv2.solvePnP(obj_points, img_points, Camera_intrinsic["mtx"], Camera_intrinsic["dist"])  # 解算位姿
        # print(rvec)
        #print(tvec)
        distance = math.sqrt(tvec[0][0] ** 2 + tvec[1][0] ** 2 + tvec[2][0] ** 2)  # 计算距离，距离相机的距离
        #将旋转向量转换成欧拉角（绕x轴转动pitch，绕y轴转动yaw，绕z轴转动roll）
        rvec_matrix = cv2.Rodrigues(rvec)[0]  # 旋转向量->旋转矩阵
        proj_matrix = np.hstack((rvec_matrix, tvec))  # hstack: 水平合并
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]  # 欧拉角
        #print(eulerAngles)
        pitch, yaw, roll = eulerAngles[0][0], eulerAngles[1][0], eulerAngles[2][0]
        cv2.putText(frame, "dist: %.2fmm, yaw: %.2f, pitch: %.2f, roll: %.2f" % (distance, yaw, pitch, roll), (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # 按ESC键退出
            break
    else:  # 画面中没有棋盘格
        cv2.putText(frame, "Unable to Detect Chessboard", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (0, 0, 255), 3)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # 按ESC键退出
            break


#3，画坐标轴和立方体
len = b_w
def draw(img, corners, imgpts, imgpts2):
    #line必须转为int型才能绘制
    corners = np.int32(corners)
    imgpts2 = np.int32(imgpts2)
    corner = tuple(corners[0].ravel())  # ravel()方法将数组维度拉成一维数组
    # img要画的图像，corner起点，tuple终点，颜色，粗细
    img = cv2.line(img, corner, tuple(imgpts2[0].ravel()), (255, 0, 0), 8)
    img = cv2.line(img, corner, tuple(imgpts2[1].ravel()), (0, 255, 0), 8)
    img = cv2.line(img, corner, tuple(imgpts2[2].ravel()), (0, 0, 255), 8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'X', tuple(imgpts2[0].ravel() + 2), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, 'Y', tuple(imgpts2[1].ravel() + 2), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, 'Z', tuple(imgpts2[2].ravel() + 2), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    imgpts = np.int32(imgpts).reshape(-1, 2)  # draw ground floor in green
    for i, j in zip(range(4), range(4, 8)):  # 正方体顶点逐个连接

        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 215, 0), 3)  # draw top layer in red color
    # imgpts[4:]是八个顶点中上面四个顶点
    # imgpts[:4]是八个顶点中下面四个顶点
    # 用函数drawContours画出上下两个盖子，它的第一个参数是原始图像，第二个参数是轮廓，一个python列表，第三个参数是轮廓的索引（当设置为-1时绘制所有轮廓）
    img = cv2.drawContours(img, [imgpts[4:]], -1, (255, 215, 0), 3)
    img = cv2.drawContours(img, [imgpts[:4]], -1, (255, 215, 0), 3)

    return img


objp = np.zeros((w * h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w * len:len, 0:h * len:len].T.reshape(-1, 2)
axis = np.float32([[0, 0, 0], [0, 2 * len, 0], [2 * len, 2 * len, 0], [2 * len, 0, 0],
                   [0, 0, -2 * len], [0, 2 * len, -2 * len], [2 * len, 2 * len, -2 * len], [2 * len, 0, -2 * len]])
axis2 = np.float32([[3 * len, 0, 0], [0, 3 * len, 0], [0, 0, -3 * len]]).reshape(-1, 3)
# images = glob.glob('*.jpg')
i = 1
for fname in images:
    img = cv2.imread(fname)
    # cv2.imshow('世界坐标系与小盒子', img)
    # cv2.waitKey(0)
    #img = cv2.resize(img, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 找到棋盘格角点
    # 寻找角点，存入corners，ret是找到角点的flag
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    if ret is True:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        #print(corners2)
        # 求解物体位姿的需要
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
        # projectPoints()根据所给的3D坐标和已知的几何变换来求解投影后的2D坐标。
        # imgpts是整体的8个顶点
        imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        # imgpts2是三个坐标轴的x,y,z划线终点
        imgpts2, _ = cv2.projectPoints(axis2, rvecs, tvecs, mtx, dist)
        #绘制方格
        img = draw(img, corners2, imgpts, imgpts2)

        #绘制x、y、z
        distance = math.sqrt(tvec[0][0] ** 2 + tvec[1][0] ** 2 + tvec[2][0] ** 2)  # 计算距离
        # print(distance)
        rvec_matrix = cv2.Rodrigues(rvec)[0]  # 旋转向量->旋转矩阵
        proj_matrix = np.hstack((rvec_matrix, tvec))  # hstack: 水平合并
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]  # 欧拉角
        # print(eulerAngles)
        pitch, yaw, roll = eulerAngles[0][0], eulerAngles[1][0], eulerAngles[2][0]
        p0 = tuple(corners[0].ravel())
        cv2.putText(img, "x: %.2f, y: %.2f, dist: %.2fmm, yaw: %.2f, pitch: %.2f, roll: %.2f" % (p0[0],p0[1],distance, yaw, pitch, roll),
                    (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('世界坐标系与小盒子', img)
        #cv2.imwrite(str(i) + '.png', img)
        cv2.waitKey(0)
        i += 1

cv2.destroyAllWindows()
print("完毕")
