import cv2
import numpy as np

min_w = 50
min_h = 50
line_coord = 340
offset = 8
carno = 0
vehicle_center_points = {}
vehicle_id = 0

def center(x, y, w, h):
    """计算边界框的中心点"""
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

cap = cv2.VideoCapture('./car_recognize/video/vehicle.mp4')  # 读取视频
bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()  # 创建背景减法器
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 创建形态学操作的核


# 获取视频的帧率和尺寸
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 初始化VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./car_recognize/video/output.mp4', fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()  # 读取每一帧
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    blur = cv2.GaussianBlur(frame_gray, (3, 3), 5)  # 高斯滤波去噪
    #cv2.imshow('blur', blur)
    fg_mask = bg_subtractor.apply(blur)  # 应用背景减法以获取前景掩码
    _, mask = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)  # 二值化掩码

    # 形态学操作：腐蚀和膨胀
    erode = cv2.erode(mask, kernel, iterations=1)
    dilate = cv2.dilate(erode, kernel, iterations=3)
    close = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow('close',close)
    # 在前景掩模中找到轮廓
    contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选轮廓
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < 2000:  # 过滤小轮廓
            continue
        x, y, w, h = cv2.boundingRect(contour)  # 获取外接矩形
        if w >= min_w and h >= min_h:  # 检查边界矩形是否满足最小尺寸要求
            boxes.append([x, y, x + w, y + h])  # 添加边界框坐标

    # 跟踪车辆
    new_center_points = {}
    for box in boxes:
        x, y, x2, y2 = box
        cx, cy = center(x, y, x2 - x, y2 - y)  # 计算边界框的中心点
        same_vehicle_detected = False
        for id, pts in vehicle_center_points.items():
            dist = np.sqrt((cx - pts[-1][0]) ** 2 + (cy - pts[-1][1]) ** 2)
            if dist < 50:  # 如果距离小于阈值，则认为是同一辆车
                new_center_points[id] = pts + [(cx, cy)]
                same_vehicle_detected = True
                break
        if not same_vehicle_detected:
            new_center_points[vehicle_id] = [(cx, cy)]
            vehicle_id += 1

    vehicle_center_points = new_center_points

    # 画出检测框和中心点
    for id, pts in vehicle_center_points.items():
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], (0, 255, 255), 2)  # 画出轨迹线
        x, y = pts[-1]
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # 画出中心点
        
        # 画出边界矩形
        for box in boxes:
            x1, y1, x2, y2 = box
            if abs(x - ((x1 + x2) // 2)) < 10 and abs(y - ((y1 + y2) // 2)) < 10:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                break

    # 计数逻辑
    for id, pts in list(vehicle_center_points.items()):
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                if (pts[i][1] <= line_coord < pts[i + 1][1]) or (pts[i][1] >= line_coord > pts[i + 1][1]):
                    carno += 1
                    vehicle_center_points.pop(id)
                    break

    # 显示计数和计数线
    cv2.putText(frame, "Car Count: " + str(carno), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.line(frame, (0, line_coord), (frame.shape[1], line_coord), (0, 255, 0), 2)

    #cv2.imshow('video', frame)

    # 将处理后的帧写入视频文件
    out.write(frame)
    key = cv2.waitKey(10)
    if key == 27:  # 按下ESC键退出
        break

print("Total car count:", carno)  # 输出总车数

cap.release()
out.release()  # 释放VideoWriter对象
cv2.destroyAllWindows()
