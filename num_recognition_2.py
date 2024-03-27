import os
import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import rotate
from skimage.color import rgb2gray
import glob
import tensorflow as tf
import requests, io
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage

# Функция для обработки и отображения изображения
def display_image(image_path):
    app = QApplication([])
    window = QWidget()

    print('TensorFlow version:', tf.__version__)
    teach_file = glob.glob('./Object_Detection/test/*')
    print(len(teach_file), teach_file)
    i = 0
    img_name1 = teach_file[i]
    path = img_name1
    image0 = cv2.imread(img_name1, 1)
    image_height, image_width, _ = image0.shape
    image = cv2.resize(image0, (1024, 1024))
    image = image.astype(np.float32)
    paths = './model_resnet.tflite'
    interpreter = tf.lite.Interpreter(model_path=paths)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    X_data1 = np.float32(image.reshape(1, 1024, 1024, 3))
    input_index = (interpreter.get_input_details()[0]['index'])
    interpreter.set_tensor(input_details[0]['index'], X_data1)
    interpreter.invoke()
    detection = interpreter.get_tensor(output_details[0]['index'])
    net_out_value2 = interpreter.get_tensor(output_details[1]['index'])
    net_out_value3 = interpreter.get_tensor(output_details[2]['index'])
    net_out_value4 = interpreter.get_tensor(output_details[3]['index'])
    img = image0
    razmer = img.shape
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Преобразует из одной цветовой модели в другую
    img3 = img[:, :, :]
    box_x = int(detection[0, 0, 0] * image_height)
    box_y = int(detection[0, 0, 1] * image_width)
    box_width = int(detection[0, 0, 2] * image_height)
    box_height = int(detection[0, 0, 3] * image_width)
    cv2.rectangle(img2, (box_y, box_x), (box_height, box_width), (230, 230, 21), thickness=5)
    image = image0[box_x:box_width, box_y:box_height, :]
    img2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    grayscale = rgb2gray(image)
    edges = canny(grayscale, sigma=3.0)
    out, angles, distances = hough_line(edges)
    h, theta, d = out, angles, distances
    angle_step = 0.5 * np.diff(theta).mean()
    d_step = 0.5 * np.diff(d).mean()
    bounds = [np.rad2deg(theta[0] - angle_step), np.rad2deg(theta[-1] + angle_step), d[-1] + d_step, d[0] - d_step]
    _, angles_peaks, _ = hough_line_peaks(out, angles, distances, num_peaks=20)
    angle = np.mean(np.rad2deg(angles_peaks))
    angle
    if 0 <= angle <= 90:
        rot_angle = angle - 90
    elif -45 <= angle < 0:
        rot_angle = angle - 90
    elif -90 <= angle < -45:
        rot_angle = 90 + angle
    if abs(rot_angle) > 20:
        rot_angle = 0
    print('угол наклона', rot_angle)
    rotated = rotate(image, rot_angle, resize=True) * 255
    rotated = rotated.astype(np.uint8)
    rotated1 = rotated[:, :, :]
    if rotated.shape[1] / rotated.shape[0] < 2:
        minus = np.abs(int(np.sin(np.radians(rot_angle)) * rotated.shape[0]))
        rotated1 = rotated[minus:-minus, :, :]
        print(minus)
    lab = cv2.cvtColor(rotated1, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Преобразование изображения в формат QPixmap
    height, width, _ = final.shape
    bytesPerLine = 3 * width
    qimg = QPixmap.fromImage(QImage(final.data, width, height, bytesPerLine, QImage.Format_RGB888))

    # Создание QLabel для отображения изображения
    label = QLabel()
    label.setPixmap(qimg)

    # Настройка главного окна
    window.setWindowTitle(os.path.basename(image_path))
    window.resize(width, height)
    layout = QVBoxLayout()
    layout.addWidget(label)
    window.setLayout(layout)

    # Отображение главного окна
    window.show()

    app.exec_()

if __name__ == "__main__":
    # Получение списка путей к изображениям
    image_paths = glob.glob('./Object_Detection/test/*')

    # Вывод всех изображений
    for path in image_paths:
        display_image(path)