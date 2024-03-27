import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
import cv2
import numpy as np
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import rotate
from skimage.color import rgb2gray
import itertools
import tensorflow as tf


def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(letters):
                outstr += letters[c]
        ret.append(outstr)
    return ret

letters=['0' ,'1' ,'2','3' ,'4' ,'5' ,'6' , '7' ,'8' ,'9' ,'A' ,'B' ,'C' ,'E' ,'H' ,'K' ,'M' ,'O' ,'P' ,'T' ,'X' ,'Y' ]

def process_image(image_path):
    image0 = cv2.imread(image_path, 1)
    image_height, image_width, _ = image0.shape
    image = cv2.resize(image0, (1024, 1024))
    image = image.astype(np.float32)

    paths = 'model_resnet.tflite'
    interpreter = tf.lite.Interpreter(model_path=paths)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    X_data1 = np.float32(image.reshape(1, 1024, 1024, 3))
    input_index = interpreter.get_input_details()[0]['index']
    interpreter.set_tensor(input_details[0]['index'], X_data1)
    interpreter.invoke()
    detection = interpreter.get_tensor(output_details[0]['index'])
    net_out_value2 = interpreter.get_tensor(output_details[1]['index'])
    net_out_value3 = interpreter.get_tensor(output_details[2]['index'])
    net_out_value4 = interpreter.get_tensor(output_details[3]['index'])
    img = image0
    razmer = img.shape

    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img3 = img[:, :, :]

    box_x = int(detection[0, 0, 0] * image_height)
    box_y = int(detection[0, 0, 1] * image_width)
    box_width = int(detection[0, 0, 2] * image_height)
    box_height = int(detection[0, 0, 3] * image_width)
    if np.min(detection[0, 0, :]) >= 0:
        cv2.rectangle(img2, (box_y, box_x), (box_height, box_width), (255, 0, 0), thickness=2)
        image = img3[box_x:box_width, box_y:box_height, :]
        grayscale = rgb2gray(image)
        edges = canny(grayscale, sigma=3.0)
        out, angles, distances = hough_line(edges)
        _, angles_peaks, _ = hough_line_peaks(out, angles, distances, num_peaks=20)
        angle = np.mean(np.rad2deg(angles_peaks))
        if 0 <= angle <= 90:
            rot_angle = angle - 90
        elif -45 <= angle < 0:
            rot_angle = angle - 90
        elif -90 <= angle < -45:
            rot_angle = 90 + angle
        if abs(rot_angle) > 20:
            rot_angle = 0
        rotated = rotate(image, rot_angle, resize=True) * 255
        rotated = rotated.astype(np.uint8)
        rotated1 = rotated[:, :, :]
        minus = np.abs(int(np.sin(np.radians(rot_angle)) * rotated.shape[0]))
        if rotated.shape[1] / rotated.shape[0] < 2 and minus > 10:
            rotated1 = rotated[minus:-minus, :, :]
        lab = cv2.cvtColor(rotated1, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        paths = 'numberplate_model.tflite'
        interpreter = tf.lite.Interpreter(model_path=paths)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        img = final  # лучше работает при плохом освещении
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128, 64))
        img = img.astype(np.float32)
        img /= 255

        img1 = img.T
        X_data1 = np.float32(img1.reshape(1, 128, 64, 1))
        input_index = interpreter.get_input_details()[0]['index']
        interpreter.set_tensor(input_details[0]['index'], X_data1)

        interpreter.invoke()
        net_out_value = interpreter.get_tensor(output_details[0]['index'])
        pred_texts = decode_batch(net_out_value)
        return pred_texts, img2, rotated1
    else:
        return ['нет'], None, None

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Распознавание номерных знаков")
        self.geometry("1024x768")

        self.label = ctk.CTkLabel(self, text="Выберите изображение для обработки")
        self.label.pack(pady=10)

        self.button = ctk.CTkButton(self, text="Выбрать изображение", command=self.open_file)
        self.button.pack(pady=10)

        self.result_label = ctk.CTkLabel(self, text="")
        self.result_label.pack(pady=10)

        self.canvas = ctk.CTkCanvas(self, width=600, height=400)
        self.canvas.pack()

        self.plate_canvas = ctk.CTkCanvas(self, width=400, height=100, bg="white")
        self.plate_canvas.pack(pady=10)

        self.image_list = []
        self.plate_list = []

    def clear_main_canvas(self):
        self.canvas.delete("all")
        self.image_list = []

    def clear_plate_canvas(self):
        self.plate_canvas.delete("all")
        self.plate_list = []

    def open_file(self):
        file_path = filedialog.askopenfilename(title="Выберите изображение", filetypes=[("Image files", "*.jpg;*.png;*.bmp")])
        if file_path:
            pred_texts, img2, rotated1 = process_image(file_path)
            self.result_label.configure(text=f"Распознанный номер: {', '.join(pred_texts)}")
            if img2 is not None:
                img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
                img2 = cv2.resize(img2, (600, 400))
                photo = tk.PhotoImage(data=cv2.imencode('.png', img2)[1].tobytes())
                self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.canvas.photo = photo
                self.image_list.append(photo)

                plate_img = rotated1
                plate_img = cv2.cvtColor(plate_img, cv2.COLOR_RGB2BGR)
                plate_img = cv2.resize(plate_img, (400, 100))
                plate_photo = tk.PhotoImage(data=cv2.imencode('.png', plate_img)[1].tobytes())
                self.plate_canvas.create_image(0, 0, anchor=tk.NW, image=plate_photo)
                self.plate_canvas.photo = plate_photo
                self.plate_list.append(plate_photo)

if __name__ == "__main__":
    app = App()
    app.mainloop()