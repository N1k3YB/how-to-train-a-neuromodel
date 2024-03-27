import tensorflow as tf
import os
from os.path import join
import json
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.models import Model, load_model
from keras.layers import LSTM,Bidirectional
from keras.optimizers import Adam
from keras.saving import register_keras_serializable
from keras.callbacks import ProgbarLogger
import cv2
import glob
from collections import Counter

def get_counter(dirpath):
    dirname = os.path.basename(dirpath)
    ann_dirpath = join(dirpath, r'D:\dataset1\OCR\train\ann')
    letters = ''
    lens = []
    for filename in os.listdir(ann_dirpath):
        json_filepath = join(ann_dirpath, filename)
        description = json.load(open(json_filepath, 'r'))['description']
        lens.append(len(description))
        letters += description
    print('Максимальная длина номерного знака в "%s":' % dirname, max(Counter(lens).keys()))
    return Counter(letters)

c_val = get_counter(r'D:\dataset1\OCR\test')
c_train = get_counter(r'D:\dataset1\OCR\train')
letters_train = set(c_train.keys())
letters_val = set(c_val.keys())
if letters_train == letters_val:
    print('Символы в train и val совпадают')
else:
    raise Exception()
letters = sorted(list(letters_train))
print('Символы:', ' '.join(letters))

def labels_to_text(labels):
    labels1=[]
    for i in labels:
        if i < len(letters):
            labels1.append(i)
        
    return ''.join(list(map(lambda x: letters[int(x)], labels1)))

def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))

def is_valid_str(s):
    for ch in s:
        if not ch in letters:
            print('ddddddddddddd')
            return False
    return True

class TextImageGenerator:
    
    def __init__(self, 
                 dirpath, 
                 img_w, img_h, 
                 batch_size, 
                 downsample_factor,
                 max_text_len=9):
        
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor
        
        img_dirpath = join(dirpath, r'D:\dataset1\OCR\test\img')
        ann_dirpath = join(dirpath, r'D:\dataset1\OCR\test\ann')
        self.samples = []
        for filename in os.listdir(img_dirpath):
            name, ext = os.path.splitext(filename)
            if ext == '.png':
                img_filepath = join(img_dirpath, filename)
                json_filepath = join(ann_dirpath, name + '.json')
                description = json.load(open(json_filepath, 'r'))['description']
                if is_valid_str(description) and len(description)>0 and len(description)<10:
                    self.samples.append([img_filepath, description])
        
        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.cur_index = 0
        
    def build_data(self):
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []
        for i, (img_filepath, text) in enumerate(self.samples):
            img = cv2.imread(img_filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.img_w, self.img_h))
            img = img.astype(np.float32)
            img /= 255
            self.imgs[i, :, :] = img
            self.texts.append(text)
        
    def get_output_size(self):
        return len(letters) + 1
    
    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]
    
    def next_batch(self):
        while True:
            if K.image_data_format() == 'channels_first':
                X_data = np.ones([self.batch_size, 1, self.img_w, self.img_h])
            else:
                X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])
            Y_data = np.ones([self.batch_size, self.max_text_len])*22
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.zeros((self.batch_size, 1))
            source_str = []
                                   
            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                if K.image_data_format() == 'channels_first':
                    img = np.expand_dims(img, 0)
                else:
                    img = np.expand_dims(img, -1)
                X_data[i] = img
                aaaa=text_to_labels(text)
                laaaa=len(aaaa)
                Y_data[i,:laaaa] = aaaa
                source_str.append(text)
                label_length[i] = len(text)
                
            inputs = {
                'the_input': X_data,
                'the_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length,
            }
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs, outputs)

@register_keras_serializable()
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    input_length = tf.cast(tf.squeeze(input_length, axis=-1), dtype=tf.int32)  # Cast input_length to int32
    label_length = tf.cast(tf.squeeze(label_length, axis=-1), dtype=tf.int32)

    labels = K.ctc_label_dense_to_sparse(tf.cast(labels, dtype=tf.int32), label_length)

    loss = tf.nn.ctc_loss(
        labels=labels,
        logits=y_pred,
        label_length=label_length,
        logit_length=input_length,
        blank_index=-1,
        logits_time_major=False
    )
    return tf.reduce_mean(loss)

def ctc_loss_function(y_true, y_pred):
    return y_pred



def train(img_w, load=False):
    img_h = 64
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)
        
    batch_size = 32
    downsample_factor = pool_size ** 2
    tiger_train = TextImageGenerator(r'DD:\dataset1\OCR\train', img_w, img_h, batch_size, downsample_factor)
    tiger_train.build_data()
    tiger_val = TextImageGenerator(r'D:\dataset1\OCR\test', img_w, img_h, batch_size, downsample_factor)
    tiger_val.build_data()

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)
    gru_1 = LSTM(rnn_size, return_sequences=True,  name='gru1')
    gru_1b = LSTM(rnn_size, return_sequences=True, go_backwards=True, name='gru1_b')
    gru1_merged = Bidirectional(gru_1, backward_layer=gru_1b )(inner)
    gru_2 = LSTM(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')
    gru_2b = LSTM(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')
    gru2_merged = Bidirectional(gru_2, backward_layer=gru_2b )(gru1_merged)
    inner = Dense(tiger_train.get_output_size(), kernel_initializer='he_normal',
                  name='dense2')(gru2_merged)
    y_pred = Activation('softmax', name='softmax')(inner)
    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[tiger_train.max_text_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    sgd=Adam()
    if load:
        model = load_model('numberplate_model.keras', compile=False)
    else:
        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    model.compile(loss={'ctc': ctc_loss_function}, optimizer=sgd)
    if not load:
        def data_generator(tiger):
            while True:
                for inp_value, _ in tiger.next_batch():
                    X_data = inp_value['the_input']
                    Y_data = inp_value['the_labels']
                    input_length = inp_value['input_length']
                    label_length = inp_value['label_length']
                    yield X_data, Y_data, input_length, label_length
                callbacks = [ProgbarLogger()]
                model.fit(
                    data_generator(tiger_train),
                    steps_per_epoch=tiger_train.n,
                    epochs=1,
                    validation_data=data_generator(tiger_val),
                    validation_steps=tiger_val.n,
                    callbacks=callbacks
                )

        return model
    
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



model = train(128, load=False)
model.save('numberplate_model.keras')

paths='numberplate_model.tflite'
interpreter = tf.lite.Interpreter(model_path=paths)
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Путь к папке с изображениями
image_dir = r'D:\dataset1\OCR\test\img'

# Получаем список всех файлов с расширением .jpg, .png или .bmp в указанной папке
image_files = glob.glob(image_dir + '/*.*')

# Цикл по всем файлам
for file in image_files:
    # Читаем изображение
    img = cv2.imread(file)
    
    # Обработка изображения здесь
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 64))
    img = img.astype(np.float32)
    img /= 255

    # # width and height are backwards from typical Keras convention
    # because width is the time dimension when it gets fed into the RNN
    img1=img.T
    img1.shape
    X_data1=np.float32(img1.reshape(1,128, 64,1))
    X_data1.shape
    input_index = (interpreter.get_input_details()[0]['index'])
    interpreter.set_tensor(input_details[0]['index'], X_data1)

    interpreter.invoke()

    net_out_value = interpreter.get_tensor(output_details[0]['index'])
    pred_texts = decode_batch(net_out_value)
    pred_texts
    fig = plt.figure(figsize=(10, 10))
    outer = gridspec.GridSpec(2, 1, wspace=10, hspace=0.1)
    ax1 = plt.Subplot(fig, outer[0])
    fig.add_subplot(ax1)
    ax2 = plt.Subplot(fig, outer[1])
    fig.add_subplot(ax2)
    print('Predicted:', pred_texts[0])
    img = X_data1[0,:, :, 0].T
    ax1.set_title('Input img')
    ax1.imshow(img, cmap='gray')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_title('Acrtivations')
    ax2.imshow(net_out_value[0].T, cmap='binary', interpolation='nearest')
    ax2.set_yticks(list(range(len(letters) + 1)))
    ax2.set_yticklabels(letters + ['blank'])
    ax2.grid(False)
    for h in np.arange(-0.5, len(letters) + 1 + 0.5, 1):
        ax2.axhline(h, linestyle='-', color='k', alpha=0.5, linewidth=1)

    #ax.axvline(x, linestyle='--', color='k')
    plt.show()




