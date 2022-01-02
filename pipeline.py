import numpy as np
import pandas as pd
from numpy import asarray
import pickle

#pip install opencv-python
import cv2 
import PIL
from PIL import Image

#pip install cmake
#pip install dlib
import dlib
from skimage import io

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

def df_detect_2(vid_path , model):
    imgPath = []
    n_frames = 11
    resize= 1
    v_cap = cv2.VideoCapture(vid_path)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    sample = np.linspace(0, v_len - 1, n_frames).astype(int) 

    frames = []
    for j in range(v_len):
        success = v_cap.grab()
        if j in sample:
            success, frame = v_cap.read()
            if not success:          
                continue
            
            frame = Image.fromarray(frame)
            
            if resize is not None:
                frame = frame.resize([int(d * resize) for d in frame.size])
                frame = np.asarray(frame)
            frames.append(frame)
    
    currentCount = 0
    success = 1
  
    while success and currentCount<min(n_frames,len(frames)):
        cv2.imwrite("/media_imgs/img_" + str(currentCount) + '.png' ,frames[int(currentCount)])
        imgPath.append("/media_imgs/img_" + str(currentCount) + '.png')
        currentCount += 1

    face_tensor = []

    for i in range(len(imgPath)):
        image = io.imread(imgPath[i])

        face_detector = dlib.get_frontal_face_detector()
        detected_faces = face_detector(image, 1)
        face_frames = [(x.left(), x.top(), x.right(), x.bottom()) for x in detected_faces]
        
        for n, face_rect in enumerate(face_frames):
            face = Image.fromarray(image).crop(face_rect)
            final_face = asarray(face)
            resize_face = cv2.resize(final_face,(128, 128), interpolation=cv2.INTER_CUBIC)
            finalImg = cv2.cvtColor(resize_face, cv2.COLOR_BGR2RGB)

            tensor = np.round(np.array(finalImg,dtype=np.float32))
            data_tensor = tf.convert_to_tensor(tensor)
            data_tensor_ = tf.expand_dims(data_tensor, axis=0)
            face_tensor.append(data_tensor_)

    count = 0
    for i in range(len(face_tensor)):
        p =  model.predict(face_tensor[i])
        pr = 0 if p[0][0] > p[0][1] else 1
        count += pr
    if count < 6:
        print('The video is FAKE!!')
    else :
        print('The video is REAL!!')

model_f = pickle.load(open('new_model.pkl','rb'))

print('Done')