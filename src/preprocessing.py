import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

def preprocess_data(df):
    print("Preprocessing data...")
    
    img_width, img_height = 48, 48
    
    pixels = df['pixels'].tolist()
    faces = []
    
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(img_width, img_height)
        faces.append(face.astype('float32'))
        
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    
    faces /= 255.0
    
    emotions = to_categorical(df['emotion'], num_classes=7)
    
    return faces, emotions
