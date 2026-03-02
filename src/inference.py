import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import traceback
import sys

def run_real_time_inference():
    print("Starting real-time inference...")
    log_file = open("inference_log.txt", "w")
    
    try:
        model_path = 'models/emotion_model.h5'
        if not os.path.exists(model_path):
            msg = "Model not found. Please train the model first."
            print(msg)
            log_file.write(msg + "\n")
            return
            
        try:
            model = load_model(model_path)
        except Exception as e:
             msg = f"Error loading model: {e}"
             print(msg)
             log_file.write(msg + "\n")
             return

        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if face_cascade.empty():
            msg = f"Error: Could not load Haar Cascade from {cascade_path}"
            print(msg)
            log_file.write(msg + "\n")
            return
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            msg = "Error: Could not open webcam. Ensure camera is connected and permissions are granted."
            print(msg)
            log_file.write(msg + "\n")
            return
            
        print("Press 'q' to quit.")
        log_file.write("Camera opened successfully. Starting loop.\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                log_file.write("Failed to read frame.\n")
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_pixels = roi_gray.astype('float32') / 255.0
                roi_pixels = np.expand_dims(roi_pixels, axis=0)
                roi_pixels = np.expand_dims(roi_pixels, axis=-1)
                
                prediction = model.predict(roi_pixels, verbose=0)
                max_index = int(np.argmax(prediction))
                emotion = emotion_labels[max_index]
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
            cv2.imshow('Facial Emotion Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        log_file.write("Inference finished successfully.\n")
        
    except Exception as e:
        msg = f"An unexpected error occurred: {e}\n{traceback.format_exc()}"
        print(msg)
        log_file.write(msg)
    finally:
        log_file.close()
