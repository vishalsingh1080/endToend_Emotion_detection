# import streamlit as st
import tensorflow as tf
import cv2
import numpy as np


model = tf.keras.models.load_model('VGG16-global.h5')
classes = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}
print('Model loaded Successfully!')

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError('cannot open webcam')

def normalize(x):
    return (x.astype(float) - 128) / 128

while True:
     ret,frame = cap.read()
     facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
     colour = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

     faces = facecascade.detectMultiScale(colour,1.1,4)
     for x,y,w,h in faces:
        #roi_gray = gray[y:y+h,x:x+w]
        roi_color = colour[y:y+h,x:x+w]
        cv2.rectangle(colour,(x,y),(x+w,y+h),(255,0,0),2)
        faces1 = facecascade.detectMultiScale(roi_color)
        for (ex,ey,ew,eh) in faces1:
            face_roi = roi_color[ey: ey+eh,ex: ex+ew]    
    
     face_roi = normalize(face_roi)

     final_img = cv2.resize(face_roi,(197,197))
     final_img = np.expand_dims(final_img,axis=0)
     
     preds = model.predict(final_img)[0]
     i = np.argmax(preds)
     label = classes[i]

     font = cv2.FONT_HERSHEY_SIMPLEX
     org = (50, 50)
     fontScale = 1
     color = (255, 0, 0)
     thickness = 2
     frame = cv2.putText(frame, label, org, font,fontScale, color, thickness, cv2.LINE_AA)

     cv2.imshow('img', frame)
     # Stop if (Q) key is pressed
     k = cv2.waitKey(30)
     if k==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


