import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
# from streamlit_webrtc import  WebRtcMode, RTCConfiguration
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import cv2
import numpy as np
import gdown

link = 'https://drive.google.com/uc?id=1JEqLQ3AL9_9aZelSGvPowdVMFpnhBz7t'
output = 'model.h5'
gdown.download(link, output, quiet=False)

model = tf.keras.models.load_model(output)
classes = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}
print('Model loaded Successfully!')

def normalize(x):
    return (x.astype(float) - 128) / 128

facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# RTC_CONFIGURATION = RTCConfiguration(
#     {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
# )


class VideoTransformer(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img_colour = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = facecascade.detectMultiScale(image=img_colour, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_color = img_colour[y:y + h, x:x + w]
            roi_color = cv2.resize(roi_color, (197, 197), interpolation=cv2.INTER_AREA)
            if np.sum([roi_color]) != 0:
                roi = normalize(roi_color)
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = model.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = classes[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    
    st.title("Emotion Detection using Live Webcam")

    st.sidebar.markdown(
        """ Developed by Vishal Singh    
            Email : vishalsingh1080@gmail.com  
            [LinkedIn] (https://www.linkedin.com/in/vishal-singh-20a180108/)""")

    st.header("Webcam Live Feed")
    st.write("Click on start to use webcam for real-time emotion detection.")
    webrtc_streamer(key="example",video_processor_factory=VideoTransformer,async_processing=True)

if __name__ == "__main__":
    main()            


    