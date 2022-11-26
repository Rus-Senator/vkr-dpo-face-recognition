import cv2
import streamlit as st
import numpy as np

def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        return opencv_image
    else:
        return None

st.title('WEB-приложение: распознавание лиц на фото')
st.write('Распознает лица на фотографии, рисует квадрат красного цвета на области лица')
img = load_image()

result = st.button('Распознать лица')

if result:

    if len(img):
        # Load the cascade
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # Read the input image
        #img = cv2.imread(img)

        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1,4)
        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Display the output
        st.image(img,channels="BGR")
        #cv2.imshow('img', img)
