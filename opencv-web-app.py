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
        return ''


st.title('WEB-приложение: распознавание лиц на фото')

st.write('Распознает лица на фотографии, рисует квадрат синего цвета вокруг распознанной области лица. Используется '
         'OpenCV, каскад Хаара haarcascade_frontalface_default.xml. ')
img = load_image()

result = st.button('Распознать лица')

if result:

    if len(img):
        # Загружаем Каскад
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        if img.shape[1] > 1000:
            dsize = (1000, int((1000 / img.shape[1]) * img.shape[0]))
            img = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)

        # Преобразование в серый, нужно для OpenCV
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Вызов метода обнаружения лиц
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(20, 20))
        st.write(f'Распознано лиц: {len(faces)}')
        # Рисуем квадрат по точкам
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Отображаем содержимое через st
        st.image(img, channels="BGR")
