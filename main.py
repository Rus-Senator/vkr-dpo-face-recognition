import PIL.Image
import PIL.ImageDraw
import face_recognition as fr
import streamlit as st
import io

def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return io.BytesIO(image_data)
    else:
        return None

st.title('WEB-приложение: распознавание лиц на фото')
st.write('Распознает лица на фотографии, рисует квадрат красного цвета на области лица')
img = load_image()

result = st.button('Распознать лица')

if result:

    if img:
        img = fr.load_image_file(img)
        face_loc = fr.face_locations(img,number_of_times_to_upsample=2)
        no_of_faces = len(face_loc)

        pil_image = PIL.Image.fromarray(img)
        for face_location in face_loc:
            top,right,bottom,left = face_location
            draw_shape = PIL.ImageDraw.Draw(pil_image)
            draw_shape.rectangle([left, top, right, bottom], outline="red",width=5)

        st.write(f'Число лиц на изображении: {no_of_faces}')
        st.image(pil_image)
    else:
        st.write('Загрузите изображение для распознавания')
    #pil_image.save("output.jpg")