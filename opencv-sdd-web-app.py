import cv2
import numpy as np

# https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
prototxt_path = "weights-prototxt.txt"

# https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
#model_path = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
model_path = "res_ssd_300Dim.caffeModel"

# загрузим модель Caffe
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# читаем изображение
image = cv2.imread("test.png")

# получаем ширину и высоту изображения
(height, width) = image.shape[:2]

# предварительная обработка: изменение размера и вычитание среднего
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))

# устанавливаем на вход нейронной сети изображение
model.setInput(blob)
# выполняем логический вывод и получаем результат
detections = model.forward()

# loop over the detections to extract specific confidence
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the
    # prediction
    confidence = detections[0, 0, i, 2]

    # greater than the minimum confidence
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        (x1, y1, x2, y2) = box.astype("int")

        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100) + " ( " + str(y2 - y1) + ", " + str(x2 - x1) + " )"
        y = y1 - 10 if y1 - 10 > 10 else y1 + 10
        cv2.rectangle(image, (x1, y1), (x2, y2),
                      (0, 0, 255), 2)
        cv2.putText(image, text, (x1, y),
                    cv2.LINE_AA, 0.45, (0, 0, 255), 2)
# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
