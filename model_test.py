import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model
from keras.utils import load_img, img_to_array
from keras.preprocessing import image

# Загрузка модели
model_path = 'trained_brain_tumor_model.keras'
model = load_model(model_path)

# Определение функций
last_conv_layer_name = "Top_Conv_Layer"
img_size = (224, 224)

def get_img_array(img_path, size):

    img = load_img(img_path, target_size=size)
    array = img_to_array(img)
    array = np.expand_dims(array, axis=0)

    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Создание модели, которая сопоставляет входное изображение с активациями последнего conv слоя, а также выходным предсказаниям
    grad_model = keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    # Вычисление градиента топ-1 предсказанного класса для входного изображения относительно активаций последнего conv слоя
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Это градиент выходного нейрона (топ-1 предсказанного или выбранного) по отношению к выходной карте признаков последнего conv слоя
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Это вектор, каждая запись которого представляет собой среднюю интенсивность градиента
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Умножение каждого канала на "степень важности этого канала" по отношению к топ-1 предсказанному классу и суммирование всех каналов,
    # для получения тепловой карты активации класса
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Нормализация тепловой карты между 0 и 1 для визуализации
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Загрузка исходного изображения и преобразование его в массив.
    img = load_img(img_path)
    img = img_to_array(img)

    # Преобразование тепловой карты (heatmap) в 8-битное изображение.
    heatmap = np.uint8(255 * heatmap)

    # Применение цветовой карты "jet" к тепловой карте.
    jet = plt.cm.get_cmap("jet")

    # Использование значений RGB для цветовой карты
    jet_colors = jet(np.arange(256))[:, :3]

    # Создание изображения с тепловой картой в RGB палитре
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = img_to_array(jet_heatmap)
    jet_heatmap = tf.image.resize(jet_heatmap, (img.shape[0], img.shape[1]))

    # Наложение тепловой карты на исходное изображение
    superimposed_img = jet_heatmap * alpha + img

    # Сохранение наложенного изображения
    superimposed_img = keras.utils.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)

    return cam_path

def decode_predictions(preds):
    classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    return classes[np.argmax(preds)]

def make_prediction(img_path, model, last_conv_layer_name, cam_path="cam.jpg"):
    img_array = get_img_array(img_path, img_size)
    preds = model.predict(img_array)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    cam_path = save_and_display_gradcam(img_path, heatmap, cam_path)
    prediction = decode_predictions(preds)
    return cam_path, prediction


