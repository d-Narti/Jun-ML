import telebot
from telebot import types
from keras.models import load_model
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import model_test

# Загрузка модели
model = load_model('trained_brain_tumor_model.keras')

# Инициализация бота
TOKEN = '**'
bot = telebot.TeleBot(TOKEN)

# Размер изображения
img_size = (224, 224)

# Определяем названия классов
classes = ['Глиома', 'Менингиома', 'Без опухоли', 'Гипофиз']

# Обработчик команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    item1 = types.KeyboardButton("Полный анализ")
    item2 = types.KeyboardButton("Результат анализа")
    markup.add(item1, item2)
    bot.send_message(message.chat.id,
                     "Привет! Я бот для классификации опухолей головного мозга по снимкам МРТ. Моя задача в том, чтобы помочь вам диагностировать опухоль. Я умею классифицировать 3 вида опухоли: Глиому, Менингиому, Гипофиз, а также скажу, если опухоли на снимке нет. Выберите один из вариантов анализа ниже.",
                     reply_markup=markup)

# Обработчик команды /help
@bot.message_handler(commands=['help'])
def send_help(message):
    bot.reply_to(message,
                 "Отправьте мне изображение МРТ, и я определю, есть ли признаки опухоли, а также предоставлю сегментированное изображение по запросу.")

# Обработчик кнопок
@bot.message_handler(func=lambda message: message.text in ["Полный анализ", "Результат анализа"])
def handle_analysis_choice(message):
    if message.text == "Полный анализ":
        msg = bot.reply_to(message, "Отправьте изображение для полного анализа.")
        bot.register_next_step_handler(msg, process_image, full_analysis=True)
    elif message.text == "Результат анализа":
        msg = bot.reply_to(message, "Отправьте изображение для анализа.")
        bot.register_next_step_handler(msg, process_image, full_analysis=False)

# Обработчик изображений
def process_image(message, full_analysis):
    try:
        # Получаем изображение
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # Загружаем изображение в формате PIL
        img = Image.open(io.BytesIO(downloaded_file))

        # Преобразуем изображение в массив
        img_array = my_model_test.get_img_array(img)

        # Делаем предсказание
        prediction, top_class = my_model_test.make_prediction(img_array, model)

        # Отправляем результат
        if full_analysis:
            result_text = "Вероятности классов:\n" + "\n".join(prediction)
        else:
            result_text = f"Результат анализа: {top_class}"

        bot.reply_to(message, result_text)
    except Exception as e:
        bot.reply_to(message, f"Произошла ошибка при обработке изображения: {e}")

# Запуск бота
bot.polling()
