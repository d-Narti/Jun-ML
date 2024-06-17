import os
import keras
from keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.applications import EfficientNetB0

def load_data(train_dir, test_dir, img_size, batch_size):
    train_data = keras.preprocessing.image_dataset_from_directory(
        directory=train_dir,
        image_size=img_size,
        label_mode="categorical",
        batch_size=batch_size,
    )

    test_data = keras.preprocessing.image_dataset_from_directory(
        directory=test_dir,
        image_size=img_size,
        label_mode="categorical",
        batch_size=batch_size,
    )
    return train_data, test_data

def build_and_fit_model(train_data, test_data, epochs):

    base_model = EfficientNetB0(include_top=False)
    base_model.trainable = False
    inputs = Input(shape=(224, 224, 3), name="Input_layer")
    x = base_model(inputs)
    x = Conv2D(32, 3, padding='same', activation="relu", name="Top_Conv_Layer")(x)
    x = GlobalAveragePooling2D(name="Global_avg_Pooling_2D")(x)
    outputs = Dense(4, activation="softmax", name="Output_layer")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"]
    )
    callback_list = [
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.8, monitor="val_accuracy", patience=3)
    ]
    # Шаг 1 Обучение
    model.fit(train_data, validation_data=test_data, epochs=epochs, verbose=1, callbacks=callback_list)
    # Шаг 2 Fine-Tuning
    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        metrics=["accuracy"]
    )
    fine_tune_epochs = 5
    initial_epoch = epochs
    model.fit(train_data, epochs=initial_epoch + fine_tune_epochs,
                        validation_data=test_data, validation_steps=len(test_data),
                        initial_epoch=initial_epoch)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.00001),
        metrics=["accuracy"]
    )

    initial_epoch += fine_tune_epochs
    fine_tune_epochs = 1

    model.fit(train_data, epochs=initial_epoch + fine_tune_epochs,
                        validation_data=test_data, validation_steps=len(test_data),
                        initial_epoch=initial_epoch)

    # Оценка итоговой точности модели
    loss, accuracy = model.evaluate(test_data)
    print(f'Итоговая точность модели: {accuracy * 100:.2f}%')
    print(f'Итоговая ошибка модели: {loss:.4f}')
    return model

def save_model(model, file_path):
    # Проверяем, существует ли файл, и удаляем его перед сохранением новой модели
    if os.path.exists(file_path):
        os.remove(file_path)
    # Сохраняем модель в формате .keras
    model.save(file_path, save_format='keras')

if __name__ == "__main__":
    train_dir = 'Training'
    test_dir = 'Testing'
    img_size = (224, 224)
    batch_size = 32
    epochs = 5  # Number of epochs for initial training

    train_data, test_data = load_data(train_dir, test_dir, img_size, batch_size)
    trained_model = build_and_fit_model(train_data, test_data, epochs=epochs)
    save_model(trained_model, 'trained_brain_tumor_model_2.keras')  # Save the model
