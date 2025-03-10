import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import joblib


# Настройка логгера
def setup_logger():
    """
    Настройка логгера для записи в файл.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Создание форматтера
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Создание обработчика для записи в файл
    file_handler = logging.FileHandler('training_logs_dnn2017.log', mode='w')
    file_handler.setFormatter(formatter)
    
    # Добавление обработчика к логгеру
    logger.addHandler(file_handler)
    
    return logger

# Логгер
logger = setup_logger()

# Параметры обучения
EPOCHS_PER_ROUND = 10  # Эпох на раунд
NUM_ROUNDS = 1         # Количество «раундов» (можете оставить 1)
BATCH_SIZE = 9192
LEARNING_RATE = 0.05
DATA_FILE = "2017data/data_normalized.csv"

# Создание DNN модели (многоклассовый вариант)
def create_dnn_model(input_dim, num_classes):
    """
    Создаёт многоклассовую DNN-модель.
    Параметры:
      - input_dim: число признаков (int)
      - num_classes: число уникальных классов (int)
    """
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),  # input_dim должен быть кортежем (например, (N,))
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # многоклассовая классификация
    ])
    
    # Для целочисленных меток используем sparse_categorical_crossentropy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_and_preprocess_data(data_file):
    """
    Загрузка и предобработка данных.
    1) Считываем CSV
    2) Убираем inf и NaN
    3) Кодируем текстовые метки в целые числа
    4) Делим на train/test (80/20)
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Файл {data_file} не найден.")
    
    data = pd.read_csv(data_file)
    # Заменяем ±inf на NaN и удаляем строки с NaN
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    
    if 'Label' not in data.columns:
        raise ValueError("Столбец 'Label' отсутствует в данных.")
    
    # Кодируем метку, если она строковая (или вообще категориальная)
    if data['Label'].dtype == object:
        label_encoder = LabelEncoder()
        data['Label'] = label_encoder.fit_transform(data['Label'])
        joblib.dump(label_encoder, "label_encoder.pkl")

    # Делим на train/test
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    return train_data, test_data

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    """
    Обучение модели (c tqdm-колбэком).
    """
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[tqdm_callback()]
    )
    return history

def tqdm_callback():
    """
    Callback для tqdm (просто выводит текущий номер эпохи).
    """
    class TqdmCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"\rEpoch {epoch + 1}/{EPOCHS_PER_ROUND}", end="")
    return TqdmCallback()

def plot_training_history(history):
    """
    Построение графиков обучения (loss и accuracy).
    Исправляем xLabel -> xlabel, yLabel -> ylabel.
    """
    plt.figure(figsize=(12, 5))
    
    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Загрузка данных
    train_data, test_data = load_and_preprocess_data(DATA_FILE)
    logger.info(f"Полный размер датасета: {len(train_data) + len(test_data)}")
    logger.info(f"Для обучения (80%): {len(train_data)}")
    logger.info(f"Для тестирования (20%): {len(test_data)}")
    
    # Подготовка данных (разделяем признаки и метки)
    X_train = train_data.drop(columns=['Label']).values
    y_train = train_data['Label'].values
    X_test = test_data.drop(columns=['Label']).values
    y_test = test_data['Label'].values
    
    # Определяем число признаков и число уникальных классов
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))  # кол-во уникальных меток
    
    logger.info(f"Число признаков (input_dim): {input_dim}")
    logger.info(f"Число классов (num_classes): {num_classes}")
    
    # Создаём папку для сохранения моделей
    os.makedirs("saved_models", exist_ok=True)
    
    # Обучаем в цикле NUM_ROUNDS (может быть и 1)
    for round_num in range(1, NUM_ROUNDS + 1):
        logger.info(f"=== Раунд {round_num}/{NUM_ROUNDS} ===")
        
        # Создание новой модели для каждого раунда
        model = create_dnn_model(input_dim, num_classes)
        logger.info("Модель создана (многокласс).")
        
        # Обучение
        logger.info("Начало обучения...")
        history = train_model(model, X_train, y_train, X_test, y_test, EPOCHS_PER_ROUND, BATCH_SIZE)
        logger.info("Обучение завершено.")
        
        # Оценка
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        # Сохранение модели
        model_name = f"saved_models/model_round_{round_num}v3.h5"
        model.save(model_name)
        logger.info(f"Модель сохранена: {model_name}")
    
    # Построение графиков для последнего раунда
    plot_training_history(history)

if __name__ == "__main__":
    main()
