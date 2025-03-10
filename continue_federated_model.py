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

##############################################
# Настройка логгера
##############################################
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('federated_logs_dnn_continue.log', mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

##############################################
# Глобальные параметры
##############################################
EPOCHS_PER_CLIENT = 10      # Сколько эпох локально обучаем на каждом устройстве
NUM_ROUNDS = 10            # Сколько раундов федеративного обучения
DEVICE_COUNTS = [5,10,15,20]    # Пример: 5 и 10 устройств; можете расширить [5,10,15,20,...]
BATCH_SIZE = 1024
LEARNING_RATE = 0.01

# Укажите CSV-файл (многоклассовый, с колонкой "Label")
DATA_FILE = "2017data/data_normalized.csv"
LABEL_COL = "Label"

# Укажите, хотим ли мы продолжить с уже готовой моделью
CONTINUE_MODEL_PATH = None # <-- Если не хотим продолжить, ставим None или указываем несуществующий файл.

##############################################
# Настройка GPU (по желанию)
##############################################
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logger.info(f"TensorFlow использует GPU: {gpus[0]}")
    except RuntimeError as e:
        logger.error(e)
else:
    logger.info("GPU не найден. TensorFlow использует CPU.")


##############################################
# Шаг 1: Читаем датасет и делим (80/20)
##############################################
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Файл {DATA_FILE} не найден.")

data = pd.read_csv(DATA_FILE)
data = data.replace([np.inf, -np.inf], np.nan).dropna()

if LABEL_COL not in data.columns:
    raise ValueError(f"Столбец '{LABEL_COL}' отсутствует в данных.")

# Если Label - object/строка, кодируем
if data[LABEL_COL].dtype == object:
    label_encoder = LabelEncoder()
    data[LABEL_COL] = label_encoder.fit_transform(data[LABEL_COL])
    joblib.dump(label_encoder, "federated_label_encoder.pkl")
    logger.info("Сохранён label_encoder в federated_label_encoder.pkl")

# Делим: 80% -> big_train_data, 20% -> manual_test_data
big_train_data, manual_test_data = train_test_split(data, test_size=0.2, random_state=42)

os.makedirs('data_split', exist_ok=True)
manual_test_data.to_csv('data_split/manual_test_data.csv', index=False)

logger.info(f"Полный размер датасета: {len(data)}")
logger.info(f"Для федеративного обучения (80%): {len(big_train_data)}")
logger.info(f"Для ручного тестирования (20%): {len(manual_test_data)}")


##############################################
# Функция создания DNN (многоклассовая)
##############################################
def create_dnn_model(input_dim, num_classes):
    """
    Простая многоклассовая модель с Dense-слоями.
    """
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


##############################################
# Делим big_train_data на устройства
##############################################
def split_for_devices(train_data: pd.DataFrame, num_devices: int, folder_path: str, label_col: str):
    os.makedirs(folder_path, exist_ok=True)
    rows_per_device = len(train_data) // num_devices

    client_files = []
    for i in range(num_devices):
        start_idx = i * rows_per_device
        end_idx = start_idx + rows_per_device if i != num_devices - 1 else len(train_data)
        
        device_data = train_data.iloc[start_idx:end_idx]
        client_file = os.path.join(folder_path, f"device_{i+1}.csv")
        device_data.to_csv(client_file, index=False)
        client_files.append(client_file)

    return client_files


##############################################
# Локальное обучение
##############################################
def train_local_model(file_path, global_weights, client_id, num_classes, label_col):
    data_local = pd.read_csv(file_path)
    if label_col not in data_local.columns:
        return None, None, None

    # (80/20) локальный train/val
    local_train_data, local_val_data = train_test_split(data_local, test_size=0.2, random_state=42)

    X_train = local_train_data.drop(columns=[label_col]).values
    y_train = local_train_data[label_col].values
    X_val = local_val_data.drop(columns=[label_col]).values
    y_val = local_val_data[label_col].values

    # Создаём модель и загружаем глобальные веса
    input_dim = X_train.shape[1]
    model = create_dnn_model(input_dim, num_classes)
    model.set_weights(global_weights)

    logger.info(f"[Устройство {client_id}] локальное обучение: train={len(X_train)}, val={len(X_val)}")

    # Обучаем EPOCHS_PER_CLIENT эпох
    with tqdm(total=EPOCHS_PER_CLIENT, desc=f"Device {client_id}", unit="epoch") as pbar:
        for _ in range(EPOCHS_PER_CLIENT):
            model.fit(X_train, y_train, epochs=1, batch_size=BATCH_SIZE, verbose=0)
            pbar.update(1)

    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    logger.info(f"[Устройство {client_id}] val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}")

    return model.get_weights(), val_loss, val_accuracy


##############################################
# Функция для «продолжения» федеративного обучения
##############################################
def federated_training(continue_from=None):
    """
    continue_from: путь к .h5-файлу с глобальной моделью, 
                   если None или не существует, инициализируем заново.
    """
    results = {}
    
    # Число классов
    all_labels = big_train_data[LABEL_COL].unique()
    num_classes = len(all_labels)

    for num_devices in DEVICE_COUNTS:
        logger.info(f"=== Федеративное обучение для {num_devices} устройств ===")

        # Делим датасет на num_devices
        folder_path = f"./data_split/devices_{num_devices}"
        client_files = split_for_devices(big_train_data, num_devices, folder_path, LABEL_COL)

        # Попробуем загрузить существующую модель
        if continue_from and os.path.exists(continue_from):
            logger.info(f"Загружаем существующую модель: {continue_from}")
            global_model = tf.keras.models.load_model(continue_from)
            global_weights = global_model.get_weights()
        else:
            # Инициализируем с нуля
            sample_data = pd.read_csv(client_files[0])
            X_sample = sample_data.drop(columns=[LABEL_COL]).values
            input_dim = X_sample.shape[1]
            global_model = create_dnn_model(input_dim, num_classes)
            global_weights = global_model.get_weights()

        rounds_loss = []
        rounds_acc = []

        # NUM_ROUNDS — сколько раундов делаем сейчас
        for round_num in range(1, NUM_ROUNDS + 1):
            logger.info(f"Раунд {round_num}/{NUM_ROUNDS} (устройств={num_devices})")
            local_weights_list = []
            local_losses = []
            local_accuracies = []

            # Обходим устройства
            for client_id, client_file in enumerate(client_files, start=1):
                updated_weights, loss, accuracy = train_local_model(
                    file_path=client_file,
                    global_weights=global_weights,
                    client_id=client_id,
                    num_classes=num_classes,
                    label_col=LABEL_COL
                )
                if updated_weights is not None:
                    local_weights_list.append(updated_weights)
                    local_losses.append(loss)
                    local_accuracies.append(accuracy)

            # Усредняем веса (федеративная агрегация)
            if local_weights_list:
                new_global_weights = []
                for layer_idx in range(len(local_weights_list[0])):
                    layer_collector = [lw[layer_idx] for lw in local_weights_list]
                    avg_layer = np.mean(layer_collector, axis=0)
                    new_global_weights.append(avg_layer)
                global_weights = new_global_weights

            # Средние локальные метрики
            if local_losses and local_accuracies:
                avg_loss = np.mean(local_losses)
                avg_acc = np.mean(local_accuracies)
                rounds_loss.append(avg_loss)
                rounds_acc.append(avg_acc)
                logger.info(f"[Среднее по устройствам] val_loss={avg_loss:.4f}, val_acc={avg_acc:.4f}")

        # Сохраняем веса в глобальную модель
        global_model.set_weights(global_weights)
        
        # Сохраняем новую/продолженную модель
        model_name = f"federated_global_model_{num_devices}_continued.h5"
        global_model.save(model_name)
        logger.info(f"[{num_devices} устройств] Итоговая глобальная модель (продолженная) -> {model_name}")

        results[num_devices] = {
            'loss': rounds_loss,
            'accuracy': rounds_acc
        }

    return results

##############################################
# Запуск скрипта
##############################################
if __name__ == "__main__":
    # Запускаем федеративное обучение,
    # при этом попытаемся продолжить с CONTINUE_MODEL_PATH (если он задан и существует)
    fed_results = federated_training(continue_from=CONTINUE_MODEL_PATH)

    # И, для примера, построим графики Loss/Acc (если нужно)
    for num_devs, metrics in fed_results.items():
        rounds_axis = range(1, len(metrics['loss']) + 1)
        
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.plot(rounds_axis, metrics['loss'], marker='o')
        plt.title(f"{num_devs} устройств: Loss")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.grid(True)

        plt.subplot(1,2,2)
        plt.plot(rounds_axis, metrics['accuracy'], marker='o')
        plt.title(f"{num_devs} устройств: Accuracy")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.grid(True)

        plt.tight_layout()
        plt.show()
