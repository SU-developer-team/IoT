import os
import gc
import random
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm

class FederatedLearning:
    def __init__(self, data_file, label_col, model_type="DNN", seed=42):
        """
        Initialize the federated learning module.
        :param data_file: Path to the dataset file.
        :param label_col: Name of the label column in the dataset.
        :param model_type: Type of model to use ("DNN" or "CNN_BiLSTM").
        :param seed: Random seed for reproducibility.
        """
        self.SEED = seed
        self.DATA_FILE = data_file
        self.LABEL_COL = label_col
        self.MODEL_TYPE = model_type

        # Hyperparameters
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 0.001
        self.EPOCHS_PER_CLIENT = 8
        self.NUM_ROUNDS = 5
        self.DEVICE_COUNTS = [100, 200, 300]
        self.MANUAL_DATA_SIZE = 0.001

        # Data and model placeholders
        self.data = None
        self.big_train_data = None
        self.manual_test_data = None
        self.global_model = None
        self.global_weights = None

        # Set random seeds
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        tf.random.set_seed(self.SEED)

        # GPU configuration
        self._configure_gpu()

        # Experiment directory setup
        self.experiment_dir = self._create_experiment_directory()

        # Настраиваем логирование в файл внутри каталога эксперимента
        log_file = os.path.join(self.experiment_dir, "federated_training.log")
        logging.basicConfig(
            filename=log_file,
            filemode="w",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("FederatedLearning initialized with custom logging")

    def _configure_gpu(self):
        """Configure GPU settings."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.set_visible_devices(gpus[0], 'GPU')
                # print(f"Using GPU: {gpus[0]}")
            except Exception as e:
                self.logger.error(f"Error configuring GPU: {e}")
        else:
            pass
            # print("GPU not found, using CPU.")

    def _create_experiment_directory(self):
        """Create a new experiment directory with an auto-incremented name."""
        base_dir = "experiments"
        os.makedirs(base_dir, exist_ok=True)

        # Find the next available experiment number
        existing_dirs = [d for d in os.listdir(base_dir) if d.startswith("exp_")]
        exp_numbers = [int(d.split("_")[1]) for d in existing_dirs if "_" in d]
        next_exp_number = max(exp_numbers, default=0) + 1

        # Create the new experiment directory
        experiment_dir = os.path.join(base_dir, f"exp_{next_exp_number}")
        os.makedirs(experiment_dir, exist_ok=True)

        # print(f"Created experiment directory: {experiment_dir}")
        return experiment_dir

    def load_and_preprocess_data(self):
        """Load and preprocess the dataset."""
        if not os.path.exists(self.DATA_FILE):
            raise FileNotFoundError(f"File '{self.DATA_FILE}' not found!")

        self.data = pd.read_csv(self.DATA_FILE)
        # Убираем бесконечности
        self.data = self.data.replace([np.inf, -np.inf], np.nan).dropna()

        if self.LABEL_COL not in self.data.columns:
            raise ValueError(f"Column '{self.LABEL_COL}' not found in the dataset!")

        # Encode labels if they are strings
        if self.data[self.LABEL_COL].dtype == object or True:
            label_encoder = LabelEncoder()
            self.data[self.LABEL_COL] = label_encoder.fit_transform(self.data[self.LABEL_COL])
            joblib.dump(label_encoder, os.path.join(self.experiment_dir, "label_encoder.pkl"))
            self.logger.info("LabelEncoder saved to label_encoder.pkl")

        self.big_train_data, self.manual_test_data = train_test_split(
            self.data, test_size=self.MANUAL_DATA_SIZE, random_state=self.SEED
        )

        os.makedirs(os.path.join(self.experiment_dir, "data_split"), exist_ok=True)
        self.manual_test_data.to_csv(os.path.join(self.experiment_dir, "data_split/manual_test_data.csv"), index=False)

        self.logger.info(f"Total dataset size: {len(self.data)}")
        self.logger.info(f"For federated training: {len(self.big_train_data)}")
        self.logger.info(f"For manual testing: {len(self.manual_test_data)}")
        self.logger.info(f"Data loaded and preprocessed")
        self.logger.info(f"Running device counts: {self.DEVICE_COUNTS}")
        self.logger.info(f"Model type: {self.MODEL_TYPE}")
        self.logger.info(f"Random seed: {self.SEED}")
        self.logger.info(f"Hyperparameters: "
                         f"batch_size={self.BATCH_SIZE}, "
                         f"learning_rate={self.LEARNING_RATE}, "
                         f"epochs_per_client={self.EPOCHS_PER_CLIENT}, "
                         f"num_rounds={self.NUM_ROUNDS}")
        self.logger.info(f"Manual data size: {self.MANUAL_DATA_SIZE}")
        self.logger.info(f"Experiment directory: {self.experiment_dir}")
        self.logger.info(f"Data file: {self.DATA_FILE}")

    def create_model(self, input_dim, num_classes):
        """Create a model based on the specified type."""
        if self.MODEL_TYPE == "DNN":
            return self._create_dnn_model(input_dim, num_classes)
        elif self.MODEL_TYPE == "CNN_BiLSTM":
            return self._create_cnn_bilstm_model(input_dim, num_classes)
        else:
            raise ValueError(f"Unsupported model type: {self.MODEL_TYPE}")

    def _create_dnn_model(self, input_dim, num_classes):
        """Create a DNN model."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def _create_cnn_bilstm_model(self, input_dim, num_classes):
        """Create a CNN-BiLSTM model adapted for multi-class classification."""
        model = tf.keras.Sequential([
            tf.keras.layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
            tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.2)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, dropout=0.2)),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def split_for_devices(self, train_data, num_devices, folder_path):
        """
        Split training data into subsets for each device.
        """
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

    def train_local_model(self, file_path, global_weights, client_id, num_classes):
        """
        Train a local model on a client's data, using a single progress bar for all epochs.
        """
        local_data = pd.read_csv(file_path)
        if self.LABEL_COL not in local_data.columns:
            return None, None, None

        # Локальный сплит
        local_train, local_val = train_test_split(local_data, test_size=0.2, random_state=self.SEED)

        # Разделяем на X и y
        X_train = local_train.drop(columns=[self.LABEL_COL]).values
        y_train = local_train[self.LABEL_COL].values
        X_val = local_val.drop(columns=[self.LABEL_COL]).values
        y_val = local_val[self.LABEL_COL].values

        # Создаем и инициализируем модель
        input_dim = X_train.shape[1]
        model = self.create_model(input_dim, num_classes)
        model.set_weights(global_weights)

        msg = f"[Device {client_id}] Local training: train={len(X_train)}, val={len(X_val)}"
        self.logger.info(msg)
        # print(msg)

        # Создаём датасеты
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=4096, seed=self.SEED).batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        # Один прогресс-бар на все эпохи обучения на данном устройстве
        with tqdm(total=self.EPOCHS_PER_CLIENT, desc=f"Device {client_id}", unit="epoch") as pbar:
            for epoch_i in range(self.EPOCHS_PER_CLIENT):
                # Обучаем модель ровно 1 эпоху
                history = model.fit(train_dataset, validation_data=val_dataset, epochs=1, verbose=0)

                # Достаём метрики
                train_loss = history.history['loss'][-1]
                train_acc = history.history['accuracy'][-1]
                val_loss = history.history['val_loss'][-1]
                val_acc = history.history['val_accuracy'][-1]

                # Обновляем прогресс-бар, постфикс - метрики
                pbar.update(1)
                pbar.set_postfix({
                    "train_loss": f"{train_loss:.4f}",
                    "train_acc": f"{train_acc:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "val_acc": f"{val_acc:.4f}"
                })

                # Печатаем и логируем детально, если хотите
                epoch_msg = (f"[Device {client_id}, Epoch {epoch_i+1}/{self.EPOCHS_PER_CLIENT}] "
                             f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                             f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
                # self.logger.info(epoch_msg)

        # Итоговая оценка на валидации после всех эпох
        final_val_loss, final_val_acc = model.evaluate(val_dataset, verbose=0)
        last_msg = f"[Device {client_id}] FINAL val_loss={final_val_loss:.4f}, val_acc={final_val_acc:.4f}"
        self.logger.info(last_msg)
        # print(last_msg)

        updated_weights = model.get_weights()

        # Освобождаем ресурсы
        del local_data, local_train, local_val, X_train, X_val, y_train, y_val, train_dataset, val_dataset, model
        tf.keras.backend.clear_session()
        gc.collect()

        return updated_weights, final_val_loss, final_val_acc

    def federated_training(self):
        """Perform federated learning."""
        results = {}
        num_classes = len(self.big_train_data[self.LABEL_COL].unique())
        # print(f"Number of classes: {num_classes}")

        for num_devices in self.DEVICE_COUNTS:
            self.logger.info(f"=== Federated Learning: {num_devices} devices ===")
            # print(f"\n========== FEDERATED LEARNING with {num_devices} DEVICES ==========")
            folder_path = os.path.join(self.experiment_dir, f"data_split/dataset_{num_devices}")
            client_files = self.split_for_devices(self.big_train_data, num_devices, folder_path)

            rounds_loss = []
            rounds_acc = []

            for round_idx in range(self.NUM_ROUNDS):
                self.logger.info(f"Round {round_idx+1}/{self.NUM_ROUNDS} (for {num_devices} devices)")
                # print(f"\n--- Round {round_idx+1}/{self.NUM_ROUNDS} ---")
                local_weights_list = []
                local_losses = []
                local_accuracies = []

                for client_id, client_file in enumerate(client_files, start=1):
                    # Инициализация глобальных весов (если ещё нет)
                    if self.global_weights is None:
                        sample_df = pd.read_csv(client_file, nrows=1)
                        sample_x = sample_df.drop(columns=[self.LABEL_COL]).values
                        input_dim = sample_x.shape[1]
                        self.global_model = self.create_model(input_dim, num_classes)
                        self.global_weights = self.global_model.get_weights()

                    # Локальное обучение
                    updated_weights, loss, acc = self.train_local_model(
                        file_path=client_file,
                        global_weights=self.global_weights,
                        client_id=client_id,
                        num_classes=num_classes
                    )
                    if updated_weights is not None:
                        local_weights_list.append(updated_weights)
                        local_losses.append(loss)
                        local_accuracies.append(acc)

                # Усредняем веса
                if local_weights_list:
                    new_global_weights = []
                    for layer_idx in range(len(local_weights_list[0])):
                        layer_stack = np.array([w[layer_idx] for w in local_weights_list])
                        new_global_weights.append(np.mean(layer_stack, axis=0))
                    self.global_weights = new_global_weights

                # Средние метрики по устройствам в этом раунде
                if local_losses and local_accuracies:
                    avg_loss = np.mean(local_losses)
                    avg_acc = np.mean(local_accuracies)
                    rounds_loss.append(avg_loss)
                    rounds_acc.append(avg_acc)
                    msg = (f"[Devices={num_devices}, Round={round_idx+1}] "
                           f"avg_val_loss={avg_loss:.4f}, avg_val_acc={avg_acc:.4f}")
                    self.logger.info(msg)
                    # print(msg)

                # Очистка
                del local_weights_list, local_losses, local_accuracies
                gc.collect()

            # Сохраняем метрики для всех раундов (данное число устройств)
            results[num_devices] = {'loss': rounds_loss, 'acc': rounds_acc}

            # Финальные метрики (после всех раундов)
            if rounds_loss and rounds_acc:
                # print(f"\n==== FINAL (AVERAGE) METRICS for {num_devices} DEVICES ====")
                # print(f"Final Loss: {rounds_loss[-1]:.4f}, Final Accuracy: {rounds_acc[-1]:.4f}")
                pass
            # Сохраняем итоговую глобальную модель
            if self.global_model is None and self.global_weights is not None:
                sample_x = self.big_train_data.drop(columns=[self.LABEL_COL]).values[:1]
                input_dim = sample_x.shape[1]
                self.global_model = self.create_model(input_dim, num_classes)

            if self.global_model is not None and self.global_weights is not None:
                self.global_model.set_weights(self.global_weights)
                final_name = os.path.join(self.experiment_dir, f"federated_global_model_{num_devices}.h5")
                self.global_model.save(final_name)
                self.logger.info(f"Saved global model: {final_name}")
                # print(f"Saved global model: {final_name}")

            # Сброс модели
            self.global_model = None
            self.global_weights = None

        return results

    def plot_results(self, results):
        """Plot federated learning results for multiple device counts on one figure."""
        plt.figure(figsize=(12, 6))

        # Plot Loss
        plt.subplot(1, 2, 1)
        for num_devs, metrics in results.items():
            rounds_axis = range(1, len(metrics['loss']) + 1)
            plt.plot(rounds_axis, metrics['loss'], marker='o', label=f"{num_devs} устройств")
        plt.title("Loss по раундам")
        plt.xlabel("Раунд")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        for num_devs, metrics in results.items():
            rounds_axis = range(1, len(metrics['acc']) + 1)
            plt.plot(rounds_axis, metrics['acc'], marker='o', label=f"{num_devs} устройств")
        plt.title("Accuracy по раундам")
        plt.xlabel("Раунд")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plot_path = os.path.join(self.experiment_dir, "federated_results_plot.png")
        plt.savefig(plot_path)
        plt.show()
        self.logger.info(f"Saved combined plot: {plot_path}")
        # print(f"Saved combined plot: {plot_path}")


if __name__ == "__main__":
    # Пример использования
    fl = FederatedLearning(
        data_file="dataset_container/dataset_one_ddos/dataset.csv",
        label_col="label",
        model_type="CNN_BiLSTM"  # Или "DNN"
    )
    fl.load_and_preprocess_data()
    # Пример: можно указать несколько значений числа устройств
    results = fl.federated_training()
    fl.plot_results(results)
