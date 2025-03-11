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

# ---------------------------------------------------------------------
# ФУНКЦИЯ: Создаём 3 CSV для трёх стадий (1 – сбалансировано 50:50, 2 – 2 млн, 3 – 1 млн)
# ---------------------------------------------------------------------

def create_stage_datasets_balanced(original_csv, output_dir):
    """
    Исходные данные (3 млн строк):
      - label=0 => 1 млн (normal)
      - label=1 => 1 млн (другие атаки)
      - label=2..11 => 10 классов DDoS, по 100k (итого 1 млн)

    1) Stage1.csv: 
       - Берём ровно 1 млн normal (label=0) и 1 млн attack (label=1..11).
         => Итого 2 млн, label=0/1 (0=normal, 1=attack).

    2) Stage2.csv: 
       - Все строки, где label!=0 => 2 млн (1 млн «другие атаки» + 1 млн DDoS).
         => label=0 (не-DDoS) если orig.label=1, label=1 (DDoS) если orig.label=2..11.

    3) Stage3.csv:
       - Все строки, где label=2..11 => 1 млн (10 классов DDoS), 
         ставим label=0..9 (orig.label-2).
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(original_csv)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if 'label' not in df.columns:
        raise ValueError("В датасете нет столбца 'label'!")

    # --- 1) Stage1 (сбалансировать 1:1) ---
    df_normal = df[df['label'] == 0].copy()   # 1 млн
    df_attack = df[df['label'] != 0].copy()   # 2 млн
    # Возьмём min(1e6, 2e6) => 1e6 строк normal и 1e6 строк attack => итого 2e6
    n_normal = len(df_normal)
    n_attack = len(df_attack)
    n_to_sample = min(n_normal, n_attack)  # = 1e6

    # Выбираем ровно 1 млн normal и 1 млн attack
    df_normal_s = df_normal.sample(n=n_to_sample, random_state=42)
    df_attack_s = df_attack.sample(n=n_to_sample, random_state=42)

    # Объединяем
    df_stage1 = pd.concat([df_normal_s, df_attack_s], ignore_index=True)
    # Перемешаем
    df_stage1 = df_stage1.sample(frac=1.0, random_state=42).reset_index(drop=True)
    # Теперь превращаем: label=0 => stage_label=0, label!=0 => 1
    df_stage1['stage_label'] = (df_stage1['label'] != 0).astype(int)
    df_stage1.drop(columns=['label'], inplace=True)
    df_stage1.rename(columns={'stage_label': 'label'}, inplace=True)
    stage1_path = os.path.join(output_dir, "stage1.csv")
    df_stage1.to_csv(stage1_path, index=False)

    # --- 2) Stage2 (2 млн, label!=0 => label=1 => DDoS, label=0 => не-DDoS) ---
    df_attack_full = df[df['label'] != 0].copy()  # это ~ 2 млн
    # Среди них: orig.label=1 => не-DDoS, orig.label=2..11 => DDoS
    df_attack_full['stage_label'] = (df_attack_full['label'] != 1).astype(int)
    df_attack_full.drop(columns=['label'], inplace=True)
    df_attack_full.rename(columns={'stage_label': 'label'}, inplace=True)
    stage2_path = os.path.join(output_dir, "stage2.csv")
    df_attack_full.to_csv(stage2_path, index=False)

    # --- 3) Stage3 (1 млн, label=2..11 => ставим 0..9) ---
    df_ddos = df[df['label'] >= 2].copy()  # =1 млн
    df_ddos['stage_label'] = df_ddos['label'] - 2
    df_ddos.drop(columns=['label'], inplace=True)
    df_ddos.rename(columns={'stage_label': 'label'}, inplace=True)
    stage3_path = os.path.join(output_dir, "stage3.csv")
    df_ddos.to_csv(stage3_path, index=False)

    print(f"[create_stage_datasets_balanced] stage1: {df_stage1.shape}, stage2: {df_attack_full.shape}, stage3: {df_ddos.shape}")
    return stage1_path, stage2_path, stage3_path


# ---------------------------------------------------------------------
# Класс FederatedLearning — с подробным логированием на каждой эпохе устройства
# ---------------------------------------------------------------------

class FederatedLearning:
    def __init__(self, data_file, label_col, model_type="DNN", seed=42, experiment_dir=None):
        self.SEED = seed
        self.DATA_FILE = data_file
        self.LABEL_COL = label_col
        self.MODEL_TYPE = model_type

        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 0.001
        self.EPOCHS_PER_CLIENT = 8
        self.NUM_ROUNDS = 5
        self.DEVICE_COUNTS = [100, 200, 300]
        self.MANUAL_DATA_SIZE = 0.001

        self.data = None
        self.big_train_data = None
        self.manual_test_data = None
        self.global_model = None
        self.global_weights = None

        random.seed(self.SEED)
        np.random.seed(self.SEED)
        tf.random.set_seed(self.SEED)

        self._configure_gpu()

        # Если experiment_dir=None => создаём новую папку exp_N
        if experiment_dir is None:
            self.experiment_dir = self._create_experiment_directory()
        else:
            self.experiment_dir = experiment_dir
            os.makedirs(self.experiment_dir, exist_ok=True)

        log_file = os.path.join(self.experiment_dir, "federated_training.log")
        logging.basicConfig(
            filename=log_file,
            filemode="a",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)
        msg = f"FederatedLearning init. Data={data_file}, exp_dir={self.experiment_dir}"
        print(msg)
        self.logger.info(msg)

    def _configure_gpu(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.set_visible_devices(gpus[0], 'GPU')
            except Exception as e:
                print(f"Error GPU config: {e}")

    def _create_experiment_directory(self):
        base_dir = "experiments"
        os.makedirs(base_dir, exist_ok=True)
        existing_dirs = [d for d in os.listdir(base_dir) if d.startswith("exp_")]
        exp_nums = []
        for d in existing_dirs:
            try:
                num = int(d.split("_")[1])
                exp_nums.append(num)
            except:
                pass
        nxt = max(exp_nums, default=0) + 1
        experiment_dir = os.path.join(base_dir, f"exp_{nxt}")
        os.makedirs(experiment_dir, exist_ok=True)
        print(f"[FederatedLearning] Created: {experiment_dir}")
        return experiment_dir

    def load_and_preprocess_data(self):
        if not os.path.exists(self.DATA_FILE):
            raise FileNotFoundError(f"File {self.DATA_FILE} not found!")

        df = pd.read_csv(self.DATA_FILE)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        if self.LABEL_COL not in df.columns:
            raise ValueError("No label column!")

        if df[self.LABEL_COL].dtype == object:
            label_encoder = LabelEncoder()
            df[self.LABEL_COL] = label_encoder.fit_transform(df[self.LABEL_COL])
            joblib.dump(label_encoder, os.path.join(self.experiment_dir, "label_encoder.pkl"))

        self.big_train_data, self.manual_test_data = train_test_split(
            df, test_size=self.MANUAL_DATA_SIZE, random_state=self.SEED
        )

        ds_dir = os.path.join(self.experiment_dir, "stage_datasets")
        os.makedirs(ds_dir, exist_ok=True)
        test_file = os.path.join(ds_dir, f"manual_test_{os.path.basename(self.DATA_FILE)}")
        self.manual_test_data.to_csv(test_file, index=False)

        self.data = df

        self.logger.info(f"[load_and_preprocess_data] file={self.DATA_FILE}, total={len(df)} train={len(self.big_train_data)} test={len(self.manual_test_data)}")

    def create_model(self, input_dim, num_classes):
        if self.MODEL_TYPE == "DNN":
            return self._create_dnn_model(input_dim, num_classes)
        elif self.MODEL_TYPE == "CNN_BiLSTM":
            return self._create_cnn_bilstm_model(input_dim, num_classes)
        else:
            raise ValueError(f"Unknown model {self.MODEL_TYPE}")

    def _create_dnn_model(self, input_dim, num_classes):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        opt = tf.keras.optimizers.Adam(self.LEARNING_RATE)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def _create_cnn_bilstm_model(self, input_dim, num_classes):
        model = tf.keras.Sequential([
            tf.keras.layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
            tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.2)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, dropout=0.2)),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        opt = tf.keras.optimizers.Adam(self.LEARNING_RATE)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def split_for_devices(self, train_data, num_devices, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        rows_per_device = len(train_data) // num_devices
        files = []
        for i in range(num_devices):
            st = i * rows_per_device
            end = st + rows_per_device if i != num_devices - 1 else len(train_data)
            subset = train_data.iloc[st:end]
            fn = os.path.join(folder_path, f"device_{i+1}.csv")
            subset.to_csv(fn, index=False)
            files.append(fn)
        return files

    def train_local_model(self, file_path, global_weights, client_id, num_classes):
        """
        Обучаем модель на данных одного клиента, подробно логируем каждую эпоху.
        """
        df_local = pd.read_csv(file_path)
        if self.LABEL_COL not in df_local.columns:
            return None, None, None

        df_tr, df_val = train_test_split(df_local, test_size=0.2, random_state=self.SEED)
        X_train = df_tr.drop(columns=[self.LABEL_COL]).values
        y_train = df_tr[self.LABEL_COL].values
        X_val = df_val.drop(columns=[self.LABEL_COL]).values
        y_val = df_val[self.LABEL_COL].values

        input_dim = X_train.shape[1]
        model = self.create_model(input_dim, num_classes)
        model.set_weights(global_weights)

        ds_tr = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
                              .shuffle(4096, seed=self.SEED)\
                              .batch(self.BATCH_SIZE)\
                              .prefetch(tf.data.AUTOTUNE)
        ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))\
                              .batch(self.BATCH_SIZE)\
                              .prefetch(tf.data.AUTOTUNE)

        with tqdm(total=self.EPOCHS_PER_CLIENT, desc=f"Device {client_id}", unit="epoch") as pb:
            for epoch_i in range(self.EPOCHS_PER_CLIENT):
                history = model.fit(ds_tr, validation_data=ds_val, epochs=1, verbose=0)
                
                # Извлекаем метрики обучения и валидации
                train_loss = history.history['loss'][-1]
                train_acc = history.history['accuracy'][-1]
                val_loss = history.history['val_loss'][-1]
                val_acc = history.history['val_accuracy'][-1]

                # Обновляем прогресс-бар
                pb.update(1)
                pb.set_postfix({
                    "train_loss": f"{train_loss:.4f}",
                    "train_acc": f"{train_acc:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "val_acc": f"{val_acc:.4f}"
                })

                # Пишем детальный лог
                self.logger.info(
                    f"[Device={client_id}, Epoch={epoch_i+1}/{self.EPOCHS_PER_CLIENT}] "
                    f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )

        # Итоговая валидация (после всех эпох)
        final_val_loss, final_val_acc = model.evaluate(ds_val, verbose=0)
        self.logger.info(
            f"[Device={client_id}] Final val_loss={final_val_loss:.4f}, val_acc={final_val_acc:.4f}"
        )

        up_weights = model.get_weights()

        # Освобождаем память
        del df_local, df_tr, df_val, X_train, y_train, X_val, y_val, ds_tr, ds_val, model
        tf.keras.backend.clear_session()
        gc.collect()

        return up_weights, final_val_loss, final_val_acc

    def federated_training(self, stage_name=""):
        """
        Запускаем федеративное обучение. Для каждого количества устройств 
        (из self.DEVICE_COUNTS) и для каждого раунда (self.NUM_ROUNDS).
        """
        results = {}
        num_classes = len(self.big_train_data[self.LABEL_COL].unique())
        self.logger.info(f"[federated_training] stage={stage_name}, num_classes={num_classes}")

        for devcount in self.DEVICE_COUNTS:
            device_folder = os.path.join(self.experiment_dir, f"devices_{stage_name}_{devcount}")
            client_files = self.split_for_devices(self.big_train_data, devcount, device_folder)

            rounds_loss = []
            rounds_acc = []
            self.global_model = None
            self.global_weights = None

            for round_idx in range(self.NUM_ROUNDS):
                local_weights_list = []
                local_losses = []
                local_accuracies = []

                self.logger.info(f"--- Round {round_idx+1}/{self.NUM_ROUNDS}, devices={devcount} ---")

                for cid, cf in enumerate(client_files, start=1):
                    if self.global_weights is None:
                        # инициализация
                        df_sm = pd.read_csv(cf, nrows=1)
                        X_sm = df_sm.drop(columns=[self.LABEL_COL]).values
                        inp_dim = X_sm.shape[1]
                        self.global_model = self.create_model(inp_dim, num_classes)
                        self.global_weights = self.global_model.get_weights()

                    upd_w, l, a = self.train_local_model(cf, self.global_weights, cid, num_classes)
                    if upd_w is not None:
                        local_weights_list.append(upd_w)
                        local_losses.append(l)
                        local_accuracies.append(a)

                # Усредняем веса
                if local_weights_list:
                    new_g = []
                    for layer_i in range(len(local_weights_list[0])):
                        arr = np.array([w[layer_i] for w in local_weights_list])
                        new_g.append(arr.mean(axis=0))
                    self.global_weights = new_g

                # Средние метрики по устройствам
                if local_losses:
                    avg_loss = np.mean(local_losses)
                    avg_acc = np.mean(local_accuracies)
                    rounds_loss.append(avg_loss)
                    rounds_acc.append(avg_acc)
                    self.logger.info(
                        f"[Round={round_idx+1}, devices={devcount}] "
                        f"AVG val_loss={avg_loss:.4f}, AVG val_acc={avg_acc:.4f}"
                    )
                
            results[devcount] = {'loss': rounds_loss, 'acc': rounds_acc}

            # Сохраняем финальную модель
            if self.global_weights is not None:
                if self.global_model is None:
                    # создадим модель
                    sample_x = self.big_train_data.drop(columns=[self.LABEL_COL]).values[:1]
                    inp_dim = sample_x.shape[1]
                    self.global_model = self.create_model(inp_dim, num_classes)
                self.global_model.set_weights(self.global_weights)
                fn = os.path.join(self.experiment_dir, f"federated_global_model_{stage_name}_{devcount}.h5")
                self.global_model.save(fn)
                self.logger.info(f"Saved => {fn}")

        return results

    def plot_results(self, results, stage_name=""):
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        for dev,met in results.items():
            if len(met['loss'])==0:
                continue
            x = range(1,len(met['loss'])+1)
            plt.plot(x, met['loss'], marker='o', label=f"{dev} devices")
        plt.title(f"Loss [{stage_name}]")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()

        plt.subplot(1,2,2)
        for dev,met in results.items():
            if len(met['acc'])==0:
                continue
            x = range(1,len(met['acc'])+1)
            plt.plot(x, met['acc'], marker='o', label=f"{dev} devices")
        plt.title(f"Accuracy [{stage_name}]")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()

        fn_plot = os.path.join(self.experiment_dir, f"federated_results_plot_{stage_name}.png")
        plt.tight_layout()
        plt.savefig(fn_plot)
        plt.show()


# ---------------------------------------------------------------------
# ОСНОВНОЙ СКРИПТ
# ---------------------------------------------------------------------

if __name__ == "__main__":
    original_csv = "dataset_container/testing_data/dataset_v1.csv"
 
    
    fl_stage1 = FederatedLearning(
        data_file=original_csv,
        label_col="label",
        model_type="CNN_BiLSTM",
        seed=42,
        experiment_dir=None
    )
    fl_stage1.load_and_preprocess_data()
    res1 = fl_stage1.federated_training(stage_name="stage1")
    fl_stage1.plot_results(res1, stage_name="stage1")
 