import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib  # для загрузки label_encoder.pkl

def main():
    # 1. Загружаем ранее обученную модель
    model_path = "federated_global_model_15.h5"   # Путь к модели
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Не найдена модель по пути: {model_path}")
    
    print(f"Загружаем модель: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # 1.1 Загружаем LabelEncoder (если у вас есть файл label_encoder.pkl)
    encoder_path = "federated_label_encoder.pkl"  # <-- или путь, который вы указали при сохранении
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Не найден LabelEncoder по пути: {encoder_path}")

    label_encoder = joblib.load(encoder_path)
    print("LabelEncoder загружен.")

    # 2. Считываем тестовые данные (CSV)
    test_csv = "2017data/man.csv"  # <-- Ваш CSV для инференса
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Не найден файл {test_csv}")

    print(f"Считываем данные из: {test_csv}")
    df = pd.read_csv(test_csv)

    # 3. Предобработка: убираем ±inf и пропуски
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Если в тестовом наборе есть столбец Label — уберём, чтобы модель его не использовала
    if "Label" in df.columns:
        df.drop(columns=["Label"], inplace=True)

    # ВАЖНО: повторяем те же преобразования признаков, которые делали на обучении!
    # Допустим, при обучении вы просто делали: X = data.drop(columns=["Label"]).values
    # Тогда здесь:
    X = df.values

    # 4. Прогоняем данные через модель
    print("Выполняем предсказание...")
    pred_probs = model.predict(X)  # форма (num_samples, num_classes)

    # Ищем индекс класса с максимальной вероятностью
    pred_classes = np.argmax(pred_probs, axis=1)  # (num_samples,)
    # Ищем саму максимальную вероятность (уже для этого класса)
    pred_confidence = np.max(pred_probs, axis=1)  # (num_samples,)

    # Преобразуем индексы обратно в текст (например, "BENIGN", "DDoS", ...)
    pred_labels_text = label_encoder.inverse_transform(pred_classes)

    # 5. Записываем результат в исходный DataFrame
    # (можно сохранить исходные признаки + prediction + confidence)
    df["prediction"] = pred_labels_text
    df["confidence"] = pred_confidence  # это "коэффициент уверенности" в выбранном классе

    # 6. Сохраняем в новый CSV
    out_csv = "test_data_with_preds.csv"
    df.to_csv(out_csv, index=False)
    print(f"Результаты сохранены в: {out_csv}")

if __name__ == "__main__":
    main()
