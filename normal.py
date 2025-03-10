import pandas as pd
import numpy as np

def normalize_dataset(df: pd.DataFrame):
    """
    Нормализует данные:
    - Колонку 'Rate' с помощью логарифмического преобразования log(1 + x)
    - Остальные числовые колонки с помощью min-max нормализации

    :param df: DataFrame с данными
    :return: DataFrame с нормализованными значениями
    """
    df_normalized = df.copy()

    # Логарифмическое преобразование для 'Rate'
    if "Rate" in df_normalized.columns:
        df_normalized["Rate"] = np.log1p(df_normalized["Rate"])
    # # Логарифмическое преобразование для 'Flow Packets/s'
    # if "Flow Packets/s" in df_normalized.columns:
    #     df_normalized["Flow Packets/s"] = np.log1p(df_normalized["Flow Packets/s"])
    
    # Min-Max нормализация для остальных числовых колонок (кроме 'Rate')
    numeric_columns = df_normalized.select_dtypes(include=['number']).columns
    for column in numeric_columns:
        if column != "label":  # Исключаем 'label'
            min_val = df_normalized[column].min()
            max_val = df_normalized[column].max()
            if max_val - min_val > 0:  # Проверка деления на 0
                df_normalized[column] = (df_normalized[column] - min_val) / (max_val - min_val)

    return df_normalized

# Загружаем данные
file_path = "dataset_container/2023data/dataset1.csv"  # Используем исправленный файл
df = pd.read_csv(file_path)

# Применяем нормализацию
df_normalized = normalize_dataset(df)

# Проверяем результат
print(df_normalized.describe())

# Сохраняем нормализованные данные в новый CSV
df_normalized.to_csv("dataset_container/2023data/dataset1.csv", index=False)
print("\nНормализованный датасет сохранен в 'data_normalized.csv'")
