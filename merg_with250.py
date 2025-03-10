import os
import pandas as pd

# Папка с файлами
folder_path = "dataset_container/2023data"
output_file = "dataset_container/2023data/dataset.csv"

# Количество строк для выборки
rows_per_file = 250_000

# Список для хранения частей данных
dataframes = []

# Проход по всем CSV-файлам в папке
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):  # Фильтруем только CSV
        file_path = os.path.join(folder_path, file_name)
        
        # Читаем первые 250000 строк
        df = pd.read_csv(file_path, nrows=rows_per_file)
        
        # Добавляем столбец с названием файла (для идентификации источника)
        # df["source_file"] = file_name
        
        # Добавляем в список
        dataframes.append(df)

# Объединяем все DataFrame
merged_df = pd.concat(dataframes, ignore_index=True)

# Сохраняем в единый CSV
merged_df.to_csv(output_file, index=False)

print(f"Объединенный файл сохранен: {output_file}")
