import pandas as pd
import sys

if len(sys.argv) != 2:
    print("Usage: python remove.py <label>")
    sys.exit(1)

label_to_filter = int(sys.argv[1])

# Замените 'input.csv' на название вашего файла
input_file = 'dataset_container/2023data/dataset_balanced.csv'
output_file = 'dataset_container/2023data/dataset_balanced1.csv'

# Загрузка данных
df = pd.read_csv(input_file)

# Разделяем данные по меткам
df_label = df[df['label'] == label_to_filter]
df_other = df[df['label'] != label_to_filter]  # Все остальные метки

# Проверяем количество данных в каждой метке
print(f"Количество строк с label={label_to_filter}: {len(df_label)}")

# Удаляем 750000 случайных строк из каждой метки (если столько есть)
if len(df_label) > 1000000:
    df_label = df_label.sample(n=len(df_label) - 1000000, random_state=42)
else:
    print("Недостаточно данных для удаления будет удалено все")
    df_label = df_label.sample(n=0, random_state=42)

# Объединяем обратно
df_filtered = pd.concat([df_label, df_other])

# Сохраняем результат
df_filtered.to_csv(output_file, index=False)

print(f"Фильтрованный файл сохранен как {output_file}")
