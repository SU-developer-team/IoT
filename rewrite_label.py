import pandas as pd

# Загрузка данных
df = pd.read_csv('dataset_container/dataset/data_normalized.csv')

# Замена значений в столбце 'label'
df['label'] = df['label'].apply(lambda x: min(x, 2))

# Сохранение результата
df.to_csv('dataset_container/dataset/3class_normalized.csv', index=False)